import torch
from torch.utils.data.dataloader import DataLoader

from pathlib import Path
import json
import logging

from typing import Union

import numpy as np

from src.experiments.generation.dataset import _unpatch_lc, _patch_lc_batch
from src.utils.preprocessing import Preprocessing
from src.experiments.generation.network.modeling import L3MGenerator

from pathlib import Path

import torch.multiprocessing as mp

@torch.inference_mode()
def evaluate_generation(
        test_dataset: DataLoader,
        model: L3MGenerator,
        output_dir: str,
        eval_t_max: int,
        eval_t_init: int,
        train_t_max: int,
        train_t_init: int,
        patch_shape: tuple[int, int, int],
        lc_preprocessor: Preprocessing = None,
):
    """
    Generates new lightcone slices conditioned on eval_t_init initial slices.
    
    The generated sublightcones can be longer than the ones the model has been trained on, eval_t_max > model_t_max.
    This is accomplished by iteratively predicting model_t_max-eval_t_init slices.

    Arguments:
        test_dataset (`torch.utils.data.dataloader.DataLoader`):
            Dataloader yielding the evaluation sublightcones consisting of eval_t_max slices.
        model (`src.experiments.generation.network.modeling.L3MGenerator`):
            The model is evaluated.
        output_dir (`str`):
            The folder where the generated slices are saved.
        eval_t_max (`int`):
            The number of slices an evaluation sublightcone consists of.

            It must satisfy `eval_t_max > eval_t_init`.
        eval_t_init (`int`):
            The number of initial slices that the generation is condition on. 
            
            It must satisfy `eval_t_max > eval_t_init` and `eval_t_init >= train_t_init`.
        train_t_max (`int`):
            The number of slices a training sublightcone consists of.
        train_t_init (`int`):
            The number of initial slices that the generation was condition on during training. 
        patch_shape (`tuple[int, int, int]`):
            The patch shape of the generated sublightcone of the form (time, x, y).
        lc_preprocessor (`src.utils.preprocessing.Preprocessing`, defaults to `None`):
            The preprocessor instance to postprocess the lightcone values. If `None`, no postprocess is applied.
    """

    assert eval_t_max > eval_t_init, f"eval_t_max > eval_t_init must be satified. Specified values: eval_t_max={eval_t_max}, eval_t_init={eval_t_init}"
    assert eval_t_init >= train_t_init, f"eval_t_max >= train_t_init must be satified. Specified values: eval_t_max={eval_t_max}, train_t_init={train_t_init}"

    data_queue = mp.Queue(test_dataset.batch_size)
    write_data_worker = mp.Process(target=save_thread, args=(output_dir, data_queue), daemon=True)
    write_data_worker.start()

    model.eval()

    delta_target = eval_t_max - eval_t_init
    delta_forecasted = train_t_max - eval_t_init

    l2s = {}

    for batch in iter(test_dataset):
        for k, v in batch.items():
            batch[k] = v.to(device="cuda")
        
        batch_size = batch["input_ids"].size(0)
        _lc_shape = batch["lc_shape"].cpu().tolist()

        lcs_init = batch["lcs_init"]
        temp_lcs_init = lcs_init.clone()

        lcs_remaining = batch["lcs_remaining"]

        lcs_pred = torch.zeros(
            batch_size,
            delta_target*patch_shape[0],
            _lc_shape[1]*patch_shape[1],
            _lc_shape[2]*patch_shape[2],
            dtype=lcs_remaining.dtype,
            device=lcs_remaining.device
        )

        t_indices = batch["t_indices"]
        temp_t_indices = t_indices.clone()
        
        t_start = 0
        while t_start < delta_target:
            if t_start > 0:
                _temp_lcs_init = []
                if delta_t < eval_t_init:
                    _temp_lcs_init.append(temp_lcs_init[:, delta_t:])
                _keep_pred_lcs = min(delta_t, eval_t_init)
                _temp_lcs_init.append(_patch_lc_batch(lcs_pred[:, t_start-_keep_pred_lcs:t_start], patch_shape)[0])
                temp_lcs_init = torch.cat(_temp_lcs_init, dim=1)

            batch["lcs_init"] = temp_lcs_init

            batch["t_indices"] = temp_t_indices + t_start
            
            delta_t = min(delta_forecasted, delta_target-t_start)
            batch["t_max"] = torch.tensor([delta_t+eval_t_init])

            res = model.generate(**batch)

            lcs_pred[:, t_start:t_start+delta_t] = res["lcs_pred"]

            t_start += delta_t

        lc_ids = res["ids"].float().cpu().numpy()

        lcs_init = lcs_init.reshape(batch_size, eval_t_init, *_lc_shape[1:3], *patch_shape).permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch_size,
            eval_t_init*patch_shape[0],
            _lc_shape[1]*patch_shape[1],
            _lc_shape[2]*patch_shape[2],
        )
        lcs_remaining = lcs_remaining.reshape(batch_size, delta_target, *_lc_shape[1:3], *patch_shape).permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch_size,
            delta_target*patch_shape[0],
            _lc_shape[1]*patch_shape[1],
            _lc_shape[2]*patch_shape[2],
        )

        _l2 = lcs_remaining - lcs_pred[..., -1]
        _l2 = _l2 * _l2
        _l2 = torch.mean(_l2, dim=(1,2,3))

        if lc_preprocessor != None:
            lcs_init = lc_preprocessor.reverse(lcs_init)
            lcs_remaining = lc_preprocessor.reverse(lcs_remaining)
            lcs_pred = lc_preprocessor.reverse(lcs_pred)

        lcs_init = lcs_init.float().cpu().numpy().astype("float16")
        lcs_remaining = lcs_remaining.float().cpu().numpy().astype("float16")
        lcs_pred = lcs_pred.float().cpu().numpy().astype("float16")

        for i in range(lc_ids.size):
            _id = int(lc_ids[i])
            _t = int(t_indices[i])

            if _id in l2s:
                l2s[_id].append(_l2[i])
            else:
                l2s[_id] = [_l2[i]]

            data = {
                "lc_id": _id,
                "t_index": _t,
                "init_lc": lcs_init[i],
                "remaining_lc": lcs_remaining[i],
                "pred_lc": lcs_pred[i],
            }

            data_queue.put(data)
        
    data_queue.put(None)

    for id in l2s:
        l2s[id] = torch.mean(torch.stack(l2s[id])).item()

    write_data_worker.join()

    with open(f"{output_dir}/l2.json", "w") as f:
        json.dump(l2s, f, indent=2)

@torch.inference_mode()
def evaluate_next_patch(
        test_dataset: DataLoader,
        model: L3MGenerator,
        output_dir,
        lc_preprocessor: Preprocessing = None
):
    """
    Generates new lightcone slices conditioned on the ground truth of the previous patches.
    
    Arguments:
        test_dataset (`torch.utils.data.dataloader.DataLoader`):
            Dataloader yielding the evaluation sublightcones consisting of eval_t_max slices.
        model (`src.experiments.generation.network.modeling.L3MGenerator`):
            The model is evaluated.
        output_dir (`str`):
            The folder where the generated slices are saved.
        lc_preprocessor (`src.utils.preprocessing.Preprocessing`, defaults to `None`):
            The preprocessor instance to postprocess the lightcone values. If `None`, no postprocess is applied.
    """

    data_queue = mp.Queue(test_dataset.batch_size)
    write_data_worker = mp.Process(target=save_thread, args=(output_dir, data_queue), daemon=True)
    write_data_worker.start()

    model.eval()

    l2s = {}

    for batch in iter(test_dataset):
        lc_ids = batch["lc_id"].float().cpu().numpy()
        t_indices = batch["t_indices"].float().cpu().numpy()
        batch_size = lc_ids.size

        batch.pop("labels", None)

        for k, v in batch.items():
            batch[k] = v.to(device="cuda")
        res = model(**batch, use_cache=False)

        bt_vals_pred = model.lc_head.sample(res["logits"])
        _lcs_pred = bt_vals_pred

        t_max = batch["t_max"][0].item()
        t_init = batch["t_init"][0].item()
        t_size = t_max - t_init
        lc_shape = batch["lc_shapes"][0].cpu().tolist()

        patch_shape = batch["patch_shapes"][0].cpu().tolist()

        _lcs_pred = _lcs_pred.reshape(batch_size, t_size, -1, *lc_shape[1:3], model.config.lc_patch_dim)[:batch_size]
        
        lc_values = batch["lc_values"].reshape(batch_size, -1, batch["lc_values"].size(-1))

        lcs_init = []
        lcs_pred = []
        lcs_remaining = []

        for i in range(lc_ids.size):
            _lc_values = lc_values[i].reshape(*lc_shape)
            _lc_init = _lc_values[:t_init]
            _lc_remaining = _lc_values[t_init:]

            lcs_init.append(_unpatch_lc(_lc_init, patch_shape))
            lcs_remaining.append(_unpatch_lc(_lc_remaining, patch_shape))

            _lc_pred = _lcs_pred[i].reshape(t_max - t_init, *lc_shape[1:])

            lcs_pred.append(_unpatch_lc(_lc_pred, patch_shape))

        lcs_init = torch.stack(lcs_init)
        lcs_pred = torch.stack(lcs_pred)
        lcs_remaining = torch.stack(lcs_remaining)

        _l2 = lcs_remaining - lcs_pred[..., -1]
        _l2 = _l2 * _l2
        _l2 = torch.mean(_l2, dim=(1,2,3))

        if lc_preprocessor != None:
            lcs_init = lc_preprocessor.reverse(lcs_init)
            lcs_remaining = lc_preprocessor.reverse(lcs_remaining)
            lcs_pred = lc_preprocessor.reverse(lcs_pred)

        lcs_init = lcs_init.float().cpu().numpy().astype("float16")
        lcs_remaining = lcs_remaining.float().cpu().numpy().astype("float16")
        lcs_pred = lcs_pred.float().cpu().numpy().astype("float16")

        for i in range(lc_ids.size):
            _id = int(lc_ids[i])
            _t = int(t_indices[i])

            if _id in l2s:
                l2s[_id].append(_l2[i])
            else:
                l2s[_id] = [_l2[i]]

            data = {
                "lc_id": _id,
                "t_index": _t,
                "init_lc": lcs_init[i],
                "remaining_lc": lcs_remaining[i],
                "pred_lc": lcs_pred[i]
            }
            data_queue.put(data)
            
    data_queue.put(None)

    for id in l2s:
        l2s[id] = torch.mean(torch.stack(l2s[id])).item()

    write_data_worker.join()

    with open(f"{output_dir}/l2.json", "w") as f:
        json.dump(l2s, f, indent=2)
   
@torch.inference_mode()
def evaluate_next_patch_l2(
        test_dataset: DataLoader,
        model: L3MGenerator,
        output_dir: str,
        n_samples: int = 10,
        logger: logging.Logger = None
):
    """
    Computes the L2 loss of generated lightcone patches where each generated patch is conditioned on the ground truth of the previous patches.
    
    Arguments:
        test_dataset (`torch.utils.data.dataloader.DataLoader`):
            Dataloader yielding the evaluation sublightcones consisting of eval_t_max slices.
        model (`src.experiments.generation.network.modeling.L3MGenerator`):
            The model is evaluated.
        output_dir (`str`):
            The folder where the generated slices are saved.
        n_samples (`int`, defaults to 10):
            The number of samples to generate per patch.
        lc_preprocessor (`src.utils.preprocessing.Preprocessing`, defaults to `None`):
            The preprocessor instance to postprocess the lightcone values. If `None`, no postprocess is applied.
        logger (`logging.Logger`, defaults to `None`):
            If specified, the progress is logged.
    """

    model.eval()

    l2 = None
    n_batches = 0

    if logger != None:
        logger.info(f"evaluate_next_patch_l2 has {len(test_dataset)} batches")

    for batch in iter(test_dataset):
        batch_size = batch["lc_id"].shape[0]

        for k, v in batch.items():
            batch[k] = v.to(device="cuda")

        labels = batch.pop("labels", None).reshape(batch_size, -1, model.lc_head.lc_patch_dim).unsqueeze(0).expand(n_samples, -1, -1, -1)

        res = model(**batch, use_cache=False)
        
        logits = res["logits"]
        bt_vals_pred = model.lc_head.sample(logits, n_samples=n_samples).reshape(n_samples, batch_size, -1, model.lc_head.lc_patch_dim)

        _l2 = (labels - bt_vals_pred)**2
        _l2 = _l2.mean(dim=(0, 1, 3))

        if l2 == None:
            l2 = _l2
        else:
            l2 = l2 + _l2

        n_batches += 1

        if logger != None:
            if n_batches % 10 == 0:
                logger.info(f"Evaluated batch {n_batches}")
        
    l2 = l2 / n_batches
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(l2, f"{output_dir}/l2.pt")

def save_thread(output_dir, data_queue):
    while True:
        data = data_queue.get()
        if data == None:
            break
        save_lc(output_dir, **data)

def save_lc(output_dir, lc_id, t_index, **data):
    lc_dir = f"{output_dir}/{lc_id:03d}"
    out_file = f"{lc_dir}/{t_index:04d}.npz"

    Path(lc_dir).mkdir(parents=True, exist_ok=True)

    np.savez(
        out_file,
        **data
    )
