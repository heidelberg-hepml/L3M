import torch

import os
import logging
from glob import glob
import gc

from typing import Optional

def get_compute_metrics_torch(label_names):
    param_dim = len(label_names)

    @torch.inference_mode()
    def compute_metrics(values, labels):
        _params = values[:,:param_dim]
        res = {}
        for i, name in enumerate(label_names):
            res[f"l1/{name}"] = torch.mean(torch.abs(_params[:,i] - labels[:,i]))
        return res

    return compute_metrics

def _load_weights(model, model_keys, bin_file, logger, map_location=None, submodule:Optional[str]=None):
    _num_loaded_weights = 0

    if logger != None:
        logger.info(f"Loading weights from the bin file {bin_file}")

    pretrained_state_dict = torch.load(bin_file, weights_only=False, map_location=map_location)
    if submodule != None:
        pretrained_state_dict = { k.replace(f"{submodule}.", ""): v for k, v in pretrained_state_dict.items() if k.startswith(submodule) }
    compatible_modules = { k: v for k, v in pretrained_state_dict.items() if k in model_keys and 'drop' not in k }

    if logger != None:
        logger.info(f"Loading pretrained weights ({len(compatible_modules)} compatible modules)")
        logger.debug(f"Compatible modules:\n" + "\n".join([k for k in compatible_modules]))

    model.load_state_dict(compatible_modules, strict=False)

    _num_loaded_weights += len(compatible_modules)

    del pretrained_state_dict, compatible_modules

    return model, _num_loaded_weights

def load_weights(bin_dir: str, model: torch.nn.Module, logger: logging.Logger, map_location=None, submodule:Optional[str]=None):
    num_loaded_weights = 0

    model_keys = model.state_dict().keys()
    
    if logger != None:
        logger.info("Loading weights from pretrained model")

    bin_file = f"{bin_dir}/pytorch_model.bin"
    if logger != None:
        logger.info(f"Checking if the bin file {bin_file} exists.")
    if os.path.exists(bin_file):
        model, _num_loaded_weights = _load_weights(model=model, model_keys=model_keys, bin_file=bin_file, logger=logger, map_location=map_location, submodule=submodule)
        num_loaded_weights += _num_loaded_weights
    else:
        if logger != None:
            logger.info(f"Searching for shareded bin files.")
        for bin_file in glob(f"{bin_dir}/pytorch_model-*-of-*.bin"):
            model, _num_loaded_weights = _load_weights(model=model, model_keys=model_keys, bin_file=bin_file, logger=logger, map_location=map_location, submodule=submodule)
            num_loaded_weights += _num_loaded_weights

    assert num_loaded_weights > 0, "No weights were loaded"

    if logger != None:
        logger.info(f"{num_loaded_weights} preetrained weights have been loaded succesfully.")

    torch.cuda.empty_cache()
    gc.collect()

    return model
