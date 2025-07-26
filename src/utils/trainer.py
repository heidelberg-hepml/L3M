import torch

import os
from pathlib import Path
import math
import json
import shutil

import contextlib

import logging

from src.utils.training import load_weights

import gc

logger = logging.getLogger(__name__)

from transformers import __version__

from src.utils.scheduler import get_wsd_schedule
from src.utils.exp_config import ExpConfig

from torch.utils.tensorboard import SummaryWriter

import psutil

from typing import Optional

class ExpState:
    """
    Contains information about the current state of the experiment.
    """

    def __init__(self):
        self.epoch: int = 0
        self.total_step: int = 0
        self.epoch_step: int = 0
        self.best_epoch: int = 0
        self.best_checkpoint: str = None
        self.last_checkpoint: str = None
        self.best_eval_metric: float = None
        self.last_eval_metric: float = None
        self.log_step: int = 0
        self.last_eval_epoch: int = -1


    def save(self, output_dir: str):
        with open(os.path.join(output_dir, "exp_state.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, dir: str) -> "ExpState":
        with open(os.path.join(dir, "exp_state.json"), "r") as f:
            _exp_state = json.load(f)
        exp_state = ExpState()
        for k, v in _exp_state.items():
            exp_state.__dict__[k] = v
        return exp_state

class Trainer:
    """
    Trainer performs the training of the experiment.

    Arguments:
        model (torch.nn.Module):
            The network to train
        train_dataset (torch.utils.data.DataLoader):
            Training dataset wrapped by a DataLoader
        eval_dataset (torch.utils.data.DataLoader):
            Validation dataset by a DataLoader
        run_dir (str):
            Directory to store data of the run
        logger (logging.Logger):
            Logger
        batch_size (int):
            Batch size for training
        eval_batch_size (int):
            Batch size for validation
        gradient_accumulation_steps (int):
            Gradient accumulation steps
        num_epochs (int):
            Number of training epochs.
        exp_config (src.utils.exp_config.ExpConfig):
            Full config of the experiment.
        head (torch.nn.Module, defaults to `None`):
            Head of the network for the head dedicated training.
        opt_model (torch.nn.Module, defaults to `None`):
            Torch compiled model.
        opt_head (torch.nn.Module, defaults to `None`):
            Torch compiled head.
        optimizers (list[torch.optim.Optimizer], defaults to `None`):
            List of optimizers for the model. These optimizers can also train the head.
        optimizer_head (torch.optim.Optimizer, defaults to `None`):
            Optimizer for the head. 
        train_head (bool, defaults to `True`):
            Use dedicated head training.
        train_only_head (bool, defaults to `False`):
            Use only the dedicated head training.
        train_interleaved (bool, defaults to `False`):
            This setting determines the order in which the model training and the head training occur.
            `False` means that the model is trained for one whole epoch, after which the head is trained for one epoch,.
            `True` means that for each batch, fist the model is trained and then the head.
        epochs_until_eval (int, defaults to 1):
            Frequency for evaluation the network performance in epochs.
        lr (float, defaults to `5e-5`):
            Learning rate.
        weight_decay (float, defaults to 0):
            Weight decay.
        max_grad_norm (float, defaults to 0):
            Maximal gradient norm for gradient clipping. A non-positive value disables gradient clipping.
        warmup_epochs (int, defaults to 0):
            Number of warmup epochs.
        warmup_epochs (int, defaults to 0):
            Number of warmup epochs.
        decay_epochs (int, defaults to 0):
            Number of decay epochs.
        log_dir (str, defaults to `None`):
            Directory for logging files. If `None`, `run_dir\log` is used.
        log_steps (int, defaults to `None`):
        `   Steps after which the training loss is logged. If `None`, the loss is logged 10 times per epoch.
        checkpoints_dir (str, defaults to `None`):
            Directory to save the checkpoints. If `None`, `run_dir\checkpoints` is used.
        max_num_checkpoints (int, defaults to `None`):
            Number of maximal checkpoints to save. The best checkpoints is always saved. A value of -1 or None implies that all checkpoints are saved.
        fp16_mp (bool, defaults to `False`):
            Use fp16 mixed precision training.
        bp16_mp (bool, defaults to `False`):
            Use bf16 mixed precision training. If fp16_mp is toggled, this flag is ignored.
        compute_metrics (defaults to `None`)::
            Custom metrics used for validation.
        force_model_saving (bool, defaults to `False`):
            If toggled, the last checkpoints is always saved.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: torch.utils.data.DataLoader,
            eval_dataset: torch.utils.data.DataLoader,
            run_dir: str,
            logger: logging.Logger,
            batch_size: int,
            eval_batch_size: int,
            gradient_accumulation_steps: int,
            num_epochs: int,
            exp_config: ExpConfig,
            head: Optional[torch.nn.Module] = None,
            opt_model: Optional[torch.nn.Module] = None,
            opt_head: Optional[torch.nn.Module] = None,
            optimizers: Optional[list[torch.optim.Optimizer]] = None,
            optimizer_head: Optional[torch.optim.Optimizer] = None,
            train_head: bool = True,
            train_only_head = False,
            train_interleaved: bool = False,
            epochs_until_eval: int = 1,
            lr: float = 5.e-5,
            weight_decay: float = 0.,
            max_grad_norm: float = 0,
            warmup_epochs: int = 0,
            decay_epochs: int = 0,
            log_dir: Optional[str] = None,
            log_steps: Optional[int] = None,
            checkpoints_dir: str = None,
            max_num_checkpoints = None,
            fp16_mp: bool = False,
            bf16_mp: bool = False,
            compute_metrics = None,
            force_model_saving: bool = False,
    ):
        self._model = model
        self._head = head
        self.model = opt_model if opt_model != None else model
        self.head = opt_head if opt_head != None else head

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.run_dir = run_dir

        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.actual_batch_size = self.batch_size * self.gradient_accumulation_steps
        assert self.actual_batch_size <= len(self.train_dataset.dataset), f"Actual batch size: {self.actual_batch_size}, train dataset size: {len(self.train_dataset.dataset)}"

        self.num_epochs = num_epochs
        self.epochs_until_eval = epochs_until_eval

        self.steps_per_epoch = math.floor(len(train_dataset.dataset) / self.actual_batch_size) if self.train_dataset.drop_last \
            else math.ceil(len(train_dataset.dataset) / self.actual_batch_size)
        self.total_steps_per_epoch = self.steps_per_epoch * self.gradient_accumulation_steps

        self.eval_batch_size = eval_batch_size
        self.total_eval_steps = math.floor(len(eval_dataset.dataset) / self.eval_batch_size) if self.eval_dataset.drop_last \
            else math.ceil(len(eval_dataset.dataset) / self.eval_batch_size)

        self.log_dir = log_dir if log_dir != None else os.path.join(run_dir, "log")
        self.checkpoints_dir = checkpoints_dir if checkpoints_dir != None else os.path.join(run_dir, "checkpoints")
        self.model_dir = os.path.join(run_dir, "model")
        
        Path(self.run_dir).mkdir(exist_ok=True, parents=True)
        Path(self.log_dir).mkdir(exist_ok=True, parents=True)
        Path(self.checkpoints_dir).mkdir(exist_ok=True, parents=True)
        Path(self.model_dir).mkdir(exist_ok=True, parents=True)

        self.logger = logger

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.clip_grad_norm = self.max_grad_norm > 0.

        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs    
        self.stable_epochs = self.num_epochs - self.warmup_epochs - self.decay_epochs

        self.log_steps = log_steps if log_steps != None else math.floor(len(train_dataset.dataset) / 10 / self.actual_batch_size)
        self.max_num_checkpoints = max_num_checkpoints if max_num_checkpoints != None else -1

        self.exp_config = exp_config
        self.exp_state: ExpState = ExpState()

        self.optimizers: list[torch.optim.Optimizer] = optimizers
        self.optimizer_head: torch.optim.Optimizer = optimizer_head
        self.optimizer_params = None

        self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = None

        self.model_call_kwargs: list[dict] = None

        self.compute_metrics = compute_metrics

        self.train_head = train_head
        self.train_only_head = train_only_head
        if self.train_only_head:
            self.train_head = True
        self.train_interleaved = train_interleaved
        
        self.setup_optimizer()
        self.setup_schedular()

        self.grad_scaler = None

        self.fp16_mp = fp16_mp
        if self.fp16_mp:
            logger.info("FP16 mixed precision is active")

            self.setup_fp16_mp()

            self.bf16_mp = False
        else:
            if bf16_mp:
                logger.info("BF16 mixed precision is active")

            self.bf16_mp = bf16_mp
    
        self.summarizer = SummaryWriter(log_dir)

        self.train_loss = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        self.gradient_acc_pt = torch.tensor(float(gradient_accumulation_steps), dtype=torch.float32, device="cuda")
        self._grad_norm = torch.tensor(0.0, dtype=torch.float32, device="cuda")

        self.save_full_model = False

        self.force_model_saving = force_model_saving

    def setup_optimizer(self):
        "Sets up the optimizers"

        if self.train_only_head:
            self.optimizers = []
        else:
            if self.optimizers == None:
                self.optimizers = [
                        torch.optim.AdamW(
                        self.model.named_parameters(),
                        lr=self.lr,
                        weight_decay=self.weight_decay,
                        betas=(0.9, 0.999)
                    )
                ]
            self.model_call_kwargs = [{} for _ in self.optimizers]
        if self.train_head:
            if self.optimizer_head == None:
                assert self.head != None
                self.optimizer_head = torch.optim.AdamW(
                        self.head.named_parameters(),
                        lr=self.lr_head,
                        weight_decay=self.weight_decay,
                        betas=(0.9, 0.999)
                    )
            self.optimizers.append(self.optimizer_head)
            self.model_call_kwargs.append({"detach_for_head": True})

        self.logger.info(f"Number of optimizers: {len(self.optimizers)}.")

        _logging_text = ""

        self.optimizer_params = []
        for i, optimizer in enumerate(self.optimizers):
            _opt_params = []
            _logging_text += f"Optimizer {i}:\n"
            for j, param_group in enumerate(optimizer.param_groups):
                _opt_params += param_group["params"]
                _logging_text += f"Parameter group {j}:\n"
                _logging_text += f"Learning rate: {param_group['lr']}\n"
                _logging_text += f"Weight decay: {param_group['weight_decay']}\n"
                _logging_text += f"Params: {len(param_group['params'])}\n"
                if param_group.get("param_names", None) != None:
                    _logging_text += f"Param names: {json.dumps([str(p) for p in param_group['param_names']])}\n"
                else:
                    _logging_text += f"Param names: not available\n"
            self.optimizer_params.append(_opt_params)

        self.logger.info(_logging_text)


    def setup_schedular(self):
        "Sets up the learning rate scheduler"

        self.schedulers = [
            get_wsd_schedule(
                optimizer=optimizer,
                num_warmup_steps=self.warmup_epochs*self.steps_per_epoch,
                num_stable_steps=self.stable_epochs*self.steps_per_epoch,
                num_decay_steps=self.decay_epochs*self.steps_per_epoch
                )
            for optimizer in self.optimizers
        ]

    def setup_fp16_mp(self):
        self.grad_scaler = torch.GradScaler()

    def get_train_context(self):       
        if self.fp16_mp:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        elif self.bf16_mp:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            return contextlib.nullcontext()

    def train(self):
        "Start the training"

        parameters_overview = {
            "total": sum(param.numel() for param in self.model.parameters()),
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        parameters_overview["frac"] = parameters_overview["trainable"] / parameters_overview["total"]

        self.log_training_setup(parameters_overview)

        logger.info("***** Running training *****")
        logger.info(f"  Training dataset size = {len(self.train_dataset.dataset):,}")
        logger.info(f"  Evaluation dataset size = {len(self.eval_dataset.dataset):,}")
        logger.info(f"  Num Epochs = {self.num_epochs:,}" if self.num_epochs >= 0 else "  Train until convergence")
        logger.info(f"  Mini batch size per device = {self.batch_size:,}")
        logger.info(f"  Total train batch size = {self.actual_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Number of parameters = {parameters_overview['total']:,}")
        logger.info(f"  Number of trainable parameters = {parameters_overview['trainable']:,} ({100*parameters_overview['frac']:.03f}%)")

        if self.train_interleaved:
            train_fct = self.train_epoch_interleaved
        else:
            train_fct = self.train_epoch

        for _ in range(self.num_epochs):
            train_fct()
            gc.collect()
        
        if self.exp_state.epoch != self.exp_state.last_eval_epoch:
            self.eval()

        logger.info(f"Finished training after {self.exp_state.epoch:,} epochs.")
        self.exp_state.save(self.run_dir)
        self.load_best_model()

    def train_epoch_interleaved(self):
        self.start_of_epoch()

        self.exp_state.epoch_step = 0

        self.model.train()
        
        mini_batches = []

        for _n_batch, batch in enumerate(iter(self.train_dataset)):
            for k, v in batch.items():
                batch[k] = v.to(device="cuda")
            mini_batches.append(batch)

            if (_n_batch + 1) % self.gradient_accumulation_steps == 0:
                for optimizer, scheduler, _model_call_kwargs, _opt_params in zip(self.optimizers, self.schedulers, self.model_call_kwargs, self.optimizer_params):
                    optimizer.zero_grad()
                    for _batch in mini_batches:
                        with self.get_train_context():
                            _res = self.model(**_batch, **_model_call_kwargs)
                            with torch.autocast(device_type="cuda", enabled=False):
                                loss = _res['loss'] / self.gradient_acc_pt
                        
                        if self.fp16_mp:
                            self.grad_scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        self.train_loss += loss.detach()

                        self.exp_state.epoch_step += 1
                    
                    self.exp_state.total_step += 1
                    self.exp_state.log_step += 1

                    if self.clip_grad_norm:
                        if self.fp16_mp:
                            self.grad_scaler.unscale_(optimizer)
                        self._grad_norm += torch.nn.utils.clip_grad_norm_(_opt_params, max_norm=self.max_grad_norm).detach()
                    if self.fp16_mp:
                        self.grad_scaler.step(optimizer)
                        self.grad_scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()
                            
                    if self.exp_state.total_step % self.log_steps == 0:
                        _tr_loss = self.train_loss.detach().float().cpu().item() / self.log_steps
                        _grad_norm = self._grad_norm.detach().float().cpu().item() / self.log_steps if self.clip_grad_norm else None
                        self.log_training(_tr_loss, step=self.exp_state.log_step, scheduler=scheduler, grad_norm=_grad_norm)
                        self.train_loss[...] = 0.
                        if self.clip_grad_norm:
                            self._grad_norm[...] = 0.

                mini_batches = []
        
            if (_n_batch + 1) == self.total_steps_per_epoch:
                continue
        
        self.end_of_epoch()
                        
        if self.exp_state.epoch % self.epochs_until_eval == 0:
            self.eval()

        self.exp_state.epoch += 1

    def train_epoch(self):
        self.start_of_epoch()

        self.exp_state.epoch_step = 0

        self.model.train()
        
        for optimizer, scheduler, _model_call_kwargs, _opt_params in zip(self.optimizers, self.schedulers, self.model_call_kwargs, self.optimizer_params):
            self.model.train()
            optimizer.zero_grad()

            for batch in iter(self.train_dataset):
                for k, v in batch.items():
                    batch[k] = v.to(device="cuda")

                with self.get_train_context():
                    _res = self.model(**batch, **_model_call_kwargs)
                    with torch.autocast(device_type="cuda", enabled=False):
                        loss = _res['loss'] / self.gradient_acc_pt
                    
                if self.fp16_mp:
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                self.train_loss += loss.detach()

                self.exp_state.epoch_step += 1

                if self.exp_state.epoch_step % self.gradient_accumulation_steps == 0:
                    self.exp_state.total_step += 1
                    self.exp_state.log_step += 1

                    if self.clip_grad_norm:
                        if self.fp16_mp:
                            self.grad_scaler.unscale_(optimizer)
                        self._grad_norm += torch.nn.utils.clip_grad_norm_(_opt_params, max_norm=self.max_grad_norm).detach()
                    if self.fp16_mp:
                        self.grad_scaler.step(optimizer)
                        self.grad_scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

                    if self.exp_state.total_step % self.log_steps == 0:
                        _tr_loss = self.train_loss.detach().float().cpu().item() / self.log_steps
                        _grad_norm = self._grad_norm.detach().float().cpu().item() / self.log_steps if self.clip_grad_norm else None
                        self.log_training(_tr_loss, step=self.exp_state.log_step, scheduler=scheduler, grad_norm=_grad_norm)
                        self.train_loss[...] = 0.
                        if self.clip_grad_norm:
                            self._grad_norm[...] = 0.

                if self.exp_state.epoch_step == self.total_steps_per_epoch:
                    continue
            
            if self.exp_state.epoch % self.epochs_until_eval == 0:
                self.eval()

        self.end_of_epoch()

        self.exp_state.epoch += 1

    def start_of_epoch(self):
        pass

    def end_of_epoch(self):
        pass

    @torch.inference_mode()
    def eval(self):
        self.model.eval()
        eval_loss = torch.tensor(0.0, dtype=torch.float32, device="cuda")

        if self.compute_metrics is not None:
            values = []
            labels = []

        for batch in iter(self.eval_dataset):
            for k, v in batch.items():
                batch[k] = v.to(device="cuda")

            with self.get_train_context():
                eval_res = self.model(**batch)
            
            _eval_loss = eval_res['loss'].detach()

            if self.compute_metrics is not None:
                values.append(eval_res["logits"].detach())
                labels.append(batch["labels"].detach())

            if self.compute_metrics is not None:
                metrics = self.compute_metrics(values=torch.cat(values), labels=torch.cat(labels))
            else:
                metrics = {}

            if "ln_det_sigmas" in eval_res and eval_res["ln_det_sigmas"] != None:
                ln_det_sigmas = eval_res["ln_det_sigmas"]
                ln_det_sigmas = ln_det_sigmas.detach().cpu() / self._model.config.param_dim
                _det_sigmas = torch.mean(torch.exp(-ln_det_sigmas))
                metrics["eval/det_sigma"] = _det_sigmas

            eval_loss += _eval_loss.float()

        self.exp_state.last_eval_epoch = self.exp_state.epoch + 1 # We increment exp_state.epoch only later

        _eval_loss = eval_loss.detach().float().cpu().item() / self.total_eval_steps
        self.log_eval(_eval_loss, metrics)
        self.maybe_save_checkpoint(_eval_loss, force_save=self.force_model_saving)

    def save_model(self, dir:str=None, trainable_weights_only=False):
        if dir == None:
            dir = self.model_dir
        if trainable_weights_only and not self.save_full_model:
            state_dict = { k: v for k, v in self._model.named_parameters() if v.requires_grad }
        else:
            state_dict = self._model.state_dict()
        
        torch.save(state_dict, os.path.join(dir, "pytorch_model.bin"))

    def maybe_save_checkpoint(self, eval_metric: float, force_save=False):
        if self.exp_state.best_eval_metric == None:
            new_best_model = True
            first_eval = True
        else:
            new_best_model = eval_metric < self.exp_state.best_eval_metric
            first_eval = False
            
        if new_best_model or force_save:
            ckpt_name = f"ckpt_{self.exp_state.epoch}"
            ckpt_path = os.path.join(self.checkpoints_dir, ckpt_name)
            
            if new_best_model:
                self.exp_state.best_checkpoint = ckpt_path
                self.exp_state.best_eval_metric = eval_metric
                self.exp_state.best_epoch = self.exp_state.epoch
            self.exp_state.last_checkpoint = ckpt_path
            self.exp_state.last_eval_metric = eval_metric

            logger.info(f"Saving checkpoint {self.exp_state.epoch} with loss {self.exp_state.last_eval_metric:.04f}.")

            try:
                if self.max_num_checkpoints > 0:
                    old_ckpts = os.listdir(self.checkpoints_dir)
                    if len(old_ckpts) >= self.max_num_checkpoints:
                        if not new_best_model:
                            _best_checkpoint = os.path.basename(os.path.normpath(self.exp_state.best_checkpoint))
                            for i_ckpts in range(len(old_ckpts)):
                                if old_ckpts[i_ckpts] == _best_checkpoint:
                                    old_ckpts.pop(i_ckpts)
                                    break
                            # old_ckpts.pop(os.path.basename(self.exp_state.best_checkpoint))
                        old_ckpts.sort(key=lambda x: int(x.split('ckpt_')[1]))
                        ckpt_to_del = old_ckpts[0]
                        shutil.rmtree(os.path.join(self.checkpoints_dir, ckpt_to_del))
            except Exception as ex:
                if self.logger != None:
                    self.logger.info(f"Removing a checkpoint yielded the exception: {str(ex)}")

            self.save_checkpoint(ckpt_path)

    def save_checkpoint(self, ckpt_path:str):
        Path(ckpt_path).mkdir(exist_ok=True, parents=True)

        state_dicts = {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }
        if self.grad_scaler != None:
            state_dicts["scaler"] = self.grad_scaler.state_dict()
        torch.save(state_dicts, os.path.join(ckpt_path, "trainer.pt"))

        self.exp_state.save(ckpt_path)

        self.save_model(ckpt_path, trainable_weights_only=True)

    @torch.inference_mode()
    def log_training(
            self,
            loss: float = None,
            step: int = None,
            grad_norm: float = None,
            skip_others: bool = False,
            skip_epoch: bool = False,
            scheduler = None
    ):
        step = step if step is not None else self.exp_state.log_step

        if loss != None:
            self.summarizer.add_scalar("train/loss", loss, step)
        if not skip_epoch:
            self.summarizer.add_scalar(
                "train/epoch",
                float(self.exp_state.epoch) + min(1., (self.exp_state.epoch_step) / self.total_steps_per_epoch / len(self.optimizers)),
                step
            )
        
        if not skip_others:
            if scheduler != None:
                self.summarizer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            else:    
                self.summarizer.add_scalar("train/lr", self.schedulers[0].get_last_lr()[0], step)
            
            if grad_norm != None:
                self.summarizer.add_scalar("train/grad_norm", grad_norm, step)
            else:
                if self.clip_grad_norm:
                    self.summarizer.add_scalar("train/grad_norm", self._grad_norm.float().cpu().item() / self.log_steps, step)
                else:
                    total_norm = torch.nn.utils.get_total_norm(self.model.parameters())
                    self.summarizer.add_scalar("train/grad_norm", total_norm.float().cpu().item(), step)

    @torch.inference_mode()
    def log_eval(self, loss:float = None, metrics: dict = {}):
        if loss != None:
            self.summarizer.add_scalar("eval/loss", loss, self.exp_state.log_step)

        if metrics != None:
            for k, v in metrics.items():
                self.summarizer.add_scalar(f"eval/{k}", v.float().cpu().item(), self.exp_state.log_step)

    def log_training_setup(self, parameters_overview:dict=None):
        self.summarizer.add_text("exp_config", json.dumps(self.exp_config.to_dict(), indent=2))

        training_config = {
            "batch_size": self.batch_size, 
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "actual_batch_size": self.actual_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "clip_grad_norm": self.clip_grad_norm,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "epochs_until_eval": self.epochs_until_eval,
            "log_steps": self.log_steps,
            "max_num_checkpoints": self.max_num_checkpoints,
            "fp16_mp": self.fp16_mp,
            "bf16_mp": self.bf16_mp,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps_per_epoch": self.total_steps_per_epoch,
            "total_eval_steps": self.total_eval_steps,
            "train_head": self.train_head,
            "train_only_head": self.train_only_head,
            "train_interleaved": self.train_interleaved,
            "warmup_epochs": self.warmup_epochs,
            "stable_epochs": self.stable_epochs,
            "decay_epochs": self.decay_epochs,
            "number_of_optimizers": len(self.optimizers),
            "force_model_saving": self.force_model_saving
        }

        training_config = self.add_training_setup_log_vals(training_config)

        if parameters_overview != None:
            training_config["parameters"] = parameters_overview
        self.summarizer.add_text("training_config", json.dumps(training_config, indent=2))

        with open(os.path.join(self.log_dir, "training_config.json"), "w") as f:
            json.dump(training_config, f, indent=2)

        if self.logger != None:
            _train_param_overview = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    _train_param_overview[n] = f"{p.numel():,}"
            logger.info(f"Trainable parameters:\n{json.dumps(_train_param_overview, indent=2)}")

    def add_training_setup_log_vals(self, log_dict):
        return log_dict

    def load_model(self, ckpt):
        if self.logger != None:
            self.logger.info(f"Loading weights from the checkpoint {ckpt}.")
        self._model = load_weights(bin_dir=ckpt, model=self._model, logger=self.logger)

    def load_optimizers(self, ckpt):
        if self.logger != None:
            self.logger.info(f"Loading optimizer states from the checkpoint {ckpt}.")
        state_dicts = torch.load(f"{ckpt}/trainer.pt", weights_only=False)
        for opt, opt_state_dict in zip(self.optimizers, state_dicts["optimizers"]):
            opt.load_state_dict(opt_state_dict)

    def load_best_model(self):
        best_ckpt = self.exp_state.best_checkpoint
        if self.logger != None:
            self.logger.info(f"Loading weights from the checkpoint {best_ckpt}.")
        self._model = load_weights(bin_dir=best_ckpt, model=self._model, logger=self.logger)
