import os, psutil, gc, sys
from glob import glob
from pathlib import Path
from datetime import datetime
import random

import torch
from torch.utils.data.dataloader import DataLoader

from src.experiments.regression.network.construct_model import construct_model_config, init_l3m_regressor
from src.experiments.regression.network.configuration import L3MRegressorConfig
from src.experiments.regression.training_utils import freeze_llm, split_files
from src.experiments.regression.dataset import get_collate_function, gen_lc_data
from src.utils.datasets import L3MDataset
from src.utils.training import get_compute_metrics_torch
from src.experiments.regression.evaluation import evaluate_model, plot

from src.utils.trainer import Trainer
from src.experiments.regression.reg_exp_config import RegExpConfig

from src.experiments.regression.training_utils import create_optimizer_groups

import transformers

import logging

import math

os.environ["TOKENIZERS_PARALLELISM"] = "true"

########################################################################
## arguments ###########################################################
########################################################################

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--config', required=True, type=str)

args = parser.parse_args()

# config for the experiment
exp_config: RegExpConfig = RegExpConfig.load(args.config)

# main dir for the run
run_name = f"{datetime.today().strftime('%Y%m%d_%H%M%S')}_{exp_config.run_name}" if exp_config.run_name.strip() != "" else datetime.today().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(args.output_dir, run_name)
Path(output_dir).mkdir(exist_ok=True, parents=True)

exp_config.save(os.path.join(output_dir, "exp_config.json"))

# dir for storing the model config file
MODEL_CONFIG_DIR = os.path.join(output_dir, "model")

LLM_DIR = exp_config.LLM_DIR

########################################################################
## logging #############################################################
########################################################################

log_dir = os.path.join(output_dir, "log")
Path(log_dir).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{log_dir}/training.log"),
        logging.StreamHandler()
    ]
)
log_level = logging.INFO
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

def log_exceptions(exc_type, exc_value, exc_traceback):
    logging.critical("Error", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = log_exceptions

logger.info(f"{psutil.cpu_count()} threads available.")

########################################################################
## TF32 ################################################################
########################################################################

if exp_config.tf32:
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

########################################################################
## model config ########################################################
########################################################################

label_names = ['m', 'Om', 'E0', 'log LX', 'log Tvir', 'zeta']

if not exp_config.train_from_scratch:
    logger.info("Initializing LLM config")
    config = L3MRegressorConfig(
        lc_connector_config = exp_config.lc_connector,
        param_head_config = exp_config.head,
        preprocessing=exp_config.preprocess,
        attn_implementation=exp_config.attn_implementation,
        param_dim=len(label_names),
        from_scratch=exp_config.train_from_scratch,
        num_sys_prompt_tokens=exp_config.num_sys_prompt_tokens,
        num_pre_lc_tokens=exp_config.num_pre_lc_tokens
    )
else:
    logger.info("Initializing scratch config")
    config = L3MRegressorConfig.for_scratch_model(
        lc_connector_config = exp_config.lc_connector,
        param_head_config = exp_config.head,
        preprocessing=exp_config.preprocess,
        attn_implementation=exp_config.attn_implementation,
        param_dim=len(label_names),
        from_scratch=exp_config.train_from_scratch,
        num_sys_prompt_tokens=exp_config.num_sys_prompt_tokens,
        num_pre_lc_tokens=exp_config.num_pre_lc_tokens,
        **exp_config.scratch_config
    )

config.attention_dropout = exp_config.attn_drop

model_config = construct_model_config(
    LLM_DIR=LLM_DIR,
    SAVE_DIR=MODEL_CONFIG_DIR,
    model_config=config,
    load_llm_weights=exp_config.load_llm_weights,
    train_from_scratch=exp_config.train_from_scratch,
    custom_number_of_layers=exp_config.custom_number_of_layers,
    logger=logger
    )
    
########################################################################
## dataset #############################################################
########################################################################

files = glob(f'{exp_config.DATASET_DIR}/*.npz')
random.seed(42)
random.shuffle(files)
random.seed()

N_TEST_FILES = exp_config.test_ds_size
test_files = files[:N_TEST_FILES]
files = files[N_TEST_FILES:]
files = files[:math.ceil(len(files) * exp_config.ds_ratio)]
TRAIN_EVAL_RATIO = exp_config.train_eval_ratio
train_files, eval_files = split_files(files, TRAIN_EVAL_RATIO)

special_token_id = { tok['name']: tok['token_id'] for tok in model_config.special_token }

gen_kwargs_train = {
    'files': train_files,
    'proc_dir': MODEL_CONFIG_DIR,
    'model_config': model_config,
    'label_names': tuple(label_names),
    'patch_size': exp_config.patch_size,
    'preprocess': exp_config.preprocess,
    'pad_sequence': exp_config.pad_sequence,
    'logger': logger,
    'multiplicity': exp_config.train_ds_multiplicity,
    'minimalistic_chat_template': exp_config.minimalistic_chat_template,
    "special_tokens_id": special_token_id
}
logger.info("Generating training dataset")
train_dataset = L3MDataset(generator=gen_lc_data(**gen_kwargs_train), device="cpu")
logger.info(f"Finished generating training dataset with {len(train_dataset)} entries.")

gen_kwargs_eval = {
    'files': eval_files,
    'proc_dir': MODEL_CONFIG_DIR,
    'model_config': model_config,
    'label_names': tuple(label_names),
    'patch_size': exp_config.patch_size,
    'preprocess': exp_config.preprocess,
    'pad_sequence': exp_config.pad_sequence,
    'logger': logger,
    'minimalistic_chat_template': exp_config.minimalistic_chat_template,
    "special_tokens_id": special_token_id
}
logger.info("Generating evaluation dataset")
eval_dataset = L3MDataset(generator=gen_lc_data(**gen_kwargs_eval), device="cpu")
logger.info(f"Finished generating evaluation dataset with {len(eval_dataset)} entries.")

gen_kwargs_test = {
    'files': test_files,
    'proc_dir': MODEL_CONFIG_DIR,
    'model_config': model_config,
    'label_names': tuple(label_names),
    'patch_size': exp_config.patch_size,
    'preprocess': exp_config.preprocess,
    'pad_sequence': exp_config.pad_sequence,
    'logger': logger,
    'minimalistic_chat_template': exp_config.minimalistic_chat_template,
    "special_tokens_id": special_token_id
}
logger.info("Generating test dataset")

test_dataset = L3MDataset(generator=gen_lc_data(**gen_kwargs_test), device="cpu")
logger.info(f"Finished generating test dataset with {len(test_dataset)} entries.")

########################################################################
## training config #####################################################
########################################################################

batch_size = exp_config.batch_size
gradient_accumulation_steps = exp_config.gradient_accumulation_steps
actual_batch_size = batch_size * gradient_accumulation_steps

warmup_epochs = exp_config.warmup_epochs
num_train_epochs = exp_config.num_train_epochs

train_ds_size = len(train_dataset)
eval_ds_size = len(eval_dataset)

eff_num_ds_rows = train_ds_size
warmup_steps = math.floor(train_ds_size * warmup_epochs / actual_batch_size)
eval_batch_size = min(max(math.floor(eval_ds_size / 8) * 8, 1), 2 * batch_size)

decay_epochs = exp_config.decay_epochs
decay_steps = math.floor(train_ds_size * decay_epochs / actual_batch_size)
stable_steps = math.floor(train_ds_size * num_train_epochs / actual_batch_size) - warmup_steps - decay_steps
learning_rate = exp_config.lr

########################################################################
## training ############################################################
########################################################################
    
for i in range(exp_config.number_of_runs):
    logger.info(f"Starting run {i}")

    run_dir = os.path.join(output_dir, f"{i:02d}")
    run_save_dir = os.path.join(run_dir, "model")

    model, processor = init_l3m_regressor(
        LLM_DIR=LLM_DIR,
        SAVE_DIR=run_save_dir,
        MODEL_CONFIG_DIR=MODEL_CONFIG_DIR,
        load_llm_weights=exp_config.load_llm_weights,
        train_from_scratch=exp_config.train_from_scratch,
        dtype=torch.float32,
        _attn_implementation=exp_config.attn_implementation,
        logger=logger
    )

    if exp_config.train_from_scratch:
        model.is_causal = False
        logging.info("Setting fully connected attention mask")

    model.set_multi_dtype(
        lc_embedding_dtype=torch.float32,
        base_dtype=torch.bfloat16,
        param_head_dtype=torch.float32
    )

    model.param_head._index_diag.to(device="cuda")
    model.param_head._index_off_diag_0.to(device="cuda")
    model.param_head._index_off_diag_1.to(device="cuda") 

    logger.info(f"L3M is loaded with  _attn_implementation: {model.config._attn_implementation}")

    if not exp_config.train_from_scratch:
        freeze_llm(model)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frac = float(trainable_params) / float(total_params)

    logger.info(f"Max memory allocated during model creation {torch.cuda.max_memory_allocated() / 1024.**3:.1f}GB")

    torch.cuda.empty_cache()

    logger.info(f"Model allocates {torch.cuda.max_memory_allocated() / 1024.**3:.1f}GB")

    log_save_eval_config = {
        "logging_steps": max(math.floor(exp_config.epochs_for_log * train_ds_size / actual_batch_size), 1),
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": max(math.floor(eff_num_ds_rows / actual_batch_size), 1),
        "eval_strategy": "steps",
        "eval_steps": max(math.floor(eff_num_ds_rows / actual_batch_size), 1)
    }
    
    collate_fct = get_collate_function(
        special_tokens_id=model.special_token_id
        )
    
    train_ds_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        collate_fn=collate_fct,
        drop_last=True,
        num_workers=exp_config.dataloader_threads,
        prefetch_factor=1 if exp_config.dataloader_threads > 0 else None,
        pin_memory=exp_config.dataloader_pin_memory,
        pin_memory_device="cuda" if exp_config.dataloader_pin_memory else "",
        persistent_workers=exp_config.dataloader_persistent,
    )
    eval_ds_loader = DataLoader(
        eval_dataset,
        eval_batch_size,
        shuffle=True,
        collate_fn=collate_fct,
        drop_last=True,
        num_workers=exp_config.dataloader_threads,
        prefetch_factor=1 if exp_config.dataloader_threads > 0 else None,
        pin_memory=exp_config.dataloader_pin_memory,
        pin_memory_device="cuda" if exp_config.dataloader_pin_memory else "",
        persistent_workers=exp_config.dataloader_persistent,
    )

    if exp_config.compile:
        compile_kwargs={"backend": "inductor", "mode": "max-autotune-no-cudagraphs"}
        opt_model = torch.compile(model, **compile_kwargs)
    else:
        opt_model = model
    
    if not exp_config.train_from_scratch:
        decay_parameters = decay_parameters = [name for name, _ in opt_model.named_parameters()]
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # This can be used
        # decay_parameters = [name for name in decay_parameters if "norm" not in name]
        # decay_parameters = [name for name in decay_parameters if "token" not in name]

        _optimizer_grouped_parameters_lc_tok = create_optimizer_groups(
            opt_model,
            decay_parameters,
            learning_rate,
            weight_decay=0.,
            lr_head=0.0,
            lr_others=0.0,
            lr_lc=1.0,
            lr_tok=1.0,
            logger=logger
        )
        
        _optimizers = [
            torch.optim.AdamW(
                _optimizer_grouped_parameters_lc_tok,
                lr=learning_rate,
                weight_decay=0.,
                betas=(0.9, 0.999)
            )
        ]
    else:
        decay_parameters = decay_parameters = [name for name, _ in opt_model.named_parameters()]
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # This can be used
        # decay_parameters = [name for name in decay_parameters if "norm" not in name]
        # decay_parameters = [name for name in decay_parameters if "token" not in name]

        _optimizer_grouped_parameters_main = create_optimizer_groups(
            opt_model,
            decay_parameters,
            learning_rate,
            weight_decay=0.,
            lr_head=1.0,
            lr_others=1.0,
            lr_lc=1.0,
            lr_tok=1.0,
            logger=logger
        )

        _optimizers = [
            torch.optim.AdamW(
                _optimizer_grouped_parameters_main,
                lr=learning_rate,
                weight_decay=0.,
                betas=(0.9, 0.999)
            )
        ]

    _optimizer_grouped_parameters_head = create_optimizer_groups(
            opt_model,
            decay_parameters,
            learning_rate,
            weight_decay=0.,
            lr_head=1.0,
            lr_others=0.0,
            lr_lc=0.0,
            lr_tok=0.0,
            logger=logger
        )

    _optimizer_head = torch.optim.AdamW(
            _optimizer_grouped_parameters_head,
            lr=learning_rate,
            weight_decay=0.,
            betas=(0.9, 0.999)
        )

    trainer = Trainer(
        model=model,
        opt_model=opt_model,
        optimizers=_optimizers,
        optimizer_head=_optimizer_head,
        train_dataset=train_ds_loader,
        eval_dataset=eval_ds_loader,
        run_dir=run_dir,
        logger=logger,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_train_epochs,
        exp_config=exp_config,
        lr=learning_rate,
        weight_decay=0.,
        max_grad_norm=exp_config.max_grad_norm,
        warmup_epochs=exp_config.warmup_epochs,
        decay_epochs=exp_config.decay_epochs,
        log_dir=os.path.join(run_dir, "log"),
        log_steps=log_save_eval_config["logging_steps"],
        checkpoints_dir=os.path.join(run_dir, "checkpoints"),
        max_num_checkpoints=exp_config.max_num_checkpoints,
        fp16_mp=False,
        bf16_mp=False,
        compute_metrics=get_compute_metrics_torch(label_names),
        train_interleaved=not exp_config.train_from_scratch
    )
    train_result = trainer.train()
    trainer.save_model(run_dir)

    model = trainer._model

    collate_fct_eval = get_collate_function(special_tokens_id=model.special_token_id)

    evaluate_model(model, test_dataset, run_dir, label_names, collate_fct_eval)
    plot(out_dir=run_dir, label_names=label_names, n_samples=50)

    logger.info(f"Max memory allocated during training: {torch.cuda.max_memory_allocated() / 1024.**3:.1f}GB")

    del model, trainer, collate_fct, collate_fct_eval
    torch.cuda.empty_cache()
    gc.collect()

logger.info("Experiment finished.")
