import sys
from glob import glob
from pathlib import Path
from datetime import datetime
import math

import torch
from torch.utils.data.dataloader import DataLoader

from src.experiments.generation.training_utils import freeze_llm_embedding, get_l3m, split_files, add_id_to_lcs

from src.experiments.generation.network.construct_model import construct_l3m_generator
from src.experiments.generation.network.configuration import L3MGenerationConfig
from src.experiments.generation.dataset import get_collate_function, get_collate_function_generation, FCDatasetConfig, FCDataset, FCStaticData

from src.utils.trainer import Trainer
from src.experiments.generation.gen_exp_config import GenExpConfig

from src.utils.preprocessing import get_preprocessing

from src.experiments.generation.evaluation import evaluate_generation, evaluate_next_patch, evaluate_next_patch_l2

from src.experiments.generation.training_utils import create_optimizer_groups

import transformers

from math import floor, ceil

from peft import LoraConfig, get_peft_model

import logging

import os, psutil, gc
import random

########################################################################
## arguments ###########################################################
########################################################################

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--config', required=True, type=str)

args = parser.parse_args()

# config for the experiment
exp_config: GenExpConfig = GenExpConfig.load(args.config)

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

###############################################################
## config #####################################################
###############################################################

lc_shape = exp_config.lc_shape
patch_size = exp_config.patch_size
patch_dim = math.prod(patch_size)
label_names = ['m', 'Om', 'E0', 'log LX', 'log Tvir', 'zeta']

###############################################################
## model ######################################################
###############################################################

SAVE_DIR = os.path.join(output_dir, "model")

if not exp_config.train_from_scratch:
    logger.info("Initializing pretrained LLM")
    config: L3MGenerationConfig = L3MGenerationConfig(
        lc_patch_size=patch_size,
        preprocessing=exp_config.preprocess,
        num_parameters = 6,
        from_scratch=exp_config.train_from_scratch,
        attn_implementation = exp_config.attn_implementation,
        cfm_config = exp_config.cfm_connector
    )
else:
    logger.info("Initializing scratch model")
    config: L3MGenerationConfig = L3MGenerationConfig.for_scratch_model(
        lc_patch_size=patch_size,
        preprocessing=exp_config.preprocess,
        num_parameters = 6,
        attn_implementation = exp_config.attn_implementation,
        from_scratch=exp_config.train_from_scratch,
        **exp_config.scratch_config,
        cfm_config = exp_config.cfm_connector
    )

config.attention_dropout = exp_config.attn_drop

config.lc_connector_config = exp_config.lc_connector
config.parameter_connector_config = exp_config.parameter_connector
config.t_index_connector_config = exp_config.t_index_connector

model, processor = construct_l3m_generator(
    LLM_DIR=LLM_DIR,
    SAVE_DIR=SAVE_DIR,
    model_config=config,
    load_llm_weights=exp_config.load_llm_weights,
    train_from_scratch=exp_config.train_from_scratch,
    device="cuda",
    dtype=torch.float32,
    logger=logger,
)

if exp_config.lora.get('use', False):
    train_norms = exp_config.lora['config'].get("train_norms", False)

    modules_to_save = ["lc_embedding.lc_connector", "parameter_embedding", "t_index_embedding", "lc_head.sub_lc_head", "lc_head.cnn_layer", "lc_head.cond_layer"]
    for tok in model.special_token_embds.keys():
        modules_to_save.append(f"special_token_embds.{tok}")
    if train_norms:
        modules_to_save.append("norm")
    
    lora_r = exp_config.lora['config']['r']

    target_modules=r'layers\.\d*\.self_attn\.o_proj|layers\.\d*\.self_attn\.q_proj|layers\.\d*\.self_attn\.k_proj|layers\.\d*\.self_attn\.v_proj|layers\.\d*\.mlp\.gate_proj|layers\.\d*\.mlp\.up_proj|layers\.\d*\.mlp\.down_proj'
    if train_norms:
        for i in range(model.config.num_hidden_layers):
            modules_to_save += [f"layers.{i}.input_layernorm", f"layers.{i}.post_attention_layernorm"]

    lora_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_r * 2,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        base_model_name_or_path=SAVE_DIR,
        use_dora=exp_config.lora["config"].get("use_dora", False),
    )

    model = get_peft_model(model, lora_config, adapter_name='generation_lora')
    model.save_pretrained(save_directory=SAVE_DIR, safe_serialization=False)

model.set_multi_dtype(
    lc_embedding_dtype=torch.float32,
    parameter_embedding_dtype=torch.float32,
    base_dtype=torch.bfloat16,
    lc_head_dtype=torch.float32
)
model = model.to("cuda")

get_l3m(model, exp_config).set_multi_dtype(
    lc_embedding_dtype=torch.float32,
    parameter_embedding_dtype=torch.float32,
    base_dtype=torch.bfloat16,
    lc_head_dtype=torch.float32
)

if not exp_config.train_from_scratch:
    freeze_llm_embedding(model, exp_config)

logger.info("Lvska is loaded")

torch.cuda.empty_cache()

logger.info(f"Model allocates {torch.cuda.max_memory_allocated() / 1024.**3:.1f}GB")

###############################################################
## dataset ####################################################
###############################################################

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


logger.info("Loading dataset...")
logger.info(f"Loading training dataset from {len(train_files)} files")
train_dataset_config = FCDatasetConfig(
    label_names=label_names,
    t_init=exp_config.t_init,
    t_end=exp_config.chunked_sublc_size,
    preprocess=exp_config.preprocess,
    drop_last_slice=True,
    trim=True
)
train_dataset = FCDataset.load_from_generator(
    config=train_dataset_config,
    id_files=add_id_to_lcs(train_files),
    logger=logger,
    clone_static_input=False
)
logger.info(f"Train set loaded with {len(train_dataset)} entries.")

logger.info(f"Loading evaluation dataset from {len(eval_files)} files")
eval_dataset_config = FCDatasetConfig(
    label_names=label_names,
    t_init=exp_config.t_init,
    t_end=exp_config.chunked_sublc_size,
    preprocess=exp_config.preprocess,
    drop_last_slice=True,
    trim=True
)
eval_dataset = FCDataset.load_from_generator(
    config=eval_dataset_config,
    id_files=add_id_to_lcs(eval_files),
    logger=logger,
    clone_static_input=False
)
logger.info(f"Eval set loaded with {len(eval_dataset)} entries.")

static_input = FCStaticData.generate(
    t_max=exp_config.t_max,
    t_init=exp_config.t_init,
    t_margin=exp_config.chunked_sublc_size - exp_config.t_max,
    label_names=label_names,
    minimalistic_chat_template=exp_config.minimalistic_chat_template,
    pad_sequence=True,
    patch_size=exp_config.patch_size,
    lc_shape=lc_shape,
    processor=processor,
    model_config=model.config,
    special_tokens_id=model.special_token_id,
    parameter_token_id=model.parameter_token_id,
    t_index_id=model.t_index_token_id,
    logger=logger
)
train_dataset.static_input = static_input.data
eval_dataset.static_input = static_input.data
logger.info("Static input data has been generated and set.")

###############################################################
## run config #################################################
###############################################################

batch_size = exp_config.batch_size
gradient_accumulation_steps = exp_config.gradient_accumulation_steps
actual_batch_size = batch_size * gradient_accumulation_steps

warmup_epochs = exp_config.warmup_epochs
num_train_epochs = exp_config.num_train_epochs
decay_epochs = exp_config.decay_epochs

train_ds_size = len(train_dataset)
eval_ds_size = len(eval_dataset)

eff_num_ds_rows = train_ds_size
warmup_steps = math.floor(train_ds_size * warmup_epochs / actual_batch_size)
decay_steps = math.floor(train_ds_size * decay_epochs / actual_batch_size)
stable_steps = math.floor(train_ds_size * num_train_epochs / actual_batch_size) - warmup_steps - decay_steps
logging_steps = max(math.floor(exp_config.epochs_for_log * train_ds_size / actual_batch_size), 1)

eval_batch_size = min(max(math.floor(eval_ds_size / 8) * 8, 1), batch_size)

optimizer_kwargs = {
    "warmup_steps": warmup_steps,
    "lr_scheduler_kwargs": {"num_decay_steps": decay_steps, "num_stable_steps": stable_steps},
    "lr_scheduler_type": "warmup_stable_decay",
}
learning_rate = exp_config.lr

collate_fct = get_collate_function(
    special_tokens_id=get_l3m(model, exp_config).special_token_id,
    parameter_token_id=get_l3m(model, exp_config).parameter_token_id,
    augment=True,
    preprocess=exp_config.preprocess if train_dataset.config.preprocess == None else None
)

###############################################################
## training ###################################################
###############################################################

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

decay_parameters = decay_parameters = [name for name, _ in opt_model.named_parameters()]
decay_parameters = [name for name in decay_parameters if "bias" not in name]
# This can be used
# decay_parameters = [name for name in decay_parameters if "norm" not in name]
# decay_parameters = [name for name in decay_parameters if "token" not in name]

_optimizer_grouped_parameters_lc_tok = create_optimizer_groups(
    opt_model,
    decay_parameters,
    learning_rate,
    weight_decay=exp_config.weight_decay,
    logger=logger
)

optimizers = [
    torch.optim.AdamW(
        _optimizer_grouped_parameters_lc_tok,
        lr=learning_rate,
        weight_decay=exp_config.weight_decay,
        betas=(0.9, 0.999)
    )
]

trainer = Trainer(
    model=model,
    train_dataset=train_ds_loader,
    eval_dataset=eval_ds_loader,
    run_dir=output_dir,
    logger=logger,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_epochs=num_train_epochs,
    exp_config=exp_config,
    opt_model=opt_model,
    optimizers=optimizers,
    train_head=False,
    train_interleaved=True,
    train_only_head=False,
    lr=learning_rate,
    weight_decay=exp_config.weight_decay,
    max_grad_norm=exp_config.max_grad_norm,
    warmup_epochs=exp_config.warmup_epochs,
    decay_epochs=exp_config.decay_epochs,
    log_dir=os.path.join(output_dir, "log"),
    log_steps=logging_steps,
    checkpoints_dir=os.path.join(output_dir, "checkpoints"),
    max_num_checkpoints=exp_config.max_num_checkpoints,
    fp16_mp=False,
    compute_metrics=None,
    force_model_saving=True,
)

train_result = trainer.train()
if exp_config.lora["use"]:
    trainer._model = trainer._model.merge_and_unload()
trainer.save_model(output_dir)

del train_ds_loader, train_dataset, eval_ds_loader, eval_dataset
gc.collect()
torch.cuda.empty_cache()

###############################################################
## validation #################################################
###############################################################

model.set_multi_dtype(
    lc_embedding_dtype=torch.float32,
    parameter_embedding_dtype=torch.float32,
    base_dtype=torch.bfloat16,
    lc_head_dtype=torch.float32
)

if exp_config.compile_for_eval:
    compile_kwargs={"backend": "inductor", "mode": "max-autotune-no-cudagraphs"}
    model = torch.compile(model, **compile_kwargs)

batch_size = exp_config.eval_batch_size

static_input = FCStaticData.generate(
    t_max=exp_config.eval_t_max,
    t_init=exp_config.eval_t_init,
    t_margin=0,
    label_names=label_names,
    minimalistic_chat_template=exp_config.minimalistic_chat_template,
    pad_sequence=True,
    patch_size=exp_config.patch_size,
    lc_shape=lc_shape,
    processor=processor,
    model_config=model.config,
    special_tokens_id=model.special_token_id,
    parameter_token_id=model.parameter_token_id,
    t_index_id=model.t_index_token_id,
    logger=logger
)
logger.info("Static input data has been generated for the evaluation.")

###############################################################
## generation and next patch ##################################
###############################################################

if exp_config.preprocess != None:
    preprocessor = get_preprocessing(exp_config.preprocess)
else:
    preprocessor = None

max_lcs_generation = exp_config.eval_generation_lcs
if max_lcs_generation > 0:
    generation_test_files = test_files[:max_lcs_generation]
else:
    generation_test_files = test_files
    
logger.info(f"Loading test generation dataset from {len(generation_test_files)} files")
test_generation_dataset_config = FCDatasetConfig(
    label_names=label_names,
    t_init=exp_config.eval_t_init,
    t_end=exp_config.eval_t_max,
    preprocess=exp_config.preprocess,
    drop_last_slice=False,
    trim=True
)
test_generation_dataset = FCDataset.load_from_generator(
    config=test_generation_dataset_config,
    id_files=add_id_to_lcs(generation_test_files),
    logger=logger,
    clone_static_input=False
)
logger.info(f"Test generation set loaded with {len(test_generation_dataset)} entries.")

test_generation_dataset.static_input = static_input.data

collate_fct_test = get_collate_function_generation(
    special_tokens_id=model.special_token_id,
    parameter_token_id=model.parameter_token_id,
    preprocess=exp_config.preprocess,
    custom_t_max=exp_config.eval_t_max
)
test_generation_ds_loader = DataLoader(
    test_generation_dataset,
    batch_size,
    shuffle=False,
    collate_fn=collate_fct_test,
    drop_last=False,
    num_workers=exp_config.dataloader_threads,
    prefetch_factor=1 if exp_config.dataloader_threads > 0 else None,
    pin_memory=exp_config.dataloader_pin_memory,
    pin_memory_device="cuda" if exp_config.dataloader_pin_memory else "",
    persistent_workers=exp_config.dataloader_persistent,
)

eval_output_dir = os.path.join(output_dir, "test", "generation")
evaluate_generation(
    test_dataset=test_generation_ds_loader,
    model=model,
    output_dir=eval_output_dir,
    eval_t_max=exp_config.eval_t_max,
    eval_t_init=exp_config.eval_t_init,
    train_t_max=exp_config.t_max,
    train_t_init=exp_config.t_init,
    patch_shape=exp_config.patch_size,
    lc_preprocessor=preprocessor
)

del test_generation_ds_loader
gc.collect()
torch.cuda.empty_cache()

collate_fct_test = get_collate_function(
    special_tokens_id=model.special_token_id,
    parameter_token_id=model.parameter_token_id,
    preprocess=exp_config.preprocess,
    augment=False
)
test_generation_ds_loader = DataLoader(
    test_generation_dataset,
    batch_size,
    shuffle=False,
    collate_fn=collate_fct_test,
    drop_last=False,
    num_workers=exp_config.dataloader_threads,
    prefetch_factor=1 if exp_config.dataloader_threads > 0 else None,
    pin_memory=exp_config.dataloader_pin_memory,
    pin_memory_device="cuda" if exp_config.dataloader_pin_memory else "",
    persistent_workers=exp_config.dataloader_persistent,
)

eval_output_dir = os.path.join(output_dir, "test", "next_patch")
evaluate_next_patch(
    test_dataset=test_generation_ds_loader,
    model=model,
    output_dir=eval_output_dir,
    lc_preprocessor=preprocessor
)

del test_generation_ds_loader, test_generation_dataset
gc.collect()
torch.cuda.empty_cache()

###############################################################
## next patch l2 ##############################################
###############################################################

max_lcs_l2 = exp_config.eval_l2_lcs
if max_lcs_l2 > 0:
    l2_test_files = test_files[:max_lcs_l2]
else:
    l2_test_files = test_files
    
logger.info(f"Loading test l2 dataset from {len(l2_test_files)} files")
test_l2_dataset_config = FCDatasetConfig(
    label_names=label_names,
    t_init=exp_config.eval_t_init,
    t_end=exp_config.eval_t_max,
    preprocess=exp_config.preprocess,
    drop_last_slice=False,
    trim=True
)
test_l2_dataset = FCDataset.load_from_generator(
    config=test_l2_dataset_config,
    id_files=add_id_to_lcs(l2_test_files),
    logger=logger,
    clone_static_input=False
)
logger.info(f"Test l2 set loaded with {len(test_l2_dataset)} entries.")

test_l2_dataset.static_input = static_input.data

collate_fct_test = get_collate_function(
    special_tokens_id=model.special_token_id,
    parameter_token_id=model.parameter_token_id,
    preprocess=exp_config.preprocess,
    augment=False
)
test_l2_ds_loader = DataLoader(
    test_l2_dataset,
    batch_size,
    shuffle=False,
    collate_fn=collate_fct_test,
    drop_last=False,
    num_workers=exp_config.dataloader_threads,
    prefetch_factor=1 if exp_config.dataloader_threads > 0 else None,
    pin_memory=exp_config.dataloader_pin_memory,
    pin_memory_device="cuda" if exp_config.dataloader_pin_memory else "",
    persistent_workers=exp_config.dataloader_persistent,
)

eval_output_dir = os.path.join(output_dir, "test", "next_patch_l2")
evaluate_next_patch_l2(
    test_dataset=test_l2_ds_loader,
    model=model,
    output_dir=eval_output_dir,
    n_samples=exp_config.eval_l2_samples,
    logger=logger
)

logger.info("Experiment finished.")