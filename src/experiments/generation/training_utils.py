import torch

import os, gc
import logging
from glob import glob
import json

import sys

from typing import Union, Optional

from src.experiments.generation.network.modeling import L3MGenerator
from src.experiments.generation.gen_exp_config import GenExpConfig
# import datasets
# from datasets import Dataset

def freeze_llm_embedding(model: L3MGenerator, exp_config: GenExpConfig):
    if not exp_config.lora['use']:
        model.embed_tokens.requires_grad_(False)
        model.lc_embedding.wte.requires_grad_(False)

def create_optimizer_groups(
        opt_model,
        decay_parameters,
        lr,
        weight_decay,
        lr_head=1.,
        lr_others=1.,
        lr_tok=1.,
        lr_lc=1.,
        lr_params=1.,
        logger=None
):
    wd_lc = []
    wd_tok = []
    wd_others = []
    wd_head = []
    wd_params = []

    wd_lc_names = []
    wd_tok_names = []
    wd_others_names = []
    wd_head_names = []
    wd_params_names = []

    nwd_lc = []
    nwd_tok = []
    nwd_others = []
    nwd_head = []
    nwd_params = []

    nwd_lc_names = []
    nwd_tok_names = []
    nwd_others_names = []
    nwd_head_names = []
    nwd_params_names = []

    for n, p in opt_model.named_parameters():
        if p.requires_grad:
            if n in decay_parameters:
                if "lc_embedding" in n:
                    wd_lc.append(p)
                    wd_lc_names.append(n)
                elif "special_token_embds" in n:
                    wd_tok.append(p)
                    wd_tok_names.append(n)
                elif "lc_head" in n or "lc_head_corrector" in n:
                    wd_head.append(p)
                    wd_head_names.append(n)
                elif "parameter_embedding" in n or "t_index_embedding" in n:
                    wd_params.append(p)
                    wd_params_names.append(n)
                else:
                    wd_others.append(p)
                    wd_others_names.append(n)
            else:
                if "lc_connector" in n:
                    nwd_lc.append(p)
                    nwd_lc_names.append(n)
                elif "special_token_embds" in n:
                    nwd_tok.append(p)
                    nwd_tok_names.append(n)
                elif "lc_head" in n or "lc_head_corrector" in n:
                    nwd_head.append(p)
                    nwd_head_names.append(n)
                elif "parameter_embedding" in n or "t_index_embedding" in n:
                    nwd_params.append(p)
                    nwd_params_names.append(n)
                else:
                    nwd_others.append(p)
                    nwd_others_names.append(n)

    if logger != None:
        logger.info(f"Number of weight decay parameters: lc: {len(wd_lc)}, token: {len(wd_tok)}, head: {len(wd_head)}, params: {len(wd_params)}, others: {len(wd_others)}")
        logger.info(f"Number of non weight decay parameters: lc: {len(nwd_lc)}, token: {len(nwd_tok)}, head: {len(nwd_head)}, params: {len(nwd_params)}, others: {len(nwd_others)}") 

    optimizer_grouped_parameters = []
    if lr_others > 0.:
        optimizer_grouped_parameters += [
            {
                "params": wd_others,
                "weight_decay": weight_decay,
                "lr": lr_others * lr,
                "param_names": wd_others_names
            },
            {
                "params": nwd_others,
                "weight_decay": 0.0,
                "lr": lr_others * lr,
                "param_names": nwd_others_names
            },
        ]
    if lr_lc > 0.:
        optimizer_grouped_parameters += [
            {
                "params": wd_lc,
                "weight_decay": weight_decay,
                "lr": lr_lc * lr,
                "param_names": wd_lc_names
            },
            {
                "params": nwd_lc,
                "weight_decay": 0.0,
                "lr": lr_lc * lr,
                "param_names": nwd_lc_names
            },
        ]
    if lr_tok > 0.:
         optimizer_grouped_parameters += [
             {
                "params": wd_tok,
                "weight_decay": weight_decay,
                "lr": lr_tok * lr,
                "param_names": wd_tok_names
            },
            {
                "params": nwd_tok,
                "weight_decay": 0.0,
                "lr": lr_tok * lr,
                "param_names": nwd_tok_names
            }
         ]
    if lr_head > 0.:
        optimizer_grouped_parameters += [
            {
                "params": wd_head,
                "weight_decay": weight_decay,
                "lr": lr_head * lr,
                "param_names": wd_head_names
            },
            {
                "params": nwd_head,
                "weight_decay": 0.0,
                "lr": lr_head * lr,
                "param_names": nwd_head_names
            }
        ]

    if lr_params > 0.:
        optimizer_grouped_parameters += [
            {
                "params": wd_params,
                "weight_decay": weight_decay,
                "lr": lr_params * lr,
                "param_names": wd_params_names
            },
            {
                "params": nwd_params,
                "weight_decay": 0.0,
                "lr": lr_params * lr,
                "param_names": nwd_params_names
            }
        ]

    return optimizer_grouped_parameters

def create_optimizer_groups_lora_only_first_and_last(
    opt_model,
    decay_parameters,
    lr,
    weight_decay,
    first_n,
    last_n,
    num_hidden_layers,
    logger=None
):
    input_names = ["lc_embedding", "special_token_embds", "parameter_embedding", "t_index_embedding"] + [f"layers.{i}." for i in range(first_n)]
    output_names = ["lc_head", "lc_head_corrector"] + [f"layers.{i}." for i in range(num_hidden_layers - last_n, num_hidden_layers)]

    def is_input_param(name: str) -> bool:
        for input_name in input_names:
            if input_name in name:
                return True
        return False
    def is_output_param(name: str) -> bool:
        for output_name in output_names:
            if output_name in name:
                return True
        return False

    wd_input = []
    nwd_input = []

    wd_output = []
    nwd_output = []

    input_param_names = []
    output_param_names = []

    for n, p in opt_model.named_parameters():
        if p.requires_grad:
            if n in decay_parameters:
                if is_input_param(n):
                    wd_input.append(p)
                    input_param_names.append(n)
                elif is_output_param(n):
                    wd_output.append(p)
                    output_param_names.append(n)
            else:
                if is_input_param(n):
                    nwd_input.append(p)
                    input_param_names.append(n)
                elif is_output_param(n):
                    nwd_output.append(p)
                    output_param_names.append(n)

    if logger != None:
        logger.info("Creating optimizer groups for lora with distinct input and output training")
        logger.info(f"Input parameters: {json.dumps(input_param_names)}")
        logger.info(f"Output parameters: {json.dumps(output_param_names)}")

    return [
        [
            {
                "params": wd_input,
                "weight_decay": weight_decay,
            },
            {
                "params": nwd_input,
                "weight_decay": 0.0,
            }
        ],
        [
            {
                "params": wd_output,
                "weight_decay": weight_decay,
            },
            {
                "params": nwd_output,
                "weight_decay": 0.0,
            }
        ],
    ]
        
def split_files(files, ratio):
    n_train = int(float(len(files)) * ratio)
    train_files = files[:n_train]
    eval_files = files[n_train:]
    return train_files, eval_files

def add_id_to_lcs(files):
    return [(id, file) for id, file in enumerate(files)]

def get_l3m(model, exp_config) -> L3MGenerator:
    if exp_config.lora['use']:
        return model.model
    else:
        return model
