import torch

import os, gc
import logging
from glob import glob
import json

import sys

from typing import Union, Optional

from src.experiments.regression.network.modeling import L3MRegressor

def freeze_llm(model: L3MRegressor):
    model.requires_grad_(False)

    model.lc_embedding.requires_grad_(True)
    model.lc_embedding.wte.requires_grad_(False)
    model.param_head.requires_grad_(True)
    for tok in model.special_token_embds.keys():
        model.special_token_embds[tok].requires_grad_(True)

def split_files(files, ratio):
    n_train = int(float(len(files)) * ratio)
    train_files = files[:n_train]
    eval_files = files[n_train:]
    return train_files, eval_files
    
def create_optimizer_groups(
        opt_model,
        decay_parameters,
        lr,
        weight_decay,
        lr_head=1.,
        lr_others=0.5,
        lr_tok=0.2,
        lr_lc=0.01,
        logger=None
):
    wd_lc = []
    wd_tok = []
    wd_others = []
    wd_head = []

    nwd_lc = []
    nwd_tok = []
    nwd_others = []
    nwd_head = []

    for n, p in opt_model.named_parameters():
        if p.requires_grad:
            if n in decay_parameters:
                if "lc_connector" in n:
                    wd_lc.append(p)
                elif "special_token_embds" in n:
                    wd_tok.append(p)
                elif "param_head" in n:
                    wd_head.append(p)
                else:
                    wd_others.append(p)
            else:
                if "lc_connector" in n:
                    nwd_lc.append(p)
                elif "special_token_embds" in n:
                    nwd_tok.append(p)
                elif "param_head" in n:
                    nwd_head.append(p)
                else:
                    nwd_others.append(p)

    if logger != None:
        logger.info(f"Number of weight decay parameters: lc: {len(wd_lc)}, token: {len(wd_tok)}, head: {len(wd_head)}, others: {len(wd_others)}")
        logger.info(f"Number of non weight decay parameters: lc: {len(nwd_lc)}, token: {len(nwd_tok)}, head: {len(nwd_head)}, others: {len(nwd_others)}") 

    optimizer_grouped_parameters = []
    if lr_others > 0.:
        optimizer_grouped_parameters += [
            {
                "params": wd_others,
                "weight_decay": weight_decay,
                "lr": lr_others * lr
            },
            {
                "params": nwd_others,
                "weight_decay": 0.0,
                "lr": lr_others * lr
            },
        ]
    if lr_lc > 0.:
        optimizer_grouped_parameters += [
            {
                "params": wd_lc,
                "weight_decay": weight_decay,
                "lr": lr_lc * lr
            },
            {
                "params": nwd_lc,
                "weight_decay": 0.0,
                "lr": lr_lc * lr
            },
        ]
    if lr_tok > 0.:
         optimizer_grouped_parameters += [
             {
                "params": wd_tok,
                "weight_decay": weight_decay,
                "lr": lr_tok * lr
            },
            {
                "params": nwd_tok,
                "weight_decay": 0.0,
                "lr": lr_tok * lr
            }
         ]
    if lr_head > 0.:
        optimizer_grouped_parameters += [
            {
                "params": wd_head,
                "weight_decay": weight_decay,
                "lr": lr_head * lr
            },
            {
                "params": nwd_head,
                "weight_decay": 0.0,
                "lr": lr_head * lr
            }
        ]

    return optimizer_grouped_parameters

