import torch
import numpy as np

from typing import Optional

from transformers import PreTrainedTokenizer

from src.utils.datasets import get_preprocessing_layers
from src.utils.preprocessing import get_preprocessing

import json
import math
import random
from pathlib import Path
import os

from logging import Logger

import random

def get_collate_function(
        special_tokens_id,
        parameter_token_id,
        augment=True,
        preprocess=None
    ):

    if preprocess != None:
        do_preprocessing = True
        preprocessor = get_preprocessing(preprocess)
    else:
        do_preprocessing = False
        preprocessor = None

    @torch.no_grad()
    def collate_function(examples):
        _lc_values: list[torch.FloatTensor] = [example["lc_values"] for example in examples]
        lc_values = []

        input_ids = torch.stack([example["input_ids"] for example in examples])

        patch_shapes = torch.stack([example["patch_shape"] for example in examples])
        lc_shapes = torch.stack([example["lc_shape"] for example in examples])
        
        t_init: list[torch.LongTensor] = [example["t_init"] for example in examples]
        t_max: list[torch.LongTensor] = [example["t_max"] for example in examples]
        t_margin: list[torch.LongTensor] = [example["t_margin"] for example in examples]

        labels = []

        _t_indices = [example["t_indices"] for example in examples]

        for i in range(len(_lc_values)):
            if augment:
            # Augment
                _t_max = t_max[i].item()
                _t_margin = t_margin[i].item()
                _offset = random.randint(0, _t_margin)
                _lc_values[i] = _lc_values[i][_offset:_t_max+_offset]

                _t_indices[i] = _t_indices[i] + float(_offset)

                lc_shape = _lc_values[i].shape[:3]
                k = random.randint(0, 3)
                _lc_values[i] = torch.rot90(_lc_values[i], k, dims=[1,2])
                # Mirror:
                if random.choice([True, False]):
                    _lc_values[i] = _lc_values[i].transpose(1,2)
                # Shift
                roll = random.randint(0, lc_shape[1] - 1)
                _lc_values[i] = _lc_values[i].roll(roll, dims=1)
                roll = random.randint(0, lc_shape[2] - 1)
                _lc_values[i] = _lc_values[i].roll(roll, dims=2)
            else:
                _t_max = t_max[i].item()
                _lc_values[i] = _lc_values[i][0:_t_max]
            
            patch_shape = patch_shapes[i]
            patched_lc_value, _ = _patch_lc(_lc_values[i], patch_shape)
            labels.append(patched_lc_value[t_init[i]:].reshape(-1, math.prod(patch_shape)))
            lc_values.append(patched_lc_value.reshape(-1, math.prod(patch_shape)))

        lc_values = torch.cat(lc_values, dim=0).to(dtype=torch.float32)
        labels = torch.cat(labels, dim=0).to(dtype=torch.float32)

        if do_preprocessing:
            lc_values = preprocessor.forward(lc_values)
            labels = preprocessor.forward(labels)

        parameter_values = torch.stack([example["parameter_values"] for example in examples])
        t_indices = torch.cat([t_index for t_index in _t_indices])

        special_tokens_pos = {}
        for tok in special_tokens_id:
            special_tokens_pos[tok + '_pos_0'] = torch.cat([example[tok + '_pos_0'] + i for i, example in enumerate(examples)])
            special_tokens_pos[tok + '_pos_1'] = torch.cat([example[tok + '_pos_1'] for example in examples])

        parameter_tokens_args = {}
        for i, tok in enumerate(parameter_token_id):
            parameter_tokens_args[tok + '_pos_0'] = torch.cat([example[tok + '_pos_0'] + i for i, example in enumerate(examples)])
            parameter_tokens_args[tok + '_pos_1'] = torch.cat([example[tok + '_pos_1'] for example in examples])
            parameter_tokens_args[tok + '_vals'] = parameter_values[:, i].unsqueeze(-1)

        lc_positions_0 = torch.cat([example["lc_positions_0"] + i for i, example in enumerate(examples)])
        lc_positions_1 = torch.cat([example["lc_positions_1"] for example in examples])

        lc_patch_pos_0 = torch.cat([example["lc_patch_pos_0"] + i for i, example in enumerate(examples)])
        lc_patch_pos_1 = torch.cat([example["lc_patch_pos_1"] for example in examples])

        cache_position = torch.arange(0, input_ids.size(1))

        t_indices = t_indices.unsqueeze(-1)
        if "t_index_pos_0" in examples[0]:
            t_index_pos_0 = torch.cat([example["t_index_pos_0"] + i for i, example in enumerate(examples)])
            t_index_pos_1 = torch.cat([example["t_index_pos_1"] for example in examples])
        else:
            t_index_pos_0 = None
            t_index_pos_1 = None

        lc_shapes = torch.stack([example["lc_shape"] for example in examples])

        output = {
            "input_ids": input_ids,
            "lc_values": lc_values,
            "labels": labels,
            "t_indices": t_indices,
            "lc_positions_0": lc_positions_0,
            "lc_positions_1": lc_positions_1,
            "lc_patch_pos_0": lc_patch_pos_0,
            "lc_patch_pos_1": lc_patch_pos_1,
            "patch_shapes": patch_shapes,
            "cache_position": cache_position,
            "lc_shapes": lc_shapes,
            "t_init": torch.cat(t_init),
            "t_max": torch.cat(t_max),
            **special_tokens_pos,
            **parameter_tokens_args
        }

        if t_index_pos_0 != None:
            output["t_index_pos_0"] = t_index_pos_0
            output["t_index_pos_1"] = t_index_pos_1

        if "lc_id" in examples[0]:
            output["lc_id"] = torch.stack([example["lc_id"] for example in examples])

        return output
    
    return collate_function

def get_collate_function_generation(
        special_tokens_id,
        parameter_token_id,
        preprocess=None,
        custom_t_max=None
    ):

    if preprocess != None:
        do_preprocessing = True
        preprocessor = get_preprocessing(preprocess)
    else:
        do_preprocessing = False
        preprocessor = None

    @torch.no_grad()
    def collate_function(examples):
        lc_values: list[torch.FloatTensor] = [example["lc_values"] for example in examples]

        patch_shape = examples[0]["patch_shape"]
        lc_shape = examples[0]["lc_shape"]

        input_ids = torch.stack([example["input_ids"] for example in examples])
        
        lcs_init = []
        lcs_remaining = []

        t_init = [example["t_init"] for example in examples]
        t_max = [example["t_max"] for example in examples]

        t_indices = torch.cat([example["t_indices"] for example in examples])

        for i in range(len(lc_values)):
            _t_init = t_init[i]
            _t_max = t_max[i] if custom_t_max == None else custom_t_max
            _lc_value = lc_values[i][0:_t_max]
            
            patched_lc_value, _ = _patch_lc(_lc_value, patch_shape)
            lcs_init.append(patched_lc_value[:_t_init])
            lcs_remaining.append(patched_lc_value[_t_init:])

        lcs_init = torch.stack(lcs_init).to(dtype=torch.float32)
        lcs_remaining = torch.stack(lcs_remaining).to(dtype=torch.float32)

        if do_preprocessing:
            lcs_init = preprocessor.forward(lcs_init)
            lcs_remaining = preprocessor.forward(lcs_remaining)
        
        parameter_values = torch.stack([example["parameter_values"] for example in examples])

        parameter_tokens_args = {}
        for i, tok in enumerate(parameter_token_id):
            parameter_tokens_args[tok + '_vals'] = parameter_values[:, i].unsqueeze(-1)
            parameter_tokens_args[tok + '_pos_0'] = torch.cat([example[tok + '_pos_0'] + i for i, example in enumerate(examples)])
            parameter_tokens_args[tok + '_pos_1'] = torch.cat([example[tok + '_pos_1'] for example in examples])

        lc_patch_pos_0 = torch.cat([example["lc_patch_pos_0"] + i for i, example in enumerate(examples)])
        lc_patch_pos_1 = torch.cat([example["lc_patch_pos_1"] for example in examples])

        t_indices = t_indices.unsqueeze(-1)
        t_index_pos_0 = torch.cat([example["t_index_pos_0"] + i for i, example in enumerate(examples)])
        t_index_pos_1 = torch.cat([example["t_index_pos_1"] for example in examples])
                
        output = {
            "input_ids": input_ids,
            "lcs_init": lcs_init,
            "lcs_remaining": lcs_remaining,
            "t_indices": t_indices,
            "lc_patch_pos_0": lc_patch_pos_0,
            "lc_patch_pos_1": lc_patch_pos_1,
            "patch_shape": patch_shape,
            "lc_shape": lc_shape,
            "t_init": torch.cat(t_init),
            "t_max": torch.cat(t_max),
            "t_index_pos_0": t_index_pos_0,
            "t_index_pos_1": t_index_pos_1,
            **parameter_tokens_args
        }

        if "lc_id" in examples[0]:
            output["lc_id"] = torch.stack([example["lc_id"] for example in examples])

        return output
    
    return collate_function

@torch.no_grad()
def _preprocess_lc(
        lc,
        patch_shape,
        do_preprocessing,
        preprocessor
):
    lc = lc.permute(2,0,1)
    lc = lc.flip(0)

    if do_preprocessing:
        lc = preprocessor.forward(lc)

    return _patch_lc(lc, patch_shape)

@torch.no_grad()
def _patch_lc(
        lc,
        patch_shape
):
    num_patches = [
        lc.shape[0] // patch_shape[0],
        lc.shape[1] // patch_shape[1],
        lc.shape[2] // patch_shape[2],
        ]

    lc = lc[
        :num_patches[0] * patch_shape[0],
        :num_patches[1] * patch_shape[1],
        :num_patches[2] * patch_shape[2]
    ]

    lc = lc.reshape(
        num_patches[0],
        patch_shape[0],
        num_patches[1],
        patch_shape[1],
        num_patches[2],
        patch_shape[2]
    ).permute(0, 2, 4, 1, 3, 5).reshape(*num_patches, -1)
    return lc, num_patches

@torch.no_grad()
def _patch_lc_batch(
        lc,
        patch_shape
):
    num_patches = [
        lc.shape[1] // patch_shape[0],
        lc.shape[2] // patch_shape[1],
        lc.shape[3] // patch_shape[2],
        ]

    lc = lc[
        :,
        :num_patches[0] * patch_shape[0],
        :num_patches[1] * patch_shape[1],
        :num_patches[2] * patch_shape[2]
    ]

    lc = lc.reshape(
        -1,
        num_patches[0],
        patch_shape[0],
        num_patches[1],
        patch_shape[1],
        num_patches[2],
        patch_shape[2]
    ).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, *num_patches, math.prod(patch_shape))
    return lc, num_patches
    
@torch.no_grad()
def _unpatch_lc(
        lc,
        patch_shape
    ):
    num_patches = lc.shape[:-1]
    lc = lc.reshape(
        *num_patches,
        *patch_shape
    ).permute(0, 3, 1, 4, 2, 5).reshape(
        num_patches[0] * patch_shape[0],
        num_patches[1] * patch_shape[1],
        num_patches[2] * patch_shape[2],
    )
    return lc

class FCStaticDataConfig:
    def __init__(
            self,
            t_max,
            t_init,
            t_margin,
            label_names,
            minimalistic_chat_template,
            pad_sequence,
            patch_size,
            lc_shape,
            special_tokens_id,
            parameter_token_id,
            t_index_id
            ):
        self.t_max = t_max
        self.t_init = t_init
        self.t_margin = t_margin
        self.label_names = label_names
        self.minimalistic_chat_template = minimalistic_chat_template
        self.pad_sequence = pad_sequence
        self.patch_size = patch_size
        self.lc_shape = lc_shape
        self.special_tokens_id = special_tokens_id
        self.parameter_token_id = parameter_token_id
        self.t_index_id = t_index_id

class FCStaticData:
    def __init__(
            self,
            config: FCStaticDataConfig,
            data: dict
            ):
        self.config = config
        self.data = data

    @classmethod
    def generate(
        cls,
        t_max,
        t_init,
        t_margin,
        label_names,
        minimalistic_chat_template,
        pad_sequence,
        patch_size,
        lc_shape,
        processor,
        model_config,
        special_tokens_id,
        parameter_token_id,
        t_index_id,
        logger=None
    ):
        data = get_static_inputs(
            processor=processor,
            label_names=label_names,
            lc_shape=lc_shape,
            patch_size=patch_size,
            t_max=t_max,
            t_init=t_init,
            t_margin=t_margin,
            special_tokens_id=special_tokens_id,
            parameter_token_id=parameter_token_id,
            t_index_id=t_index_id,
            pad_sequence=pad_sequence,
            minimalistic_chat_template=minimalistic_chat_template,
            logger=logger
        )
        fc_static_data = cls(
            config = FCStaticDataConfig(
                t_max=t_max,
                t_init=t_init,
                t_margin=t_margin,
                label_names=label_names,
                minimalistic_chat_template=minimalistic_chat_template,
                pad_sequence=pad_sequence,
                patch_size=patch_size,
                lc_shape=lc_shape,
                special_tokens_id=special_tokens_id,
                parameter_token_id=parameter_token_id,
                t_index_id=t_index_id
            ),
            data=data
        )
        return fc_static_data

    def save(self, dir:str):
        with open(os.path.join(dir, "config.json"), "w") as f:
            json.dump(self.to_dict(), f)
        torch.save(self.data, os.path.join(dir, "data.pt"))
        
    @classmethod
    def load(cls, dir:str) -> "FCStaticData":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config_dict = json.load(f)
        config = FCStaticDataConfig.from_dict(config_dict)
        data = torch.load(os.path.join(dir, "data.pt"), weights_only=True)
        return cls(config=config, data=data)

    @torch.no_grad()
    def batch(self, batch_size):
        examples = [self.data for _ in range(batch_size)]

        input_ids = torch.stack([example["input_ids"] for example in examples])

        patch_shapes = torch.stack([example["patch_shape"] for example in examples])
        
        t_init: list[torch.LongTensor] = [example["t_init"] for example in examples]
        t_max: list[torch.LongTensor] = [example["t_max"] for example in examples]

        special_tokens_pos = {}
        for tok in self.config.special_tokens_id:
            special_tokens_pos[tok + '_pos_0'] = torch.cat([example[tok + '_pos_0'] + i for i, example in enumerate(examples)])
            special_tokens_pos[tok + '_pos_1'] = torch.cat([example[tok + '_pos_1'] for example in examples])

        parameter_tokens_args = {}
        for i, tok in enumerate(self.config.parameter_token_id):
            parameter_tokens_args[tok + '_pos_0'] = torch.cat([example[tok + '_pos_0'] + i for i, example in enumerate(examples)])
            parameter_tokens_args[tok + '_pos_1'] = torch.cat([example[tok + '_pos_1'] for example in examples])
            
        lc_positions_0 = torch.cat([example["lc_positions_0"] + i for i, example in enumerate(examples)])
        lc_positions_1 = torch.cat([example["lc_positions_1"] for example in examples])

        lc_patch_pos_0 = torch.cat([example["lc_patch_pos_0"] + i for i, example in enumerate(examples)])
        lc_patch_pos_1 = torch.cat([example["lc_patch_pos_1"] for example in examples])

        cache_position = torch.arange(0, input_ids.size(1))

        t_index_pos_0 = torch.cat([example["t_index_pos_0"] + i for i, example in enumerate(examples)])
        t_index_pos_1 = torch.cat([example["t_index_pos_1"] for example in examples])

        lc_shapes = torch.stack([example["lc_shape"] for example in examples])

        output = {
            "input_ids": input_ids,
            "lc_positions_0": lc_positions_0,
            "lc_positions_1": lc_positions_1,
            "lc_patch_pos_0": lc_patch_pos_0,
            "lc_patch_pos_1": lc_patch_pos_1,
            "t_index_pos_0": t_index_pos_0,
            "t_index_pos_1": t_index_pos_1,
            "patch_shapes": patch_shapes,
            "cache_position": cache_position,
            "lc_shapes": lc_shapes,
            "t_init": torch.cat(t_init),
            "t_max": torch.cat(t_max),
            **special_tokens_pos,
            **parameter_tokens_args
        }

        if "lc_id" in examples[0]:
            output["lc_id"] = torch.stack([example["lc_id"] for example in examples])

        return output

class FCDatasetConfig:
    def __init__(
            self,
            label_names,
            t_init,
            t_end,
            preprocess,
            drop_last_slice,
            trim=False
            ):
        self.label_names = label_names
        self.t_init = t_init
        self.t_end = t_end
        self.preprocess = preprocess
        self.drop_last_slice = drop_last_slice
        self.trim = trim

    def to_dict(self) -> dict:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, dict_data: dict) -> "FCDatasetConfig":
        return cls(
            label_names=dict_data["label_names"],
            t_init=dict_data["t_init"],
            t_end=dict_data["t_end"],
            preprocess=dict_data["preprocess"],
            drop_last_slice=dict_data["drop_last_slice"],
            trim=dict_data.get("trim", False)
        )

    def save(self, dir):
        with open(os.path.join(dir, "ds_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, dir) -> "FCDatasetConfig":
        with open(os.path.join(dir, "ds_config.json"), "r") as f:
            _dict = json.load(f)
        return cls.from_dict(_dict)

class FCDataset(torch.utils.data.dataset.Dataset):
    def __init__(
            self,
            config: FCDatasetConfig,
            ds: list[dict],
            static_input: Optional[dict] = None,
            clone_static_input: bool = False
        ):
        self.config = config

        self.ds: list[dict] = ds
        
        self.static_input: dict = static_input
        self.clone_static_input: bool = clone_static_input

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if self.clone_static_input:
            _cp_static_input = { k: v.clone().detach() for k, v in self.static_input.items() }
            return {
                **_cp_static_input
                **self.ds[idx]
            }
        else:
            return {
                **self.static_input,
                **self.ds[idx]
            }
    
    @classmethod
    def load_from_generator(cls, config: FCDatasetConfig, id_files: list[tuple[int, str]], logger: Logger, clone_static_input:bool=False):
        generator = gen_pure_lc_data(
            id_files = id_files,
            label_names=config.label_names,
            t_init=config.t_init,
            t_end=config.t_end,
            preprocess=config.preprocess,
            add_lc_id=True,
            drop_last_slice=config.drop_last_slice,
            trim=config.trim,
            logger=logger
        )

        ds = [el for _, _, el in generator]
        return cls(config=config, ds=ds, static_input=None, clone_static_input=clone_static_input)

@torch.no_grad()
def get_static_inputs(
    processor,
    label_names,
    lc_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    t_max,
    t_init,
    t_margin,
    special_tokens_id,
    parameter_token_id,
    t_index_id,
    pad_sequence=False,
    minimalistic_chat_template=False,
    logger=None,
) -> dict:
    num_patches = [t_max] + [lc_s // p_s for lc_s, p_s in zip(lc_shape[1:], patch_size[1:])]

    return _get_static_inputs(
        patch_size=patch_size,
        num_patches=num_patches,
        t_max=t_max,
        t_init=t_init,
        t_margin=t_margin,
        processor=processor,
        pad_sequence=pad_sequence,
        label_names=label_names,
        special_tokens_id=special_tokens_id,
        parameter_token_id=parameter_token_id,
        t_index_id=t_index_id,
        minimalistic_chat_template=minimalistic_chat_template,
        logger=logger
    )

@torch.no_grad()
def _get_static_inputs(
    patch_size,
    num_patches,
    t_max,
    t_init,
    t_margin,
    processor,
    pad_sequence,
    label_names,
    special_tokens_id,
    parameter_token_id,
    t_index_id,
    minimalistic_chat_template=False,
    logger=None,
) -> dict:
    if logger != None:
        logger.info(f"Generating static input data with patch_shape {patch_size}, t_max {t_max}, t_int {t_init} and t_margin {t_margin}.")

    patch_size_pt = torch.tensor([*patch_size], dtype=torch.long)
    lc_patch_dim = math.prod(patch_size)

    lc_shape_init = (t_init, num_patches[1], num_patches[2], lc_patch_dim)
    lc_shape_remaining = (t_max- t_init, num_patches[1], num_patches[2], lc_patch_dim)

    lc_dummy_init = torch.zeros(*lc_shape_init)
    lc_dummy_remaining = torch.zeros(*lc_shape_remaining)

    if not minimalistic_chat_template:
        parameter_string = "".join([f"<|parameter_{i}|>" for i, parameter in enumerate(label_names)])
        system_prompt = "<|lc_end|>"
        base_message = [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": f"{parameter_string}<|t_index|>" }
        ]
        base_prompt = processor.tokenizer.apply_chat_template(
            base_message,
            tokenize=False,
            add_generation_prompt=True
        )
        base_prompt += "<|lc_start|><|lightcone_1|><|lightcone_2|>"
    else:
        parameter_string = "".join([f"<|parameter_{i}|>" for i in range(len(label_names))])
        base_prompt = f"{parameter_string}<|lc_start|><|lightcone_1|><|lc_end|><|t_index|><|lc_start|><|lightcone_2|>"

    inputs = processor(
        base_prompt,
        [lc_dummy_init, lc_dummy_remaining],
        t_index_start=torch.tensor([0.]),
        return_tensors="pt"
        )
    input_ids = inputs['input_ids'].squeeze(dim=0)

    if logger != None:
        logger.info(f"prompt before replacing tokens {json.dumps(base_prompt)}")
        logger.info(f"input_ids: {json.dumps(input_ids.tolist())}")
        logger.info(f"sequence length: {input_ids.size(0)}")

    if pad_sequence:
        tokenizer: PreTrainedTokenizer = processor.tokenizer
        seq_length = len(input_ids)
        if pad_sequence:
            padded_seq_length = math.ceil(seq_length / 8) * 8
        else:
            padded_seq_length = seq_length
        if logger != None:
            logger.info(f"Sequence length: {json.dumps(seq_length)}, padded_sequence_length: {padded_seq_length}")
        
        processor.tokenizer.padding_side = "right"
        pad_config = {
            'padding': "max_length",
            'max_length': padded_seq_length
        }

        if logger != None:
            logger.info(f"Padding config has been set to: {json.dumps(pad_config)}")

        input_ids = tokenizer.pad({"input_ids": input_ids}, **pad_config, return_tensors="pt")
        input_ids: torch.LongTensor = input_ids["input_ids"]
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)

    if logger != None:
        logger.info(f"prompt={json.dumps(base_prompt)}")
        logger.info(f"Sequence length: {len(input_ids)}")
        logger.info(f"lc_shape={num_patches}, t_max={t_max}, t_init={t_init}")
        logger.info(f"input_ids: {json.dumps(input_ids.tolist())}")
        logger.info(f"image_tokens={torch.sum(input_ids < 0)}")
        logger.info(f"init_image_tokens={torch.sum(input_ids == -1)}")
        logger.info(f"remaining_image_tokens={torch.sum(input_ids == -2)}")

    output = {
        'input_ids': input_ids,
        'patch_shape': patch_size_pt,
        "t_init": torch.tensor([t_init], dtype=torch.long),
        "t_max": torch.tensor([t_max], dtype=torch.long),
        "t_margin": torch.tensor([t_margin], dtype=torch.long),
        "lc_shape": torch.tensor([num_patches[0], num_patches[1], num_patches[2], lc_patch_dim], dtype=torch.long)
    }

    for tok, id in special_tokens_id.items():
        token_mask = (inputs['input_ids'] == id)
        pos_0, pos_1 = torch.nonzero(token_mask, as_tuple=True)
        output[tok + '_pos_0'] = pos_0
        output[tok + '_pos_1'] = pos_1

    for tok, id in parameter_token_id.items():
        token_mask = (inputs['input_ids'] == id)
        pos_0, pos_1 = torch.nonzero(token_mask, as_tuple=True)
        output[tok + '_pos_0'] = pos_0
        output[tok + '_pos_1'] = pos_1

    output_lc_patch_mask = (inputs['input_ids'] == -2)
    # shift for next token prediction
    output_lc_patch_mask = torch.cat(
        [
            output_lc_patch_mask[:, 1:],
            torch.tensor([False]*output_lc_patch_mask.size(0)).unsqueeze(-1)
        ],
        dim=1)
    output_lc_patch_pos = torch.nonzero(output_lc_patch_mask, as_tuple=True)
    output["lc_patch_pos_0"] = output_lc_patch_pos[0]
    output["lc_patch_pos_1"] = output_lc_patch_pos[1]

    # lightcone positions
    input_shape = inputs['input_ids'].size()
    _input_ids = inputs['input_ids'].view(-1, input_shape[-1])
    positions = torch.nonzero((_input_ids < 0) & (_input_ids > -int(1e9)), as_tuple=True)
    output["lc_positions_0"] = positions[0]
    output["lc_positions_1"] = positions[1]

    if t_index_id != None:
        t_indices_mask = (inputs['input_ids'] == t_index_id)
        t_indices_pos_0, t_indices_pos_1 =  torch.nonzero(t_indices_mask, as_tuple=True)
        output['t_index_pos_0'] = t_indices_pos_0
        output['t_index_pos_1'] = t_indices_pos_1
        
    return output

@torch.no_grad()
def gen_pure_lc_data(
        id_files,
        label_names,
        t_init,
        t_end,
        preprocess=None,
        add_lc_id: bool=True,
        drop_last_slice=False,
        trim=False,
        logger=None,
        ):
    
    if logger != None:
        logger.info(f"Generating data with t_init={t_init}, t_margin={t_end}, preprocess={preprocess}, add_lc_id={add_lc_id}, drop_last_slice={drop_last_slice}")

    preprocessing = get_preprocessing_layers(label_names)

    if preprocess != None:
        do_preprocessing = True
        preprocessor = get_preprocessing(preprocess)
    else:
        do_preprocessing = False
        preprocessor = None

    logged_once = False

    for lc_id, f in id_files:
        with np.load(f) as record:
            lc = torch.from_numpy(record["image"]).to(torch.float32)
            
            lc = lc.permute(2,0,1)
            lc = lc.flip(0)

            if trim:
                sq_global_lc = torch.mean(lc**2, dim=(1,2))

                end = sq_global_lc.shape[0]
                while sq_global_lc[end-1] == 0:
                    end = end - 1
                end = end + 2 * t_end

                lc = lc[:end]

            if do_preprocessing:
                lc = preprocessor.forward(lc)

            parameter_values = torch.from_numpy(record["label"]).to(torch.float32)[-len(label_names):]
            parameter_values = preprocessing.forward(parameter_values)

            for i_t, output in _gen_pure_lc_data(
                lc,
                t_init,
                t_end,
                parameter_values,
                drop_last_slice=drop_last_slice,
                log_once=not logged_once,
                t_start = 0,
                logger=logger,
                ):

                if add_lc_id:
                    output["lc_id"] = torch.tensor(lc_id, dtype=torch.long)

                yield lc_id, i_t, output
                
            logged_once = True

@torch.no_grad()
def _gen_pure_lc_data(
        lc,
        t_init,
        t_end,
        parameter_values,
        drop_last_slice=False,
        log_once=False,
        t_start = 0,
        logger=None,
):
    if log_once:
        assert logger != None
        logged_once = False

    outputs = []

    for i_t in range(0, lc.shape[0]-t_init, t_end-t_init):
        if i_t + t_end > lc.shape[0]:
            if drop_last_slice:
                continue
            i_t_start = lc.shape[0] - t_end
        else:
            i_t_start = i_t

        sublc = lc[i_t_start:i_t_start+t_end]
        sublc = sublc.to(dtype=torch.float16)
        
        if log_once and not logged_once:
            if logger != None:
                logger.info(f"lc_shape={sublc.shape}, t_end={t_end}, t_init={t_init}, t_margin={t_end}")

        output = {
            'lc_values': sublc,
            'parameter_values': parameter_values
        }

        output["t_indices"] = torch.tensor([i_t + t_start], dtype=torch.float32)

        outputs.append((i_t, output))

        if log_once and not logged_once:
            logged_once = True
        
    return outputs
