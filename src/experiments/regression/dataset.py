import torch
import numpy as np

from transformers import AutoProcessor, PreTrainedTokenizer

from src.utils.datasets import get_preprocessing_layers
from src.utils.preprocessing import get_preprocessing

from src.experiments.regression.network.configuration import L3MRegressorConfig

import math

import json

import random

from typing import Optional, Tuple

def get_collate_function(
        special_tokens_id
    ):
    def collate_function(examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])

        patch_shapes = torch.stack([example["patch_shape"] for example in examples])
        
        lc_values: list[torch.FloatTensor] = [example["lc_values"] for example in examples]
        lc_values = torch.cat(lc_values, dim=0)
    
        labels = torch.stack([example["labels"] for example in examples])

        special_tokens_pos = {}
        for tok in special_tokens_id:
            special_tokens_pos[tok + '_pos_0'] = torch.cat([example[tok + '_pos_0'] + i for i, example in enumerate(examples)])
            special_tokens_pos[tok + '_pos_1'] = torch.cat([example[tok + '_pos_1'] for example in examples])

        positions_0 = torch.cat([example["positions_0"] + i for i, example in enumerate(examples)])
        positions_1 = torch.cat([example["positions_1"] for example in examples])

        return {
            "input_ids": input_ids,
            "positions_0": positions_0,
            "positions_1": positions_1,
            "lc_values": lc_values,
            "labels": labels,
            "patch_shapes": patch_shapes,
            **special_tokens_pos
        }
    
    return collate_function

def gen_lc_data(
        files,
        proc_dir,
        model_config: L3MRegressorConfig,
        label_names,
        patch_size: tuple[int, int, int],
        special_tokens_id,
        preprocess=None,
        pad_sequence=False,
        multiplicity:int = 1,
        dtype=torch.float32,
        logger=None,
        minimalistic_chat_template:bool=False
        ):
    if logger != None:
        logger.info(f"Generating data with preprocess: {preprocess}")
        logger.info(f"num_sys_tokens={model_config.num_sys_prompt_tokens}, num_pre_lc_tokens={model_config.num_pre_lc_tokens}")
        
    processor = AutoProcessor.from_pretrained(
        proc_dir,
        trust_remote_code=True,
        local_files_only=True
    )

    preprocessing = get_preprocessing_layers(label_names)

    if pad_sequence:
        pad_config = {}
    else:
        pad_config = { 'padding': False }

    if preprocess != None:
        do_preprocessing = True
        preprocessor = get_preprocessing(preprocess)
    else:
        do_preprocessing = False
        preprocessor = None

    logged_once = False

    for f in files:
        with np.load(f) as record:
            image = torch.from_numpy(record["image"]).to(dtype)
            image = image.permute(2,0,1)
            image = image.flip(0)

            labels = torch.from_numpy(record["label"]).to(dtype)[-len(label_names):]
            labels = preprocessing.forward(labels)

            if pad_config == {}:
                _outputs = _gen_outputs(
                    image,
                    patch_size,
                    model_config,
                    minimalistic_chat_template,
                    pad_config,
                    processor,
                    labels,
                    do_preprocessing,
                    preprocessor,
                    special_tokens_id=special_tokens_id,
                    log_once=True,
                    logger=logger
                )
                _input_ids = _outputs[0]["input_ids"]
                max_seq_length = len(_input_ids)
                if pad_sequence:
                    padded_seq_length = math.ceil(max_seq_length / 8) * 8
                else:
                    padded_seq_length = max_seq_length
                logger.info(f"Sequence length: {max_seq_length}, padded_sequence_length: {padded_seq_length}")
                
                processor.tokenizer.padding_side = "right"
                pad_config = {
                    'padding': "max_length",
                    'max_length': padded_seq_length
                }

                logger.info("Padding config has been setup.")

            for output in _gen_outputs(
                    image,
                    patch_size,
                    model_config,
                    minimalistic_chat_template,
                    pad_config,
                    processor,
                    labels,
                    do_preprocessing,
                    preprocessor,
                    special_tokens_id=special_tokens_id,
                    log_once=not logged_once,
                    logger=logger
                ):
                for _ in range(multiplicity):
                    yield output
                
            logged_once = True
                    
def _gen_outputs(
        image,
        patch_size,
        model_config,
        minimalistic_chat_template,
        pad_config,
        processor,
        labels,
        do_preprocessing,
        preprocessor,
        special_tokens_id,
        log_once=False,
        logger=None
):
    if log_once:
        assert logger != None
        logged_once = False

        logger.info(f"Generating data with patch shape {patch_size}.")

    outputs = []

    patch_size_pt = torch.tensor([1, *patch_size[1:]], dtype=torch.long)

    num_t_patches = image.size(0) // patch_size[0] 
    image = image.reshape(
        num_t_patches,
        patch_size[0],
        image.size(1),
        image.size(2)
    )
    image = torch.mean(image, dim=1, keepdim=True).permute(1, 0, 2, 3)

    lc_shape = (
        num_t_patches,
        image.shape[2] // patch_size[1],
        image.shape[3] // patch_size[2],
        1
    )

    sys_prompt = "".join([f"<|sys_prompt_{i}|>" for i in range(model_config.num_sys_prompt_tokens)])
    pre_lc_prompt = "".join([f"<|pre_lc_{i}|>" for i in range(model_config.num_pre_lc_tokens)])

    if not minimalistic_chat_template:
        messages = [
            { "role": "system", "content": sys_prompt },
            { "role": "user", "content": f"{pre_lc_prompt}<|lightcone_1|>" }
        ]
        
    for t_i in range(image.size(0)):
        lc = image[t_i]
        lc_dummy = torch.zeros(*lc_shape)

        if not minimalistic_chat_template:
            prompt = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt += f"<|ska_param|>"
        else:
            prompt = f"{sys_prompt}{pre_lc_prompt}<|lightcone_1|><|ska_param|>"
        
        inputs = processor(prompt, [lc_dummy])
        input_ids = inputs['input_ids'].squeeze(dim=0)

        if log_once and not logged_once:
            logger.info(f"Padding config: {json.dumps(pad_config)}")

        if pad_config != {}:
            tokenizer: PreTrainedTokenizer = processor.tokenizer
            input_ids = tokenizer.pad({"input_ids": input_ids}, **pad_config, return_tensors="pt")
            input_ids: torch.LongTensor = input_ids["input_ids"]
            if input_ids.dim() > 1:
                input_ids = input_ids.squeeze(0)

        if log_once and not logged_once:
            logger.info(f"prompt={json.dumps(prompt)}")
            logger.info(f"Sequence length: {len(input_ids)}")
            logger.info(f"image={image.shape}")
            logger.info(f"lc_shape={lc_shape}")
            logger.info(f"input_ids: {json.dumps(input_ids.tolist())}")
            logger.info(f"image_tokens={torch.sum(input_ids < 0)}")

        output = {
            'input_ids': input_ids,
            'labels': labels,
            'patch_shape': patch_size_pt
        }

        lc = _patch_lc(lc, patch_size_pt)
        lc_values = lc.reshape(-1, lc.size(-1))
        if do_preprocessing:
            lc_values = preprocessor.forward(lc_values)
        output['lc_values'] = lc_values

        for tok, id in special_tokens_id.items():
            token_mask = (inputs['input_ids'] == id)
            pos_0, pos_1 = torch.nonzero(token_mask, as_tuple=True)
            output[tok + '_pos_0'] = pos_0
            output[tok + '_pos_1'] = pos_1

        # lightcone positions
        input_shape = inputs['input_ids'].size()
        _input_ids = inputs['input_ids'].view(-1, input_shape[-1])
        positions = torch.nonzero((_input_ids < 0) & (_input_ids > -int(1e9)), as_tuple=True)
        output["positions_0"] = positions[0]
        output["positions_1"] = positions[1]

        outputs.append(output)

        if log_once and not logged_once:
            logged_once = True

            # for k, v in output.items():
            #     if v == None:
            #         print(f"Error generating dataset: {k} is NaN")
        
    return outputs

def _patch_lc(
        lc,
        patch_size
):
    num_patches = [
        lc.shape[0],
        lc.shape[1] // patch_size[1],
        1,
        lc.shape[2] // patch_size[2],
        1
        ]
    lc = lc.reshape(
        num_patches[0],
        num_patches[1],
        num_patches[2],
        patch_size[1],
        num_patches[3],
        num_patches[4],
        patch_size[2],
    ).permute(0, 1, 4, 2, 5, 3, 6).reshape(
        num_patches[0],
        num_patches[1],
        num_patches[3],
        num_patches[2] * num_patches[4],
        patch_size[1] * patch_size[2]
    )
    lc = torch.mean(lc, dim=-1, keepdim=False)
    return lc
