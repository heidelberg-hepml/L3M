import re
from typing import List, Optional, Union

import torch
import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy

from transformers.image_processing_utils import BatchFeature
from transformers.utils import TensorType, logging

from .configuration import L3MGenerationConfig

import torch

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import (
    ImageInput,
    make_list_of_images,
    is_numpy_array,
    is_torch_tensor
)
import transformers
from transformers import AutoImageProcessor

import math

logger = logging.get_logger(__name__)

class L3MLightconeProcessorGeneration(BaseImageProcessor):
    model_input_names = ["images"]

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def calc_num_image_tokens(
            self, 
            images: ImageInput
    ):
        images = make_list_of_images(images)

        if is_numpy_array(images[0]):
            images = [torch.from_numpy(img) for img in images]
        elif not is_torch_tensor(images[0]):
            raise ValueError(
                "Invalid image type. Must be of type numpy.ndarray or torch.Tensor."
            )
        
        patched_shapes = [[im.size(i) for i in range(0, im.dim()-1)] for im in images]
        num_img_tokens = [
            math.prod(im.shape[:-1]) + 
            patched_shape[0] +                  # newline token
            patched_shape[0] * patched_shape[1] # newline token
            for im, patched_shape in zip(images, patched_shapes)
        ]

        return num_img_tokens

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        images = make_list_of_images(images)

        if is_numpy_array(images[0]):
            images = [torch.from_numpy(im) for im in images]
        elif is_torch_tensor(images[0]):
            pass
        else:
            raise ValueError(
                "Invalid image type. Must be of type numpy.ndarray or torch.Tensor."
            )
        
        num_img_tokens = self.calc_num_image_tokens(images=images)

        return {
            "lc_values": images,
            "num_img_tokens": num_img_tokens
        }

AutoImageProcessor.register("L3MLightconeProcessorGeneration", L3MLightconeProcessorGeneration)
transformers.L3MLightconeProcessorGeneration = L3MLightconeProcessorGeneration

class L3MProcessorGeneration(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["nl_1_id", "nl_2_id", "t_index_id"]
    image_processor_class = "L3MLightconeProcessorGeneration"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    special_image_token = "<|lightcone|>"

    def __init__(self, image_processor, tokenizer, nl_1_id=-1, nl_2_id=-1, t_index_id=-1):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = None

        self.nl_1_id = nl_1_id
        self.nl_2_id = nl_2_id
        self.t_index_id = t_index_id

    def set_ids_from_config(self, config: L3MGenerationConfig):
        for special_token in config.special_token:
            if special_token['name'] == 'nl_1':
                self.nl_1_id = special_token['token_id']
            if special_token['name'] == 'nl_2':
                self.nl_2_id = special_token['token_id']
            if special_token['name'] == 'r':
                self.r_id = special_token['token_id']
        self.t_index_id = config.t_index_token['token_id']

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        lc_values: ImageInput = None,
        t_index_start: Optional[torch.LongTensor] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if lc_values is not None:
            lc_inputs = self.image_processor(lc_values, return_tensors=return_tensors)
        else:
            lc_inputs = {}
        inputs = self._convert(
            lc_inputs,
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            t_index_start=t_index_start,
        )
        return inputs

    def calc_num_image_tokens(self, images: ImageInput):
        return self.image_processor.calc_num_image_tokens(images)
        
    def _convert(self, images, texts, padding=False, truncation=None, max_length=None, return_tensors=None, t_index_start=None):
        if not len(images):
            model_inputs = self.tokenizer(texts, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length)
            return BatchFeature(data={**model_inputs})

        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list)
        if len(texts) > 0:
            assert isinstance(texts[0], str)

        pattern = r"<\|lightcone_\d+\|>"
        prompt_pieces = [[self.tokenizer(prompt_piece).input_ids for prompt_piece in re.split(pattern, text)] for text in texts] 

        if 'num_img_tokens' in images:
            num_img_tokens = images['num_img_tokens']
        else:
            num_img_tokens = self.calc_num_image_tokens(images['lc_values'])

        lcs = images['lc_values']

        lc_tags = [re.findall(pattern, text) for text in texts]
        lc_ids = [[int(lc_tag.split("|")[1].split("_")[-1]) for lc_tag in _lc_tags] for _lc_tags in lc_tags]
        lc_ids_pad = [[ self.set_lc_image_ids(
            id,
            num_img_tokens[id-1],
            lcs[id-1].shape[:-1]
            ) for id in _lc_ids] for _lc_ids in lc_ids]
        
        def _interleave_lists(a, b):
            if len(a) > len(b):
                b.append([])
            return [element for pair in zip(a, b) for element in pair]
        
        input_ids = []
        for _prompt_piece, _lc_ids_pad in zip(prompt_pieces, lc_ids_pad):
            _input_ids = []
            for x in _interleave_lists(_prompt_piece, _lc_ids_pad):
                _input_ids.extend(x)
            input_ids.append(_input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        lc_values = torch.cat([lc.reshape(-1, lc.size(-1)) for lc in lcs], dim=0)

        return BatchFeature(data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lc_values": lc_values
            })

    def set_lc_image_ids(self, iid, num_img_tokens, lc_shape):
        input_ids = [-iid]*num_img_tokens

        i = 0
        for ti in range(lc_shape[0]):
            for xi in range(lc_shape[1]):
                for yi in range(lc_shape[2]):
                    i += 1
                
                input_ids[i] = self.nl_2_id
                i += 1
            input_ids[i] = self.nl_1_id
            i += 1
                
        return input_ids

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
