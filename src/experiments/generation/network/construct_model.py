
from typing import Tuple, Union

from transformers import AutoTokenizer, AutoConfig, Qwen2Config
from src.experiments.generation.network.modeling import L3MGenerator
from src.experiments.generation.network.configuration import L3MGenerationConfig
from src.experiments.generation.network.processing import L3MProcessorGeneration, L3MLightconeProcessorGeneration
import shutil

import os, json, shutil
from transformers.utils.__init__ import SAFE_WEIGHTS_INDEX_NAME
from transformers.modeling_utils import load_state_dict
import gc
import torch

from pathlib import Path

@torch.no_grad()
def construct_l3m_generator(
        LLM_DIR:str,
        SAVE_DIR:str,
        model_config: L3MGenerationConfig = None,
        load_llm_weights:bool=True,
        train_from_scratch:bool=False,
        dtype=torch.float32,
        device="cuda",
        logger=None,
    ) -> Tuple[L3MGenerator, L3MProcessorGeneration]:

    if model_config == None:
        model_config = L3MGenerationConfig

    if not train_from_scratch:
        qwen_config: Qwen2Config = AutoConfig.from_pretrained(LLM_DIR)
        model_config.vocab_size=qwen_config.vocab_size
        model_config.hidden_size=qwen_config.hidden_size
        model_config.intermediate_size=qwen_config.intermediate_size
        model_config.num_hidden_layers=qwen_config.num_hidden_layers
        model_config.num_attention_heads=qwen_config.num_attention_heads
        model_config.num_key_value_heads=qwen_config.num_key_value_heads
        model_config.hidden_act=qwen_config.hidden_act
        model_config.max_position_embeddings=qwen_config.max_position_embeddings
        model_config.initializer_range=qwen_config.initializer_range
        model_config.rms_norm_eps=qwen_config.rms_norm_eps
        model_config.use_cache=qwen_config.use_cache
        model_config.tie_word_embeddings=qwen_config.tie_word_embeddings
        model_config.rope_theta=qwen_config.rope_theta
        model_config.rope_scaling=qwen_config.rope_scaling
        model_config.use_sliding_window=qwen_config.use_sliding_window
        model_config.sliding_window=qwen_config.sliding_window
        model_config.max_window_layers=qwen_config.max_window_layers

    if os.path.isdir(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    Path(SAVE_DIR).mkdir(exist_ok=True)

    if model_config.name_or_path == None or model_config.name_or_path == '':
        model_config.name_or_path = SAVE_DIR

    if model_config.lc_connector_config == {}:
        model_config.set_linear_lc_connector()
    if model_config.parameter_connector_config == {}:
        model_config.set_linear_parameter_connector()
    if model_config.cfm_config == {}:
        model_config.set_cnn_cfm()
    if model_config.t_index_connector_config == {}:
        model_config.t_index_connector_config = model_config.parameter_connector_config

    tokenizer = AutoTokenizer.from_pretrained(LLM_DIR)
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<|sys_prompt|>', '<|lc_start|>', '<|nl_1|>', '<|nl_2|>', '<|r|>'] + [f"<|parameter_{i}|>" for i in range(model_config.num_parameters)] + ['<|t_index|>']
    })
    model_config.add_special_token('lc_end', tokenizer.convert_tokens_to_ids('<|sys_prompt|>'))
    model_config.add_special_token('lc_start', tokenizer.convert_tokens_to_ids('<|lc_start|>'))
    model_config.add_special_token('nl_1', tokenizer.convert_tokens_to_ids('<|nl_1|>'))
    model_config.add_special_token('nl_2', tokenizer.convert_tokens_to_ids('<|nl_2|>'))

    for i in range(model_config.num_parameters):
        model_config.add_parameter_token(f"parameter_{i}", tokenizer.convert_tokens_to_ids(f"<|parameter_{i}|>"))
    
    model_config.t_index_token = { "name": "t_index", "token_id": tokenizer.convert_tokens_to_ids("<|t_index|>") }

    _special_token_names = [tok['name'] for tok in model_config.special_token]
    _number_token = len(_special_token_names)
    _number_unique_token = len(set(_special_token_names))
    assert _number_token == _number_unique_token, f"Some special token appear multiple times in the model_config instead of once: special_tokens={_special_token_names}."

    _parameter_token_names = [tok['name'] for tok in model_config.parameter_token]
    _number_tokens = len(_parameter_token_names)
    _number_unique_tokens = len(set(_parameter_token_names))
    assert _number_tokens == _number_unique_tokens, f"Some parameter token appear multiple times in the model_config instead of once: special_tokens={_special_token_names}."

    model_config.use_cache = False

    model = L3MGenerator(model_config)
    
    if model_config.tie_word_embeddings:
        model.tie_weights()

    if not train_from_scratch and load_llm_weights:
        def _to_be_loaded(name:str, model_state_dict) -> bool:
            if name.replace('model.', '', 1) in model_state_dict:
                return True
            else:
                return False

        model_state_dict = model.state_dict()
        available_keys = list(model_state_dict.keys())

        num_comp_modules = 0

        archive_file = os.path.join(LLM_DIR, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.isfile(archive_file):
            with open(archive_file, "r") as f:
                index = json.loads(f.read())
            shard_filenames = sorted(set(index["weight_map"].values()))
            shard_filenames = [os.path.join(LLM_DIR, f) for f in shard_filenames]
        else:
            shard_filenames = [f"{LLM_DIR}/model.safetensors"]

        for shard_file in shard_filenames:
            state_dict = load_state_dict(shard_file, False)
            state_dict_to_load = {k.replace('model.', ''): v for k,v in state_dict.items() if _to_be_loaded(k, model_state_dict)}
            state_dict_not_to_load = [k for k in state_dict if not _to_be_loaded(k, model_state_dict)]

            _del_norm = False
            for _p in state_dict_to_load:
                if _p == "norm.weight":
                    _del_norm = True            
            if _del_norm:
                state_dict_to_load.pop("norm.weight")

            unexpected_keys = set(state_dict_to_load) - set(available_keys)
            assert len(unexpected_keys) == 0, f"There are some loaded layers that do not exist for the model: {unexpected_keys}"

            num_comp_modules += len(state_dict_to_load)

            if logger != None:
                if len(state_dict_not_to_load) > 0:
                    logger.info("WARNING: Some layers of the state dict were not loaded.\n" + "\n".join(state_dict_not_to_load))
                else:
                    logger.info("All LLM weights were loaded")

            model.load_state_dict(state_dict_to_load, strict=False)

            del state_dict, state_dict_to_load, state_dict_not_to_load

            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        assert num_comp_modules > 0, "No weights have been loaded"
        if logger != None:
            logger.info(f"{num_comp_modules} pretrained LLM weights have been loaded.")
    else:
        if logger != None:
            logger.info("The LLM has been initialized randomly")

    image_processor = L3MLightconeProcessorGeneration()

    processor = L3MProcessorGeneration(image_processor, tokenizer)
    processor.set_ids_from_config(model_config)
    processor.save_pretrained(save_directory=SAVE_DIR)

    model = model.to(device=device, dtype=dtype)

    model.save_pretrained(save_directory=SAVE_DIR, safe_serialization=False)
    if logger != None:
        logger.info(f"The model has been saved in {SAVE_DIR}")

    config_file = f"{SAVE_DIR}/config.json"
    f = open(config_file, 'r')
    config_data = json.loads(f.read())
    f.close()

    config_data['auto_map'] = {
        'AutoConfig': f'configuration.{L3MGenerationConfig.__name__}',
        'AutoModelForCausalLM': f'modeling.{L3MGenerator.__name__}'
    }
    f = open(config_file, 'w')
    f.write(json.dumps(config_data, indent=2))
    f.close()
    
    config_file = f"{SAVE_DIR}/preprocessor_config.json"
    f = open(config_file, 'r')
    config_data = json.loads(f.read())
    f.close()
    config_data['auto_map'] = {
        "AutoImageProcessor": f"processing.{L3MLightconeProcessorGeneration.__name__}",
        "AutoProcessor": f"processing.{L3MProcessorGeneration.__name__}"
    }
    f = open(config_file, 'w')
    f.write(json.dumps(config_data, indent=2))
    f.close()

    config_file = f"{SAVE_DIR}/processor_config.json"
    if os.path.exists(config_file):
        f = open(config_file, 'r')
        config_data = json.loads(f.read())
        f.close()
    else:
        config_data = {}
    config_data['auto_map'] = { "AutoProcessor": f"processing.{L3MProcessorGeneration.__name__}" }
    f = open(config_file, 'w')
    f.write(json.dumps(config_data, indent=2))
    f.close()

    base_model_dir = f"{os.environ['PYTHONPATH']}/src/experiments/generation/network"
    shutil.copy(f"{base_model_dir}/modeling.py", f"{SAVE_DIR}/modeling.py")
    shutil.copy(f"{base_model_dir}/configuration.py", f"{SAVE_DIR}/configuration.py")
    shutil.copy(f"{base_model_dir}/processing.py", f"{SAVE_DIR}/processing.py")

    return model, processor
