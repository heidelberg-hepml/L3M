
from typing import Tuple, Optional

from transformers import AutoTokenizer, AutoConfig, Qwen2Config, AutoProcessor
from src.experiments.regression.network.modeling import L3MRegressor
from src.experiments.regression.network.configuration import L3MRegressorConfig
from src.experiments.regression.network.processing import L3MProcessorRegression, L3MLightconeProcessorRegression
import shutil


import os, json, shutil
from transformers.utils.__init__ import SAFE_WEIGHTS_INDEX_NAME
from transformers.modeling_utils import load_state_dict
import gc
import torch

from pathlib import Path

@torch.no_grad()
def construct_model_config(
    LLM_DIR:str,
    SAVE_DIR:str,
    model_config: Optional[L3MRegressorConfig] = None,
    load_llm_weights:bool=True,
    train_from_scratch:bool=False,
    custom_number_of_layers: Optional[int]=None,
    logger=None,
)-> L3MRegressorConfig:

    if model_config == None:
        model_config = L3MRegressorConfig()

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
        
        if custom_number_of_layers != None:
            assert not load_llm_weights, "custom_number_of_layers and load_llm_weights are exclusive."
            model_config.num_hidden_layers=custom_number_of_layers
            if logger != None:
                logger.info(f"Setting number of hidden layers to {custom_number_of_layers}")

    if os.path.isdir(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)

    if model_config.name_or_path == None or model_config.name_or_path == '':
        model_config.name_or_path = SAVE_DIR

    if model_config.lc_connector_config == {}:
        model_config.set_linear_lc_connector()
    if model_config.param_head_config == {}:
        model_config.set_linear_param_head()

    tokenizer = AutoTokenizer.from_pretrained(LLM_DIR)

    special_tokens = ['<|ska_param|>']
    special_tokens += [f"<|sys_prompt_{i}|>" for i in range(model_config.num_sys_prompt_tokens)]
    special_tokens += [f"<|pre_lc_{i}|>" for i in range(model_config.num_pre_lc_tokens)]

    tokenizer.add_special_tokens({
        'additional_special_tokens': special_tokens
    })

    model_config.add_special_token('ska_param', tokenizer.convert_tokens_to_ids('<|ska_param|>'))
    for i in range(model_config.num_sys_prompt_tokens):
        model_config.add_special_token(f"sys_prompt_{i}", tokenizer.convert_tokens_to_ids(f"<|sys_prompt_{i}|>"))
    for i in range(model_config.num_pre_lc_tokens):
        model_config.add_special_token(f"pre_lc_{i}", tokenizer.convert_tokens_to_ids(f"<|pre_lc_{i}|>"))
    
    _special_token_names = [tok['name'] for tok in model_config.special_token]
    _number_token = len(_special_token_names)
    _number_unique_token = len(set(_special_token_names))
    assert _number_token == _number_unique_token, f"Some special token appear multiple times in the model_config instead of once: special_tokens={_special_token_names}."

    model_config.use_cache = False

    model_config.save_pretrained(SAVE_DIR)

    config_file = f"{SAVE_DIR}/config.json"
    f = open(config_file, 'r')
    config_data = json.loads(f.read())
    f.close()
    config_data['auto_map'] = {
        'AutoConfig': f'configuration.{L3MRegressorConfig.__name__}',
        'AutoModelForCausalLM': f'modeling.{L3MRegressor.__name__}'
    }
    f = open(config_file, 'w')
    f.write(json.dumps(config_data, indent=2))
    f.close()

    image_processor = L3MLightconeProcessorRegression()

    processor: L3MProcessorRegression = L3MProcessorRegression(image_processor, tokenizer)
    processor.save_pretrained(save_directory=SAVE_DIR)

    config_file = f"{SAVE_DIR}/preprocessor_config.json"
    f = open(config_file, 'r')
    config_data = json.loads(f.read())
    f.close()
    config_data['auto_map'] = {
        "AutoImageProcessor": f"processing.{L3MLightconeProcessorRegression.__name__}",
        "AutoProcessor": f"processing.{L3MProcessorRegression.__name__}"
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
    config_data['auto_map'] = { "AutoProcessor": f"processing.{L3MProcessorRegression.__name__}" }
    f = open(config_file, 'w')
    f.write(json.dumps(config_data, indent=2))
    f.close()

    base_model_dir = f"{os.environ['PYTHONPATH']}/src/experiments/regression/network"
    shutil.copy(f"{base_model_dir}/modeling.py", f"{SAVE_DIR}/modeling.py")
    shutil.copy(f"{base_model_dir}/configuration.py", f"{SAVE_DIR}/configuration.py")
    shutil.copy(f"{base_model_dir}/processing.py", f"{SAVE_DIR}/processing.py")

    return model_config

@torch.no_grad()
def init_l3m_regressor(
        LLM_DIR:str,
        MODEL_CONFIG_DIR:str,
        SAVE_DIR:str,
        load_llm_weights:bool=True,
        train_from_scratch:bool=False,
        dtype=torch.float32,
        device="cuda",
        _attn_implementation="sdpa",
        logger=None
    ) -> Tuple[L3MRegressor, L3MProcessorRegression]:

    Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

    shutil.copy(f"{MODEL_CONFIG_DIR}/modeling.py", f"{SAVE_DIR}/modeling.py")
    shutil.copy(f"{MODEL_CONFIG_DIR}/configuration.py", f"{SAVE_DIR}/configuration.py")
    shutil.copy(f"{MODEL_CONFIG_DIR}/processing.py", f"{SAVE_DIR}/processing.py")

    model_config = AutoConfig.from_pretrained(
        MODEL_CONFIG_DIR,
        trust_remote_code=True,
        local_files_only=True,
        _attn_implementation=_attn_implementation
    )
    
    processor = AutoProcessor.from_pretrained(
        MODEL_CONFIG_DIR,
        trust_remote_code=True,
        local_files_only=True
    )

    model = L3MRegressor(model_config)
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

        embd_tokens_loaded = False
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
            state_dict_to_load = {k.replace('model.', '', 1): v for k,v in state_dict.items() if _to_be_loaded(k, model_state_dict)}
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
            for _p in state_dict_to_load:
                if "embed_tokens.weight" in _p:
                    embd_tokens_loaded = True

            if logger != None:
                if len(state_dict_not_to_load) > 0:
                    logger.info("WARNING: Some LLM weights were not loaded.\n" + "\n".join(state_dict_not_to_load))
                else:
                    logger.info("All LLM weights were loaded")

            model.load_state_dict(state_dict_to_load, strict=False)

            del state_dict, state_dict_to_load, state_dict_not_to_load

            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # TODO Check this
        model.norm.weight.copy_(torch.ones(*model.norm.weight.shape, dtype=model.norm.weight.dtype, device=model.norm.weight.device))

        assert num_comp_modules > 0, "No weights have been loaded"
        if logger != None:
            logger.info(f"{num_comp_modules} pretrained LLM weights have been loaded.")
    else:
        if logger != None:
            logger.info("The LLM has been initialized randomly")

    model = model.to(device=device, dtype=dtype)

    model.save_pretrained(save_directory=SAVE_DIR, safe_serialization=False) # safe_serialization=False)

    processor.save_pretrained(save_directory=SAVE_DIR)

    config_file = f"{SAVE_DIR}/config.json"
    f = open(config_file, 'r')
    config_data = json.loads(f.read())
    f.close()
    config_data['auto_map'] = {
        'AutoConfig': f'configuration.{L3MRegressorConfig.__name__}',
        'AutoModelForCausalLM': f'modeling.{L3MRegressor.__name__}'
    }
    f = open(config_file, 'w')
    f.write(json.dumps(config_data, indent=2))
    f.close()

    return model, processor
