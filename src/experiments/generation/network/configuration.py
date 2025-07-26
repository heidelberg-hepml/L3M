from transformers import PretrainedConfig

from transformers.modeling_rope_utils import rope_config_validation

import math

class L3MGenerationConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=151936,
            hidden_size=4096,
            intermediate_size=22016,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=28,
            attention_dropout=0.0,
            lc_patch_size=[1,14,14],
            special_token = [],
            parameter_token = [],
            t_index_token = { 'name': 't_index', 'token_id': -1 },
            lc_connector_config={},
            parameter_connector_config={},
            t_index_connector_config={},
            cfm_config={},
            preprocessing:str=None,
            num_parameters=0,
            from_scratch=False,
            **kwargs
    ):
        # LLM Config
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # Special tokens
        self.special_token = special_token

        # Connector config
        self.parameter_token = parameter_token
        self.t_index_token = t_index_token

        self.preprocessing = preprocessing

        self.lc_patch_size = lc_patch_size
        self.lc_patch_dim = math.prod(lc_patch_size)
        
        self.lc_connector_config = lc_connector_config
        self.parameter_connector_config = parameter_connector_config
        self.t_index_connector_config = t_index_connector_config
        self.cfm_config = cfm_config

        self.num_parameters = num_parameters

        self.from_scratch = from_scratch

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

    def set_linear_lc_connector(self):
        self.lc_connector_config = {
            'cls': 'linear'
        }

    def set_linear_parameter_connector(self):
        self.parameter_connector_config = {
            'cls': 'linear'
        }

    def set_linear_t_index_connector(self):
        self.t_index_connector_config = {
            'cls': 'linear'
        }

    def set_cnn_cfm(self):
        self.cfm_config = {
            'cls': 'cnn'
        }

    def add_special_token(self, token_name, token_id):
        if len([1 for tok in self.special_token if tok['name'] == token_name]) == 0:
            self.special_token.append(
                {
                    'name': token_name,
                    'token_id': token_id
                }
            )

    def add_parameter_token(self, token_name, token_id):
        if len([1 for tok in self.parameter_token if tok['name'] == token_name]) == 0:
            self.parameter_token.append(
                {
                    'name': token_name,
                    'token_id': token_id
                }
            )

    @classmethod
    def for_scratch_model(cls, **kwargs):
        default_config = {
            "vocab_size": 1,
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 3072,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "use_cache": True,
            "tie_word_embeddings": False,
            "rope_theta": 10000.0,
            "rope_scaling": None,
            "use_sliding_window": False,
            "sliding_window": 3072,
            "max_window_layers": 3,
            "attention_dropout": 0.0,
        }

        for k, v in default_config.items():
            if k not in kwargs:
                kwargs[k] = v

        return L3MGenerationConfig(            
            **kwargs
        )
