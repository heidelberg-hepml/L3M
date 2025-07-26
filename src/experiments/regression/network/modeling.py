import inspect
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.experiments.regression.dataset_utils import CausalLMOutputWithPastAndSigmas

from transformers.generation.utils import GenerationMixin

from transformers.utils import logging

from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

try:
    from flash_attn import flash_attn_func

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
except ImportError:
    pass

try:
    import src.utils.flash_attention_3 as flash_attention_3
    SUPPORTS_FA3 = True
except ImportError:
    SUPPORTS_FA3 = False

import torch
from torch import nn

from transformers.utils import logging

from .configuration import L3MRegressorConfig
from src.networks.base_modules import TokenEmbd
from src.experiments.regression.network.embedding import L3MLcEmbedding

from transformers import Qwen2Model, Qwen2VLModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    QWEN2_VL_ATTENTION_CLASSES,
    Qwen2VLAttention,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

import math

logger = logging.get_logger(__name__)

MAX_INPUT_ID = int(1e9)

class L3MParamHead(nn.Module):
    def __init__(self, config: L3MRegressorConfig, **kwargs):
        super().__init__()

        self.param_dim = config.param_dim
        self.output_dim = config.param_dim + (config.param_dim + 1) * config.param_dim // 2
        self._index_diag = torch.arange(0, config.param_dim)
        self._index_off_diag_0, self._index_off_diag_1 = torch.tril_indices(config.param_dim, config.param_dim, -1)

        self.register_buffer("sigma_scale", (1. / torch.sqrt(0.25 * torch.arange(1, self.param_dim + 1, dtype=torch.float32))).unsqueeze(-1))
        
        param_head_cls = config.param_head_config.get('cls', 'linear')
        if param_head_cls == 'linear':
            self.param_head = nn.Linear(config.hidden_size, self.output_dim)
        elif param_head_cls == 'mlp':
            mlp_config = config.param_head_config.get('mlp_config', {'depth: 2'})
            int_dim = mlp_config.get('interm_dim', 64)
            depth = mlp_config['depth']
            layers = []
            layers.extend([nn.Linear(config.hidden_size, int_dim), nn.GELU()])
            for _ in range(depth):
                layers.extend([nn.Linear(int_dim, int_dim), nn.GELU()])
            layers.append(nn.Linear(int_dim, self.output_dim))
            self.param_head = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'Lightcone head class {param_head_cls} is not implemented.')
        self.param_head_out_act = F.sigmoid

    def forward(self, x:torch.FloatTensor):
        x = self.param_head(x)
        params = x[...,:self.param_dim].clone()
        params = self.param_head_out_act(params)

        sigmas_diag = x[...,self.param_dim:2*self.param_dim].clone()
        sigmas_off_diag = x[...,2*self.param_dim:].clone()

        sigmas_diag = torch.nn.functional.softplus(sigmas_diag)

        _sigmas = torch.zeros(*params.shape[:-1], self.param_dim, self.param_dim, dtype=sigmas_diag.dtype, device=sigmas_diag.device)
        _sigmas[..., self._index_diag, self._index_diag] = sigmas_diag
        _sigmas[..., self._index_off_diag_0, self._index_off_diag_1] = sigmas_off_diag
        _sigmas = _sigmas * self.sigma_scale

        _diag = torch.diagonal(_sigmas, 0, -2, -1)

        # This can be used as a regulartor for the initial training
        # Once all uncertainties are estimated well, we have _diag > 1 > 20 * 10^{-4} = 0.002. There, the softmax becomes the identity.
        _logdet = torch.nn.functional.softplus(torch.prod(_diag, dim=-1), beta=1.0e4)
        _logdet = torch.log(_logdet)

        sigmas = torch.matmul(_sigmas, _sigmas.transpose(-2, -1))
        return params, sigmas, _logdet
            
class L3MRegressor(Qwen2Model, GenerationMixin):
    config_class = L3MRegressorConfig

    def __init__(self, config: L3MRegressorConfig, **kwargs):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        
        self.lc_embedding: L3MLcEmbedding = L3MLcEmbedding(config, wte=self.embed_tokens, **kwargs)
        self.param_head: L3MParamHead = L3MParamHead(config, **kwargs)
        self.gradient_checkpointing = False

        self.special_token_embds = nn.ModuleDict(
            { tok['name']: TokenEmbd([1, config.hidden_size]) for tok in config.special_token }
        )
        self.special_token_id = { tok['name']: tok['token_id'] for tok in config.special_token }
        self.special_token_pos_args = { tok['name']: [tok['name'] + '_pos_0', tok['name'] + '_pos_1' ] for tok in config.special_token }

        self.lc_embedding_dtype = self.dtype
        self.base_dtype = self.dtype
        self.param_head_dtype = self.dtype

        self.register_buffer("param_dim_pt", torch.tensor(float(self.param_head.param_dim)))
        
        self.post_init()

    def set_multi_dtype(
            self,
            lc_embedding_dtype=torch.float32,
            base_dtype=torch.bfloat16,
            param_head_dtype=torch.float32):
        self.lc_embedding_dtype = lc_embedding_dtype
        self.base_dtype = base_dtype
        self.param_head_dtype = param_head_dtype

        self.embed_tokens.to(base_dtype)
        self.special_token_embds.to(base_dtype)
        self.layers.to(base_dtype)
        self.rotary_emb.to(base_dtype)

        self.lc_embedding.to(lc_embedding_dtype)
        self.norm.to(param_head_dtype)
        self.param_head.to(param_head_dtype)
        self.param_dim_pt.to(param_head_dtype)

    # Modified from transformers.Qwen2Model
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lc_positions_0: Optional[torch.LongTensor] = None,
        lc_positions_1: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        lc_values: Optional[Union[torch.FloatTensor, torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        detach_for_head: bool = False,
        **token_args
    ) -> Union[Tuple, CausalLMOutputWithPastAndSigmas]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids
        if input_ids == None:
            raise ValueError("You have to specify input_ids")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if lc_values is not None:
            lc_values = lc_values.to(self.lc_embedding_dtype)
            inputs_embeds = self.lc_embedding(input_ids, lc_values=lc_values, positions=(lc_positions_0, lc_positions_1))
        else:
            inputs_embeds = self.embed_tokens(input_ids)
            
        inputs_embeds = inputs_embeds.to(self.base_dtype)

        for special_token in self.special_token_id:
            if token_args != None and self.special_token_pos_args[special_token][0] in token_args: 
                special_token_pos_0 = token_args[self.special_token_pos_args[special_token][0]]
                special_token_pos_1 = token_args[self.special_token_pos_args[special_token][1]]
            else:
                special_token_mask = (input_ids == self.special_token_id[special_token])
                special_token_pos_0, special_token_pos_1= torch.nonzero(special_token_mask, as_tuple=True)

            if special_token_pos_0.size(dim=0) > 0:
                vals = self.special_token_embds[special_token](None).expand(special_token_pos_0.size(dim=0), -1)
                inputs_embeds = inputs_embeds.index_put(
                    (special_token_pos_0, special_token_pos_1), vals, accumulate=False
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i_layer, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if detach_for_head and i_layer == self.config.detach_for_head_layer:
                hidden_states = hidden_states.detach()

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        param_token_states = hidden_states[token_args["ska_param_pos_0"], token_args["ska_param_pos_1"]].reshape(-1, self.config.hidden_size)
        param_token_states = param_token_states.to(self.param_head_dtype)
        param_token_states = self.norm(param_token_states)

        if output_hidden_states:
            all_hidden_states += (param_token_states,)

        _params = self.param_head(param_token_states)
        params, sigmas, _logdet = _params
        
        loss = None
        if labels != None:
            _diff = params - labels
            _diff2 = torch.matmul(sigmas, _diff.unsqueeze(-1)).squeeze(-1)

            loss = (0.5 * torch.sum(_diff * _diff2, dim=-1) - _logdet) / self.param_dim_pt
            loss = torch.mean(loss)
        
        if not return_dict:
            output = (params,) + all_hidden_states if output_hidden_states else (params, param_token_states)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndSigmas(
            loss=loss,
            logits=params,
            sigmas=sigmas,
            ln_det_sigmas=_logdet,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states if output_hidden_states else (param_token_states,),
            attentions=all_self_attns
        )

    # Modified from transformers.Qwen2Model
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation in ["flash_attention_2", "flash_attention_3"]:
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
