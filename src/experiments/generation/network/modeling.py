import inspect
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.generation.utils import (
    GenerateOutput,
    GenerationMixin
)
from transformers.utils import (
    is_torchdynamo_compiling,
    logging,
)

try:
    from flash_attn import flash_attn_func

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
except ImportError:
    pass

import torch
from torch import nn

from transformers.utils import logging

from .configuration import L3MGenerationConfig
from src.networks.base_modules import TokenEmbd, ResidualStackMLP, ResidualStackCNN
from src.experiments.generation.network.embedding import L3MLcEmbedding, L3MParameterEmbedding, L3MTIndxEmbedding

from transformers import Qwen2Model

import math

from torchdiffeq import odeint

logger = logging.get_logger(__name__)

class L3M_CFM_Head(nn.Module):
    def __init__(self, config: L3MGenerationConfig, **kwargs):
        super().__init__()

        self.lc_patch_dim = config.lc_patch_dim
        
        self.lc_head_cls = config.cfm_config.get('cls', 'mlp')
        if self.lc_head_cls == 'mlp':
            logger.info('Creating the lc head with an MLP')
            mlp_config = config.cfm_config.get('mlp_config', {'depth': 2, 'hidden_dim': -1})

            self.total_input_dim = config.hidden_size + self.lc_patch_dim + 1
            self.hidden_dim = mlp_config["hidden_dim"] if mlp_config["hidden_dim"] != -1 else self.total_input_dim
            self.output_dim = self.lc_patch_dim

            depth = mlp_config['depth']

            self.sub_lc_head = nn.Sequential(
                nn.Linear(self.total_input_dim, self.hidden_dim),
                ResidualStackMLP(self.hidden_dim, 2 * self.hidden_dim, depth),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        elif self.lc_head_cls == 'cnn':
            logger.info('Creating the lc head with a CNN')
            cnn_config = config.cfm_config.get('cnn_config', {'depth': 2, 'chs': 64, 'cond_dim': 784, 'kernel_size': 3})
            cond_dim = cnn_config.get("cond_dim", 784)
            
            assert cond_dim % config.lc_patch_dim == 0
            self.init_cond_chs = cond_dim // config.lc_patch_dim

            cnn_chs = cnn_config.get('chs', 64)
            cnn_depth = cnn_config.get('depth', 2)
            kernel_size = cnn_config.get('kernel_size', 3)
            cnn_patch_size = cnn_config.get('cnn_patch_size', None)
            residual_chs_factor = cnn_config.get('residual_chs_factor', 2)

            self.cnn_patch_size = cnn_patch_size

            self.cond_layer = nn.Linear(config.hidden_size, cond_dim)

            padding = (kernel_size - 1) // 2

            self.patch_size = config.lc_patch_size[1:]
            
            self.cnn_layer = nn.Sequential(
                    nn.Conv2d(2 + self.init_cond_chs, cnn_chs, kernel_size=kernel_size, stride=1, padding=padding),
                    ResidualStackCNN(cnn_chs, residual_chs_factor * cnn_chs, cnn_depth, kernel_size=kernel_size, padding=padding),
                    nn.Conv2d(cnn_chs, 1, kernel_size=kernel_size, stride=1, padding=padding),
                ).to(memory_format=torch.channels_last)
        else:
            raise NotImplementedError(f'Lightcone head class {self.lc_head_cls} is not implemented.')

        self.loss_fn = torch.nn.MSELoss(reduction="mean")
            
    def forward(self,
            x:torch.FloatTensor=None,
            t:torch.FloatTensor=None,
            pixels:torch.FloatTensor=None,
            logits:torch.FloatTensor=None,
            cond_info:torch.FloatTensor=None
        ) -> torch.FloatTensor:

        if self.lc_head_cls == 'mlp':            
            v = self.sub_lc_head(x)
            return v[:, :self.output_dim]
        elif self.lc_head_cls == 'cnn':

            if cond_info == None and logits != None:
                cond_info = self.cond_layer(logits).view(-1, self.init_cond_chs, *self.patch_size)
            t = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.patch_size)
            pixel = pixels.reshape(-1, 1, *self.patch_size)
            if cond_info != None:
                x = torch.cat([cond_info, t, pixel], dim=1).contiguous(memory_format=torch.channels_last)
            else:
                x = torch.cat([t, pixel], dim=1).contiguous(memory_format=torch.channels_last)
            v = self.cnn_layer(x).reshape(-1, self.lc_patch_dim).contiguous()
            return v
        else:
            raise NotImplementedError()
    
    def get_loss(self, logits, labels):
        if logits != None:
            logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])

        noise = torch.randn_like(labels)
        t = torch.rand(labels.shape[0], 1, device=labels.device)
        labels_t = (1 - t) * noise + t * labels
        v = labels - noise
        if self.lc_head_cls == 'mlp':
            v_pred = self(x=torch.cat([t, labels_t, logits], dim=-1))
        elif self.lc_head_cls == 'cnn':
            v_pred = self(t=t, pixels=labels_t, logits=logits)
        else:
            raise NotImplementedError()
        loss = self.loss_fn(v_pred, v)
        
        return loss
    
    @torch.no_grad
    def sample(self, logits, n_samples=1):

        if logits != None:
            batch_size = logits.shape[0]
            dtype = logits.dtype
            device = logits.device
        else:
            batch_size = 1
            dtype = torch.float32
            device = 'cuda'
        
        noise = torch.randn(batch_size * n_samples, self.lc_patch_dim, dtype=dtype, device=device)

        if self.lc_head_cls == "cnn":
            if logits != None:
                cond_info = self.cond_layer(logits).reshape(batch_size, -1, *self.patch_size)
                cond_info = cond_info.unsqueeze(0).expand(n_samples, -1, -1, -1, -1).reshape(n_samples * batch_size, -1, *self.patch_size)
            else:
                cond_info = None

        def ode_wrapper(t, y_t):
            if self.lc_head_cls == 'mlp':
                t = torch.full((batch_size, 1), t, dtype=y_t.dtype, device=y_t.device)
                v = self(x=torch.cat([t, y_t, logits], dim=-1))
            elif self.lc_head_cls == 'cnn':
                t = torch.full((batch_size, 1), t, dtype=y_t.dtype, device=y_t.device)
                v = self(t=t, pixels=y_t, cond_info=cond_info)
            else:
                raise NotImplementedError()
            
            return v
 
        y_t = odeint(
            func=ode_wrapper, 
            y0=noise,
            t=torch.tensor([0, 1], device=device , dtype=dtype),
            rtol=1e-3,
            atol=1e-3,
            method='dopri5'
        )

        bt_vals = y_t[-1]

        if n_samples > 1:
            bt_vals = bt_vals.reshape(n_samples, batch_size, -1, *self.patch_size)

        return bt_vals

class L3MGenerator(Qwen2Model, GenerationMixin):
    config_class = L3MGenerationConfig

    def __init__(self, config: L3MGenerationConfig, **kwargs):
        super().__init__(config)

        self.vocab_size = config.vocab_size

        self.lc_embedding: L3MLcEmbedding = L3MLcEmbedding(config, wte=self.embed_tokens, **kwargs)
        self.parameter_embedding: L3MParameterEmbedding = L3MParameterEmbedding(config, **kwargs)
        self.t_index_embedding: L3MTIndxEmbedding = L3MTIndxEmbedding(config, **kwargs)
        self.lc_head: L3M_CFM_Head = L3M_CFM_Head(config, **kwargs)
        
        self.gradient_checkpointing = False

        self.special_token_embds = nn.ModuleDict(
            { tok['name']: TokenEmbd([1, config.hidden_size]) for tok in config.special_token }
        )
        self.special_token_id = { tok['name']: tok['token_id'] for tok in config.special_token }
        self.special_token_pos_args = { tok['name']: [tok['name'] + '_pos_0', tok['name'] + '_pos_1' ] for tok in config.special_token }

        self.parameter_token_id = { tok['name']: tok['token_id'] for tok in config.parameter_token }
        self.t_index_token_id = config.t_index_token['token_id']

        self.lc_embedding_dtype = self.dtype
        self.parameter_embedding_dtype = self.dtype
        self.base_dtype = self.dtype
        self.lc_head_dtype = self.dtype

        self.lc_patch_dim = self.lc_embedding.lc_patch_dim
        self.parameter_token_id = self.parameter_embedding.parameter_token_id
        self.parameter_token_val_args = self.parameter_embedding.parameter_token_val_args

        self.post_init()

    def set_multi_dtype(
            self,
            lc_embedding_dtype=torch.float32,
            parameter_embedding_dtype=torch.float32,
            base_dtype=torch.bfloat16,
            lc_head_dtype=torch.float32):
        self.lc_embedding_dtype = lc_embedding_dtype
        self.parameter_embedding_dtype = parameter_embedding_dtype
        self.base_dtype = base_dtype
        self.lc_head_dtype = lc_head_dtype

        self.embed_tokens.to(base_dtype)
        self.special_token_embds.to(base_dtype)
        self.layers.to(base_dtype)
        self.rotary_emb.to(base_dtype)

        self.lc_embedding.to(lc_embedding_dtype)
        self.parameter_embedding.to(parameter_embedding_dtype)
        self.t_index_embedding.to(parameter_embedding_dtype)
        self.norm.to(lc_head_dtype)
        self.lc_head.to(lc_head_dtype)

    # Modified from transformers.Qwen2Model
    def forward(
        self,
        output_lc_patch_mask: Optional[torch.BoolTensor] = None,
        lc_patch_pos_0: Optional[torch.LongTensor] = None,
        lc_patch_pos_1: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        lc_positions_0: Optional[torch.LongTensor] = None,
        lc_positions_1: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        lc_values: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        t_indices: Optional[torch.FloatTensor] = None,
        t_index_pos_0: Optional[torch.LongTensor] = None,
        t_index_pos_1: Optional[torch.LongTensor] = None,
        **token_args
    ) -> Union[Tuple, CausalLMOutputWithPast]:

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
            lc_values = lc_values.to(dtype=self.lc_embedding_dtype)
            with torch.autocast(enabled=False, device_type='cuda'):
                inputs_embeds = self.lc_embedding(input_ids, lc_values=lc_values, positions=(lc_positions_0, lc_positions_1))
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        with torch.autocast(enabled=False, device_type='cuda'):
            inputs_embeds = self.parameter_embedding(inputs_embeds, input_ids, **token_args)
        with torch.autocast(enabled=False, device_type='cuda'):
            inputs_embeds = self.t_index_embedding(inputs_embeds, input_ids, t_indices, t_index_pos_0, t_index_pos_1)
        
        if self.lc_embedding_dtype != self.base_dtype:
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

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if lc_patch_pos_0 != None:
            lc_patch_states = hidden_states[lc_patch_pos_0, lc_patch_pos_1].reshape(-1, self.config.hidden_size)
        else:
            lc_patch_states = hidden_states[output_lc_patch_mask].reshape(-1, self.config.hidden_size)
        if self.base_dtype != self.lc_head_dtype:
            lc_patch_states = lc_patch_states.to(self.lc_head_dtype)
        lc_patch_states = self.norm(lc_patch_states)

        if output_hidden_states:
            all_hidden_states += (lc_patch_states,)

        with torch.autocast(enabled=False, device_type='cuda'):

            if labels != None:
                loss = self.lc_head.get_loss(lc_patch_states, labels)
            else:
                loss = None
        
        if not return_dict:
            output = (lc_patch_states,) + all_hidden_states if output_hidden_states else (lc_patch_states)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lc_patch_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states if output_hidden_states else (lc_patch_states,),
            attentions=all_self_attns
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        lc_patch_pos_0: torch.LongTensor,
        lc_patch_pos_1: torch.LongTensor,
        lcs_init: torch.FloatTensor,
        t_init: torch.FloatTensor,
        t_max: torch.FloatTensor,
        lc_shape: torch.LongTensor,
        t_indices: torch.FloatTensor,
        lc_id: torch.LongTensor,
        gen_token_pos: list[dict] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        config: L3MGenerationConfig = self.config

        batch_size = input_ids.size(0)
        patch_dim = config.lc_patch_dim

        _lc_shape = lc_shape.cpu().tolist()
        _patch_shape = config.lc_patch_size
        _t_init = t_init[0].item()
        _t_max = t_max[0].item()

        t_size = _t_max-_t_init
        sp_size = _lc_shape[1] * _lc_shape[2]

        lcs_init = lcs_init.reshape(batch_size, -1, patch_dim)
        lcs_pred = torch.zeros(batch_size, t_size, 1, *_lc_shape[1:3], *_patch_shape, dtype=lcs_init.dtype, device=lcs_init.device).reshape(batch_size, -1, patch_dim)

        lc_patch_pos_0 = lc_patch_pos_0.reshape(batch_size, -1)
        lc_patch_pos_1 = lc_patch_pos_1.reshape(batch_size, -1)

        seq_ids = lc_patch_pos_1[0,:].cpu().tolist()

        last_seq_pos = 0
        _past_key_values = None
        i_step = 0

        def _get_step_model_inputs(i_step, seq_pos, last_seq_pos, _past_key_values, last_i_step=None, skip_additional_model_input=False):
            if last_i_step == None:
                last_i_step = i_step - 1
            
            model_inputs = {
                    "input_ids": input_ids[:,last_seq_pos:seq_pos],
                    "lc_patch_pos_0": lc_patch_pos_0[:,i_step],
                    "lc_patch_pos_1": lc_patch_pos_1[:,i_step] - last_seq_pos,
                    "past_key_values": _past_key_values
                }
            
            if skip_additional_model_input:
                additional_model_input = {}
            else:   
                additional_model_input = gen_token_pos[i_step] if gen_token_pos != None else {}
            
            if i_step == 0:
                model_inputs["lc_values"] = lcs_init.reshape(-1, patch_dim)
                model_inputs["t_indices"] = t_indices
                model_inputs["t_index_pos_0"] = kwargs.get("t_index_pos_0", None)
                model_inputs["t_index_pos_1"] = kwargs.get("t_index_pos_1", None)
                for i, tok in enumerate(self.parameter_token_id):
                    model_inputs[tok + '_vals'] = kwargs[tok + '_vals']
                    model_inputs[tok + '_pos_0'] = kwargs.get(tok + '_pos_0', None)
                    model_inputs[tok + '_pos_1'] = kwargs.get(tok + '_pos_1', None)
            else:
                model_inputs["lc_values"] = lcs_pred[:,last_i_step:i_step].reshape(-1, patch_dim)
            
            return model_inputs, additional_model_input

        for t_step in range(t_size):
            for sp_step in range(sp_size):
                seq_pos = seq_ids[i_step] + 1

                model_inputs, additional_model_input = _get_step_model_inputs(i_step, seq_pos, last_seq_pos, _past_key_values)
            
                last_seq_pos = seq_pos
        
                output = self(
                    **model_inputs,
                    **additional_model_input,
                    use_cache=True,
                    return_dict=True
                )

                _past_key_values = output.past_key_values

                bt_vals = self.lc_head.sample(output.logits)
                lcs_pred[:,i_step] = bt_vals
                
                i_step += 1
        
        lcs_init = lcs_init.reshape(batch_size, _t_init, *_lc_shape[1:3], *_patch_shape).permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch_size,
            _t_init*_patch_shape[0],
            _lc_shape[1]*_patch_shape[1],
            _lc_shape[2]*_patch_shape[2],
        )
        lcs_pred = lcs_pred.reshape(batch_size, t_size, 1, *_lc_shape[1:3], *_patch_shape).permute(0, 1, 5, 3, 6, 4, 7, 2).reshape(
            batch_size,
            t_size*_patch_shape[0],
            _lc_shape[1]*_patch_shape[1],
            _lc_shape[2]*_patch_shape[2]
        )

        return {
            "ids": lc_id,
            "t_indices": t_indices,
            "lcs_init": lcs_init,
            "lcs_pred": lcs_pred
        }
