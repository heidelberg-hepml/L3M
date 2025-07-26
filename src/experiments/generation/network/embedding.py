import torch
from torch import nn
from src.experiments.generation.network.configuration import L3MGenerationConfig

from typing import List, Optional, Tuple, Union

from src.networks.base_modules import TokenEmbd

class L3MLcEmbedding(nn.Module):
    def __init__(self, config: L3MGenerationConfig, wte=None, **kwargs) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.wte = wte
    
        self.lc_patch_dim = config.lc_patch_dim
        self.hidden_size = config.hidden_size
        
        lc_connector_cls = config.lc_connector_config.get('cls', 'linear')
        if lc_connector_cls == 'linear':
            self.lc_connector = nn.Linear(self.lc_patch_dim, self.hidden_size)
        elif lc_connector_cls == 'mlp':
            mlp_config = config.lc_connector_config.get('mlp_config', {'depth': 2})
            dim_projection = self.hidden_size
            int_dim = mlp_config.get('interm_dim', 896)
            depth = mlp_config['depth']
            layers = [nn.Linear(self.lc_patch_dim, int_dim)]
            for _ in range(depth):
                layers.extend([nn.GELU(), nn.Linear(int_dim, int_dim)])
            layers.extend([nn.GELU(), nn.Linear(int_dim, dim_projection)])
            self.lc_connector = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'Lightcone connector class {lc_connector_cls} is not implemented.')

        self.vocab_size = config.vocab_size

        self.model_from_scratch = config.from_scratch

    def forward(
        self,
        input_ids: torch.LongTensor,
        lc_values: torch.FloatTensor,
        positions: Optional[Tuple[torch.LongTensor, torch.LongTensor]] = None,
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        if positions == None or positions[0] == None:
            positions = torch.nonzero((input_ids < 0), as_tuple=True)

        has_image = positions[0].size(dim=0) > 0
        input_ids = input_ids.clamp_min(0).clamp_max(self.vocab_size).detach()
        if not self.model_from_scratch:
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = torch.zeros(*input_ids.shape, self.hidden_size, dtype=self.wte.weight.dtype, device=input_ids.device)

        if has_image:
            lc_features: torch.FloatTensor = self.lc_connector(lc_values)
            hidden_states = hidden_states.index_put(
                positions, lc_features.to(hidden_states.dtype), accumulate=False
            )
        return hidden_states

class L3MParameterEmbedding(nn.Module):
    def __init__(self, config: L3MGenerationConfig, **kwargs) -> None:
        super().__init__()

        parameter_connector_cls = config.parameter_connector_config.get('cls', 'linear')
        if parameter_connector_cls == 'linear':
            self.parameter_token_connector = nn.ModuleDict(
            { tok['name']: nn.Linear(1, config.hidden_size) for tok in config.parameter_token }) 
        elif parameter_connector_cls == 'mlp':
            mlp_config = config.parameter_connector_config.get('mlp_config', {'depth': 2})
            dim_projection = config.hidden_size
            _parameter_token_connector = {}
            for tok in config.parameter_token:
                depth = mlp_config['depth']
                int_dim = mlp_config.get('interm_dim', 64)
                layers = [nn.Linear(1, int_dim)]
                for _ in range(depth):
                    layers.extend([nn.GELU(), nn.Linear(int_dim, int_dim)])
                layers.extend([nn.GELU(), nn.Linear(int_dim, dim_projection)])
                _parameter_token_connector[tok['name']] = nn.Sequential(*layers)
                self.parameter_token_connector = nn.ModuleDict(_parameter_token_connector)
        else:
            raise NotImplementedError(f'Parameter connector class {parameter_connector_cls} is not implemented.')

        self.parameter_token_id = { tok['name']: tok['token_id'] for tok in config.parameter_token }
        self.parameter_token_pos_args = { tok['name']: [tok['name'] + '_pos_0', tok['name'] + '_pos_1' ] for tok in config.parameter_token }
        self.parameter_token_val_args = { tok['name']: tok['name'] + '_vals' for tok in config.parameter_token }

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        **parameter_token_args
    ) -> torch.FloatTensor:
        if parameter_token_args == None:
            return inputs_embeds
        
        for parameter_token in self.parameter_token_id:

            if self.parameter_token_val_args[parameter_token] in parameter_token_args:
                parameter_token_val = parameter_token_args[self.parameter_token_val_args[parameter_token]]
                parameter_token_val = parameter_token_val.reshape(-1, 1)
                parameter_embd = self.parameter_token_connector[parameter_token](parameter_token_val)

                if self.parameter_token_pos_args[parameter_token][0] in parameter_token_args: 
                    parameter_token_pos_0 = parameter_token_args[self.parameter_token_pos_args[parameter_token][0]]
                    parameter_token_pos_1 = parameter_token_args[self.parameter_token_pos_args[parameter_token][1]]
                else:
                    parameter_token_mask = (input_ids == self.parameter_token_id[parameter_token])
                    parameter_token_pos_0, parameter_token_pos_1= torch.nonzero(parameter_token_mask, as_tuple=True)

                if parameter_token_pos_0.size(dim=0) > 0:
                    parameter_embd = parameter_embd.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.index_put(
                        (parameter_token_pos_0, parameter_token_pos_1), parameter_embd, accumulate=False
                    )
        
        return inputs_embeds

class L3MTIndxEmbedding(nn.Module):
    def __init__(self, config: L3MGenerationConfig, **kwargs) -> None:
        super().__init__()

        t_index_connector_cls = config.t_index_connector_config.get('cls', 'linear')
        if t_index_connector_cls == 'linear':
            self.t_index_token_connector = nn.Linear(1, config.hidden_size) 
        elif t_index_connector_cls == 'mlp':
            mlp_config = config.parameter_connector_config.get('mlp_config', {'depth': 2})
            dim_projection = config.hidden_size
            depth = mlp_config['depth']
            int_dim = mlp_config.get('interm_dim', 64)
            layers = [nn.Linear(1, int_dim)]
            for _ in range(depth):
                layers.extend([nn.GELU(), nn.Linear(int_dim, int_dim)])
            layers.extend([nn.GELU(), nn.Linear(int_dim, dim_projection)])
            self.t_index_token_connector = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'Parameter connector class {t_index_connector_cls} is not implemented.')

        self.t_index_token_id = config.t_index_token['token_id']

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        t_indices: Optional[torch.FloatTensor] = None,
        t_index_pos_0: Optional[torch.LongTensor] = None,
        t_index_pos_1: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        if t_indices == None:
            return inputs_embeds
        
        t_indices = t_indices.reshape(-1, 1)
        t_indices = (t_indices - 1175.) / 2350.
        t_indices_embd = self.t_index_token_connector(t_indices)

        if t_index_pos_0 == None:
            t_index_token_mask = (input_ids == self.t_index_token_id)
            t_index_pos_0, t_index_pos_1= torch.nonzero(t_index_token_mask, as_tuple=True)

            if t_index_pos_0.size(dim=0) > 0:
                t_indices_embd = t_indices_embd.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.index_put(
                    (t_index_pos_0, t_index_pos_1), t_indices_embd, accumulate=False
                )
    
        return inputs_embeds
