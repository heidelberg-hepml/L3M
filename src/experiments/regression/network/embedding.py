import torch
from torch import nn
from src.experiments.regression.network.configuration import L3MRegressorConfig

from typing import Optional, Tuple, Union

class L3MLcEmbedding(nn.Module):
    def __init__(self, config: L3MRegressorConfig, wte=None, **kwargs) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.wte = wte
        self.hidden_size = config.hidden_size
        
        lc_connector_cls = config.lc_connector_config.get('cls', 'linear')
        if lc_connector_cls == 'linear':
            self.lc_connector = nn.Linear(1, self.hidden_size)
        elif lc_connector_cls == 'mlp':
            mlp_config = config.lc_connector_config.get('mlp_config', {'depth': 2})
            dim_projection = self.hidden_size
            int_dim = mlp_config.get('interm_dim', 64)
            depth = mlp_config['depth']
            layers = [nn.Linear(1, int_dim)]
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
        lc_values: Union[torch.FloatTensor, torch.LongTensor],
        positions: Optional[Tuple[torch.LongTensor, torch.LongTensor]] = None
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        if positions == None or positions[0] == None:
            positions = torch.nonzero(input_ids < 0, as_tuple=True)

        has_image = positions[0].size(dim=0) > 0
        input_ids = input_ids.clamp_min(0).clamp_max(self.vocab_size).detach()
        if not self.model_from_scratch:
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = torch.zeros(*input_ids.shape, self.hidden_size, dtype=self.wte.weight.dtype, device=input_ids.device)

        if has_image:
            lc_features = self.lc_connector(lc_values)
            hidden_states = hidden_states.index_put(
                positions, lc_features.to(hidden_states.dtype), accumulate=False
            )
        return hidden_states
