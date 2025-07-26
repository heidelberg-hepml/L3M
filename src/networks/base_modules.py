import torch
import torch.nn.functional as F
from torch import nn

import math
from functools import partial
from itertools import pairwise

from typing import Optional

class TokenEmbd(nn.Module):
    def __init__(
        self,
        shape: list[int]
    ):
        super().__init__()

        self.token_embd = nn.Parameter(torch.empty(shape))
        torch.nn.init.normal_(self.token_embd)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.token_embd

class ResidualLayerMLP(nn.Module):
    def __init__(self, hidden_dim, interm_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, interm_dim),
            nn.SiLU(),
            nn.Linear(interm_dim, hidden_dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x + self.res_block(x)

class ResidualStackMLP(nn.Module):
    def __init__(self, hidden_dim, interm_dim, n_layers):
        super(ResidualStackMLP, self).__init__()
        self.stack = nn.ModuleList([ResidualLayerMLP(hidden_dim, interm_dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.silu(x)
        return x
    

class ResidualLayerCNN(nn.Module):
    def __init__(self, hidden_dim, interm_dim, kernel_size=3, padding=1):
        super(ResidualLayerCNN, self).__init__()
        self.res_block = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(
                hidden_dim,
                interm_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False # can also be set to true
            ),
            nn.SiLU(),
            nn.Conv2d(
                interm_dim,
                hidden_dim,
                kernel_size=1,
                stride=1,
                bias=False # can also be set to true
            )
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x + self.res_block(x)

class ResidualStackCNN(nn.Module):
    def __init__(self, hidden_dim, interm_dim, n_layers, kernel_size=3, padding=1):
        super(ResidualStackCNN, self).__init__()
        self.stack = nn.ModuleList(
            [
                ResidualLayerCNN(hidden_dim, interm_dim, kernel_size=kernel_size, padding=padding)
                for _ in range(n_layers)
            ]
        )
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for layer in self.stack:
            x = layer(x)
        x = F.silu(x)
        return x

