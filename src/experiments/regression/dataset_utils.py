from transformers.utils import ModelOutput

import torch

from typing import Optional, Tuple

from dataclasses import dataclass

@dataclass
class CausalLMOutputWithPastAndSigmas(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    sigmas: torch.FloatTensor = None
    ln_det_sigmas: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
