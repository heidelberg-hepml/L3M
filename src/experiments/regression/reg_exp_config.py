from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import warnings

from src.utils.exp_config import ExpConfig

@dataclass
class RegExpConfig(ExpConfig):
    number_of_runs: int = field(
        default=1.
    )
    """
    (`int`, defaults to 1):

    Number of runs in this experiment.
    """

    """
    Config of the regression experiment.
    """

    lc_connector: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """
    (`dict`, defaults to `{}`):

    Configuration of the lightcone (brightness temperature) input connector . 
    """

    head: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """
    (`dict`, defaults to `{}`):

    Configuration of the parameter output connector. 
    """

    num_sys_prompt_tokens: int = field(
        default=0
    )
    """
    (`int`, defaults to 0):

    Number of system prompt tokens. 
    """

    num_pre_lc_tokens: int = field(
        default=0
    )
    """
    (`int`, defaults to 0):

    Number of pre-lightcone tokens. 
    """

    custom_number_of_layers: Optional[int] = field(
        default=None
    )
    """
    (`int`, *optional*, defaults to `None`):
    
    For a randomly initialized backbone, a custom number of hidden backbone layers can be specified.
    """

    train_ds_multiplicity: int = field(
        default=1
    )
    """
    (`int`, defaults to 1):

    How many instances should the final train dataset contain the loaded dataset.
    """

    def check_args(self):
        super().check_args()

        assert self.number_of_runs > 0, f"The number_of_runs must be at least 1. Current value; {self.number_of_runs}"
    
        if self.load_llm_weights and self.custom_number_of_layers != None:
            warnings.warn("You are using a pretrained backbone. The field custom_number_of_layers is ignored.")

        assert self.train_ds_multiplicity > 0, f"The train_ds_multiplicity must be larger than 0. Current value: {self.train_ds_multiplicity}"
