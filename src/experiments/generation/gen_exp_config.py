from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import warnings

from src.utils.exp_config import ExpConfig

@dataclass
class GenExpConfig(ExpConfig):

    lc_shape: tuple[int, int, int] = field(
        default=(2350, 140, 140)
    )
    """
    (`tuple[int]`, defaults to `(2350, 140, 140)`):

    Shape of the lightcone of the form (time, x, y).
    """

    lc_connector: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """
    (`dict`, defaults to `{}`):

    Configuration of the lightcone (brightness temperature) input connector . 
    """

    parameter_connector: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """
    (`dict`, defaults to `{}`):

    Configuration of the parameter input connector . 
    """

    t_index_connector: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """
    (`dict`, defaults to `{}`):

    Configuration of the time input connector . 
    """

    cfm_connector: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """
    (`dict`, defaults to `{}`):

    Configuration of the cfm input connector . 
    """

    lora: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """
    (`dict`, defaults to `{}`):

    Configuration of LoRA. 
    """

    t_init: int = field(
        default=0
    )
    """
    (`int`, defaults to 0):

    The number of initial lightcone slices for which no loss is computed. 
    """

    t_max: int = field(
        default=12
    )
    """
    (`int`, defaults to 12):

    The number of slices a sublight consists of. The loss is computed over t_max - t_init slices.
    """

    chunked_sublc_size: int = field(
        default = 24
    )
    """
    (`int`, defaults to 12):

    The number of slices a chunked sublight consists of. During training, a sublightcone is consisting of t_max slices is sampled in every epoch.
    """

    weight_decay: float = field(
        default = 1.e-3
    )
    """
    (`float`, defaults to `1.e-3`):

    Weight decay.
    """

    compile_for_eval: bool = field(
        default=False
    )
    """
    (`bool`, defaults to `False`):

    Whether to use the model should be compiled for evaluation.
    """

    eval_batch_size: int = field(
        default = 8
    )
    """
    (`int`, defaults to 8):

    Batch size for evaluation.
    """

    eval_generation_lcs: int = field(
        default=10
    )
    """
    (`int`, defaults to 10):

    Number of lightcones to generate (both for proper autoregressive generation and conditioning on the ground truth) during evaluation. 

    The value -1 implies that all lightcones of the test set should be used.
    """

    eval_l2_lcs: int = field(
        default=10
    )
    """
    (`int`, defaults to 10):

    Number of lightcones to determine the l2 metric of the generated patches conditioned on the ground truth.

    The value -1 implies that all lightcones of the test set should be used.
    """

    eval_t_max: int = field(
        default=2
    )
    """
    (`int`, defaults to 2):

    The number of slices an evaluation sublightcone consists of.

    It must satisfy `eval_t_max > eval_t_init`.
    """


    eval_t_init: int = field(
        default=12
    )
    """
    (`int`, defaults to 12):

    The number of initial slices that the generation is condition on. 
    
    It must satisfy `eval_t_max > eval_t_init` and `eval_t_init >= t_init`.
    """
    
    eval_l2_samples: int = field(
        default=1
    )
    """
    (`int`, defaults to 1):

    Number of samples to generate per patch for the l2 metric of the generated patches conditioned on the ground truth.
    """

    def check_args(self):
        super().check_args()
