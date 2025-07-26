from dataclasses import dataclass, field, fields
from typing import Optional

import json

@dataclass
class ExpConfig:
    """
    Base config of the experiments
    """

    run_name: str = field(
        default=""
    )
    """
    (`str`, defaults to `""`):

    Configuration name for the run
    """

    DATASET_DIR: str = field(
        default=""
    )
    """
    (`str`, defaults to `""`):
    
    Dir containing the data
    """

    test_ds_size: int = field(
        default=250
    )
    """
    (`int`, defaults to 250):

    Test dataset size.
    """

    train_eval_ratio: float = field(
        default=0.9
    )
    """
    (`float`, defaults to 0.9):

    Ratio for the train and validation dataset. Range: (0., 1.).
    """

    ds_ratio: float = field(
        default=1.
    )
    """
    (`float`, defaults to 1.0):

    Specifies the ratio of how many examples of the training dataset should be loaded. Range: (0, 1.].
    """

    load_llm_weights: bool = field(
        default=True
    )
    """
    (`bool`, defaults to `True`):

    Whether to load pretrained LLM weights
    """

    LLM_DIR: Optional[str] = field(
        default=None
    )
    """
    (`str`, *optional*, defaults to `None`):
    
    Dir containing the LLM weights, the config file, and the model files configuration.py, modeling.py, processing.py
    """

    preprocess: Optional[str] = field(
        default=None
    )
    """
    (`str`, *optional*, defaults to `None`):
    
    Preprocessing method for the lightcone values. Options: None, Normalize, NormalizeLog
    """

    tf32: bool = field(
        default=True
    )
    """
    (`bool`, defaults to `True`):

    Enable/disable tf32
    """

    train_from_scratch: bool = field(
        default=False
    )
    """
    (`bool`, defaults to `False`):
    
    Train the network from scratch
    """

    scratch_config: Optional[dict] = field(
        default=None
    )
    """
    (`dict`, *optional*, defaults to `None`):

    Configuration for the network trained from scratch
    """

    attn_implementation: str = field(
        default="sdpa"
    )
    """
    (`str`, defaults to `"sdpa"`):
    
    Attention implementation. Options: eager, sdpa, flex_attention, flash_attention_2
    """

    attn_drop: float = field(
        default=0.
    )
    """
    (`float`, defaults to 0.):
    
    Attention dropout. Range: [0., 1.]
    """

    patch_size: tuple[int, int, int] = field(
        default=(1,1,1)
    )
    """
    (`tuple[int, int, int]`, defaults to `(1, 1, 1)`):

    Patch size of the lightcone of the form (time, x, y).
    """

    pad_sequence: bool = field(
        default=True
    )
    """
    (`bool`, defaults to `True`):

    Whether to pad the sequence lengths to a multiple of 8. This improves the efficiency of the Tensor Cores, especially for long sequences.
    """

    minimalistic_chat_template: bool = field(
        default=False
    )
    """
    (`bool`, defaults to `False`):

    Whether the prompting template should be minimal.
    """

    batch_size: int = field(
        default=1
    )
    """
    (`int`, defaults to 1):

    Instantanous batch size, i.e. without gradient accumulation.
    """

    gradient_accumulation_steps: int = field(
        default=1
    )
    """
    (`int`, defaults to 1):

    Gradient accumulation steps
    """

    warmup_epochs: int = field(
        default=0
    )
    """
    (`int`, defaults to 0):

    Number of warmup epochs
    """

    num_train_epochs: int = field(
        default=1
    )
    """
    (`int`, defaults to 0):

    Number of epochs
    """

    decay_epochs: int = field(
        default=0
    )
    """
    (`int`, defaults to 0):

    Number of decay epochs
    """

    lr: float = field(
        default=5.e-5
    )
    """
    (`float`, defaults to 5.e-5):

    Learning rate
    """

    epochs_for_log: float = field(
        default=1.
    )
    """
    (`float`, defaults to 1.):

    The training logs the training loss after that many epochs. It can also be partial epochs.
    """

    dataloader_threads: int = field(
        default=0
    )
    """
    (`int`, defaults to 0):

    Number of dataloader threads
    """

    dataloader_pin_memory: bool = field(
        default=True
    )
    """
    (`bool`, defaults to `True`):

    Whether the dataloader should pin the tensor to the gpu.
    """

    dataloader_persistent: bool = field(
        default=True
    )
    """
    (`bool`, defaults to `True`):

    Whether the dataloader is persistent.
    """

    compile: bool = field(
        default=True
    )
    """
    (`bool`, defaults to `True`):

    Whether the model should be compiled with torch.
    """

    max_grad_norm: float = field(
        default=-1
    )
    """
    (`float`, defaults to -1):

    Maximal gradient norm for gradient clipping. A value of -1 disables gradient clipping.
    """

    max_num_checkpoints: int = field(
        default=-1
    )
    """
    (`float`, defaults to -1):

    Maximum number of checkpoints to be saved. A value of -1 means that every checkpoint is saved.
    """

    def check_args(self):
        assert self.attn_drop >= 0. and self.attn_drop <= 1., f"Attention dropout must be in the range [0., 1.]. Current value: {self.attn_drop}"

        if self.load_llm_weights and self.LLM_DIR == None:
            raise ValueError("LLM_DIR must be specified if LLM weights are to be loaded.")
        
        if self.train_from_scratch and self.scratch_config == None:
            raise ValueError("scratch_config cannot be None for a network trained from scratch.")

        assert self.train_eval_ratio < 1. and self.train_eval_ratio > 0., f"The train_eval_ratio must be between 0 and 1. Current value: {self.train_eval_ratio}"
        assert self.ds_ratio <= 1. and self.ds_ratio > 0., f"ds_ratio must be in the range (0., 1.]. Current value: {self.ds_ratio}"
        assert self.epochs_for_log >= 0., f"epochs_for_log must be non-negative. Current value: {self.epochs_for_log}"


    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}
    
    def from_dict(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
        self.check_args()

    def _load(self, config_file: str):
        with open(config_file, "r") as f:
            d = json.load(f)
        self.from_dict(d)

    @classmethod
    def load(cls, config_file: str) -> "ExpConfig":
        exp_config = cls()
        exp_config._load(config_file)        
        return exp_config
    
    def save(self, config_file: str):
        d = self.to_dict()
        with open(config_file, "w") as f:
            json.dump(d, f)

