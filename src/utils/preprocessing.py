import torch

class Preprocessing:
    """
    Preprocessing class.
    """

    def __init__(self):
        pass

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError()
    
    def reverse(self, x: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError()

class Normalize(Preprocessing):
    """
    Preprocessing class for normalizing data to zero mean and unit covariance.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x -= -32.50
        x /= 65.4
        return x
        
    def reverse(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x *= 65.4
        x += -32.50
        return x
        
class NormalizeLog(Preprocessing):
    """
    Preprocessing class for normalizing data. It first applies a bi-log scaling on the data and then normalizes it to zero mean and unit covariance.
    """

    def __init__(self):
        super().__init__()

        self.clamp = Clamp()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.clamp.forward(x)
        x -= -1.51
        x /= 2.74
        return x
            
    def reverse(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x *= 2.74
        x += -1.51
        x = self.clamp.reverse(x)
        return x

PREPROCESSING_CLASSES = {
    "Normalize": Normalize,
    "NormalizeLog": NormalizeLog
}

def get_preprocessing_classes() -> list[str]:
    return list(PREPROCESSING_CLASSES.keys())

def get_preprocessing(class_name:str) -> Preprocessing:
    if class_name in PREPROCESSING_CLASSES:
        return PREPROCESSING_CLASSES[class_name]()
    else:
        raise NotImplementedError(f"The preprocessing class {class_name} is not implemented.")

# Copied from heidelberg-hepml/skatr
class Center:
    """Shift and scale a tensor into the range [0,1] given min value `lo` and max value `hi`"""

    def __init__(self, lo, hi, dtype=torch.float32):
        self.lo = torch.tensor(lo, dtype=dtype)
        self.hi = torch.tensor(hi, dtype=dtype)

    def forward(self, x):
        self.lo = self.lo.to(x.device)
        self.hi = self.hi.to(x.device)
        return (x - self.lo) / (self.hi - self.lo)

    def reverse(self, x):
        self.lo = self.lo.to(x.device)
        self.hi = self.hi.to(x.device)
        return x * (self.hi - self.lo) + self.lo

# Copied from heidelberg-hepml/skatr
class Clamp:
    """Apply a symmetric log scaling to the input."""

    def forward(self, x):
        return x.abs().add(1).log() * x.sign()

    def reverse(self, x):
        return x.abs().exp().add(-1) * x.sign()
    
