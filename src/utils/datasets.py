import torch
from torch.utils.data.dataset import Dataset

from src.utils.preprocessing import Center

class L3MDataset(Dataset):
    """
    Simple torch dataset.
    """

    def __init__(self, generator, device="cpu"):
        self.ds: list[dict] = []
        with torch.no_grad():
            for el in generator:
                _el = { k: v.clone().detach().to(device=device) for k, v in el.items()}
                self.ds.append(_el)
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def get_default_label_names() -> list[str]:
    "Returns the label names of the simulation parameters in the default order"

    return ['m', 'Om', 'log LX', 'E0', 'log Tvir', 'zeta']

# These are the bounds of the simulation parameters
_lo=[0.3, 0.20, 38., 100., 4.0, 10.]
_hi=[10.0, 0.40, 42., 1500., 5.3, 250.]
normalizing_bounds = { name: { 'lower_value': _lo[i], 'upper_value': _hi[i] } for i, name in enumerate(get_default_label_names()) }

def get_bounds(label_names: list[str]) -> tuple[list, list]:
    """
    Returns the bounds of the requested parameter names as a tuple. The first element is a list containing the lower bounds and the second element a list of the upper bounds.
    
    Arguments:
        label_names (list[str]):
            The requested simulation parameter names. Options: "m", "Om", "log LX", "E0", "log Tvir", "zeta".
    """

    low_values = []
    high_values = []

    for name in label_names:
        assert name in normalizing_bounds.keys()
        low_values.append(normalizing_bounds[name]['lower_value'])
        high_values.append(normalizing_bounds[name]['upper_value'])

    return low_values, high_values

def get_preprocessing_layers(label_names: list[str]) -> Center:
    """
    Returns the preprocessing layer for the requested simulation parameters.
    
    Arguments:
        label_names (list[str]):
            The requested simulation parameter names. Options: "m", "Om", "log LX", "E0", "log Tvir", "zeta".
    """

    assert label_names != None

    low_values, high_values = get_bounds(label_names)

    return Center(lo=low_values, hi=high_values)
