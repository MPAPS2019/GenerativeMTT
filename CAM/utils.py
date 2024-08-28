import torch
import numpy as np

class SemeanticSegmentationTarget:
    def __init__(self, mask, category = None):
        if category == None:
            self.category = 0
        else:
            self.category = category

        if isinstance(mask, torch.Tensor):
            self.mask = mask
        elif isinstance(mask, np.ndarray):
            self.mask = torch.from_numpy(mask)

    def __call__(self, output):
        return (output[self.category]*self.mask).sum()
