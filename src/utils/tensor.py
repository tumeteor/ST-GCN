import torch
import numpy.ma as ma
import numpy as np


def get_mask_from_tensor(X):
    # Build mask tensor
    X_masked = torch.where(torch.isnan(torch.from_numpy(X)), torch.tensor([0]), torch.tensor([1]))
    X_masked = X_masked.bool()
    return X_masked


def get_mask_from_numpy_tensor(X):
    mask = ma.masked_invalid(X)
    return np.logical_not(mask.mask)

