import torch
import numpy.ma as ma
import numpy as np


def get_mask_from_tensor(X):
    """
    Create a mask tensor from the original tensor, if nan -> true, else false
    Args:
        X (torch.Tensor) : Tensor with(out) nan values

    Returns:
        torch.BoolTensor
    """
    X_masked = torch.where(torch.isnan(torch.from_numpy(X)), torch.tensor([0]), torch.tensor([1]))
    X_masked = X_masked.bool()
    return X_masked


def get_mask_from_numpy_tensor(X):
    """
    Create a mask tensor from a numpy tensor
    Args:
        X (np.ndarray) : original tensor

    Returns:
        bool or ndarray of bool: Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.
        This is a scalar if `x` is a scalar.
    """
    mask = ma.masked_invalid(X)
    return np.logical_not(mask.mask)

