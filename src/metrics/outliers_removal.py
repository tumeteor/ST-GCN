import itertools
import math

import numpy as np


def remove_outliers_by_percentiles(arr, lower=2.5, upper=97.5):
    """
    A more straight-forward way of removing outliers using percentile, only the one in between lower and
    upper bounds are retained, which in the default case is 95%.
   Args:
       arr (np.ndarray): the target array of floats
       lower (float): the lower limit that data under that will be removed
       upper (float): the upper limit that data over that will be removed
   Returns:
       np.ndarray: the filtered array

   """
    percentiles = np.percentile(arr, [lower, upper])
    return arr[(percentiles[0] < arr) & (percentiles[1] > arr)]


def remove_outliers(arr, rate=1.5, poisson_distribution=True):
    """
     Use interquartile range for filtering outliers in data. Outliers here are defined as observations that fall below
     Q1 − rate * IQR or above Q3 + rate * IQR. The default rate is set to 1.5.
     For possion distribution, we follow the modification of IQR in https://wis.kuleuven.be/stat/robust/papers/2008/outlierdetectionskeweddata-revision.pdf
    Args:
        arr (np.ndarray): 1D array containing data with `float` type.
        rate (float): the rate decides the gaps for upper-bound and lower-bound
        poisson_distribution (bool): whether we assume the data distribution is Possion

    Returns:
        np.ndarray: the filtered array

    """
    upper_quartile = np.percentile(arr, 75)
    lower_quartile = np.percentile(arr, 25)
    IQR = upper_quartile - lower_quartile
    lower_offset, upper_offset = _cal_offset(arr, IQR, rate) if poisson_distribution else (IQR * rate, IQR * rate)
    quartile_set = (lower_quartile - lower_offset, upper_quartile + upper_offset)
    result = arr[np.where((arr >= quartile_set[0]) & (arr <= quartile_set[1]))]

    return np.array(result)


def _cal_offset(arr, IQR, rate):
    """
    calculate the offset for cutting-off in Poission distribution
    Args:
        arr (np.ndarray): 1D array containing data with `float` type.
        IQR (float):
        rate (float:

    Returns:
        tuple(float, float): the lower and upper offsets

    """
    left_median = []
    right_median = []
    med = np.median(arr)
    for x in arr:
        if x < med:
            left_median.append(x)
        else:
            right_median.append(x)
    h = []
    for x, y in list(itertools.product(left_median, right_median)):
        h.append(float((y - med) - (med - x)) / (y - x))
    mc = np.median(h)
    compensation_pos, compensation_neg = math.exp(-4 * mc) if mc > 0 else math.exp(-3 * mc), math.exp(3 * mc) if mc > 0 else math.exp(3 * mc)
    return IQR * rate * compensation_pos, IQR * rate * compensation_neg
