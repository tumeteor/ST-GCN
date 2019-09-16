import numpy as np


def _error(actual: np.ndarray, predicted: np.ndarray, remove_outliers=None):
    """
    Simple error
    Args:
        actual (np.ndarray):
        predicted (np.ndarray):
        remove_outliers (:func:`remove_outliers`, optional): whether :func:`remove_outliers` is applied
    Returns:
       float: the metric value, can be negative
    """
    _assert_equal_lengths(actual, predicted)

    return remove_outliers(actual - predicted) if remove_outliers else actual - predicted


def mse(actual: np.ndarray, predicted: np.ndarray, remove_outliers=None):
    """
    Mean Squared Error
    Args:
        actual (np.ndarray):
        predicted:
        remove_outliers (:func:`remove_outliers`, optional): whether :func:`remove_outliers` is applied

    Returns:
        float: the metric value, from 0 to 1.0, the smaller the better
    """
    _assert_equal_lengths(actual, predicted)

    return np.mean(np.square(_error(actual, predicted))) if remove_outliers else \
        np.mean(np.square(_error(actual, predicted, remove_outliers)))


def rmse(actual: np.ndarray, predicted: np.ndarray, remove_outliers=None):
    """
    Root Mean Squared Error
    Args:
        actual(np.ndarray):
        predicted(np.ndarray):
        remove_outliers (:func:`remove_outliers`, optional): whether :func:`remove_outliers` is applied

    Returns:
        float: the metric value, from 0 to 1.0, the smaller the better
    """
    _assert_equal_lengths(actual, predicted)

    return np.sqrt(mse(actual, predicted)) if remove_outliers else np.sqrt(mse(actual, predicted, remove_outliers))


def smape(actual: np.ndarray, predicted: np.ndarray, remove_outliers=None):
    """
    Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based
    on percentage (or relative) errors.
    Args:
        actual(np.ndarray):
        predicted(np.ndarray):
        remove_outliers (:func:`remove_outliers`, optional): whether :func:`remove_outliers` is applied

    Returns:
        float: the metric value, from 0 to 200, the smaller the better
    """
    _assert_equal_lengths(actual, predicted)
    _smape = []
    for i in range(0, len(actual)):
        _smape.append(float(2 * np.abs(actual[i] - predicted[i])) / (actual[i] + predicted[i]))

    _smape = remove_outliers(np.array(_smape)) if remove_outliers else _smape
    return float(np.sum(_smape)) / len(_smape) * 100


def percentage_error(actual: np.ndarray, predicted: np.ndarray, remove_outliers=None):
    """
    Percentage error wrt. actual value.
    Args:
        actual(np.ndarray):
        predicted(np.ndarray):
        remove_outliers (:func:`remove_outliers`, optional): whether :func:`remove_outliers` is applied

    Returns:
       float: the metric value, from 0 to 1.0, the smaller the better
    """

    _percentage_errors = []
    for i in range(0, len(actual)):
        _percentage_errors.append(abs(float(predicted[i]) / actual[i] - 1.0))

    _percentage_errors = remove_outliers(np.array(_percentage_errors)) if remove_outliers else _percentage_errors

    return np.mean(_percentage_errors)


def _assert_equal_lengths(actual: np.ndarray, predicted: np.ndarray):
    assert len(actual) == len(predicted), f"Actual values and predicted values have different length:" \
        f" {len(actual), len(predicted)}"
