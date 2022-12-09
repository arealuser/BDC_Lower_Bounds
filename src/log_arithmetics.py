from typing import Union

import numpy as np


def safe_log_arr(arr: np.ndarray):
    """
    Computes the logarithm of arr using numpy, but ensuring that 0s are mapped to -inf.
    This ensures that for any non-negative array arr: np.exp(safe_log(arr)) will be equal to arr.
    """
    return np.log(arr, out=-np.inf * np.ones(np.shape(arr)), where=(arr != 0))


def safe_log_scalar(scalar: Union[int, float, np.ndarray]):
    """
    Computes the logarithm of arr using numpy, but ensuring that 0s are mapped to -inf.
    This ensures that for any non-negative array arr: np.exp(safe_log(arr)) will be equal to arr.
    """
    if scalar == 0:
        return -np.inf
    return np.log(scalar)


def log_factorial(k):
    """
    Compute the logarithm of the factorial of k.
    """
    return float(np.sum(np.log(np.arange(1, k + 1))))


def log_sum(arr):
    """
    Compute log(sum(exp(arr)))
    """
    m = np.max(arr)
    m = np.nan_to_num(m)  # Avoid cases where m is negative infinity.
    return m + np.log(np.sum(np.exp(arr - m)))


def log_add(a, b):
    """
    Compute log(exp(a) + exp(b))
    """
    m = max(a, b)
    m = np.nan_to_num(m)  # Avoid cases where m is negative infinity.
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def log_sub(a, b):
    """
    Compute log(exp(a) - exp(b))
    """
    m = max(a, b)
    m = np.nan_to_num(m)  # Avoid cases where m is negative infinity.
    return m + safe_log_scalar(np.exp(a - m) - np.exp(b - m))


def log_sub_arr(a, b):
    """
    Compute log(exp(a) - exp(b))
    """
    m = np.maximum(a, b)
    m = np.nan_to_num(m)  # Avoid cases where m is negative infinity.
    return m + safe_log_arr(np.exp(a - m) - np.exp(b - m))


def log_over(n, k):
    """
    Returns the logarithm of n choose k.
    """
    if k > n:
        return -np.inf
    return LOG_FACTORIALS[n] - (LOG_FACTORIALS[k] + LOG_FACTORIALS[n - k])


def log_over_arr(n, k):
    """
    Returns the logarithm of n choose k when they are given as arrays.
    When n is smaller than k, the result is clipped to return 0.
    """
    res = LOG_FACTORIALS[n] - (LOG_FACTORIALS[k] + LOG_FACTORIALS[np.clip(n - k, 0, None)])
    res[n < k] = -np.inf
    return res


LOG_FACTORIALS: np.ndarray = np.zeros(1)