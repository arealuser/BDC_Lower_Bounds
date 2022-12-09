import time

import numpy as np
import tqdm

import code_for_lower_bounds.src.log_arithmetics
from code_for_lower_bounds.src.log_arithmetics import log_sub, log_over, log_sub_arr, log_over_arr, log_factorial

RZK_TABLE: np.ndarray = np.zeros(1)


def get_rzk_slow_PRC(r, z, k):
    """
    Compute log((((r+z) ^ k) - (r ^ k)) / (k!))
    """
    if r == 0:
        return (k * np.log(r + z)) - code_for_lower_bounds.src.log_arithmetics.LOG_FACTORIALS[k]
    return log_sub(k * np.log(r + z), k * np.log(r)) - code_for_lower_bounds.src.log_arithmetics.LOG_FACTORIALS[k]


def get_rzk_slow_BDC(r, z, k):
    """
    Compute log(over(r+z,k) - over(r,k))
    """
    return log_sub(log_over(r + z, k), log_over(r, k))


def get_rzk_less_slow_PRC(rs, zs, ks):
    """
    Compute log((((r+z) ^ k) - (r ^ k)) / (k!)), using numpy vectorization.
    """
    return log_sub_arr(ks * np.log(rs + zs), ks * np.log(rs)) - code_for_lower_bounds.src.log_arithmetics.LOG_FACTORIALS[1:]


def get_rzk_less_slow_BDC(rs, zs, ks):
    """
    Compute log(over(r+z, k) - over(r,k)), using numpy vectorization.
    """
    return log_sub_arr(log_over_arr(rs + zs, ks), log_over_arr(rs, ks))


def get_rzk_cache(r, z, k):
    """
    Returns the rzk value from the cache.
    If we are dealing with a PRC, then this should be equal to log((((r+z) ^ k) - (r ^ k)) / (k!)).
    If we are dealing with a BDC, then this should be equal to log(over(r+z, k) - over(r,k)) (with log(0)=-np.inf).
    """
    global RZK_TABLE
    return RZK_TABLE[r, z, k]


def get_log_transition_prob_rzkl(r, z, k, lam):
    """
    Return log(lam^k e^(-lam) ((r+z)^k - r^k) / k!)
    """
    return (k * np.log(lam)) + get_rzk_cache(r, z, k) - lam * (r + z)


def compute_RZK_table_PRC(r_max, z_max, k_max):
    """
    Computes the logarithm of the combinatorial formula used for computing lower bounds on the capacity of the PRC.
    """
    t0 = time.time()
    global RZK_TABLE
    code_for_lower_bounds.src.log_arithmetics.LOG_FACTORIALS = np.array([0.0] + [log_factorial(k) for k in range(1, k_max)])

    print(f'Generating cache table of size {r_max}x{z_max}x{k_max} (~%.1f bits)...' % (np.log2(r_max * z_max * k_max)))

    RZK_TABLE = np.zeros((r_max, z_max, k_max))
    rs = np.reshape(np.arange(1, r_max), (-1, 1, 1))
    zs = np.reshape(np.arange(1, z_max), (1, -1, 1))
    ks = np.reshape(np.arange(1, k_max), (1, 1, -1))
    RZK_TABLE[1:, 1:, 1:] = get_rzk_less_slow_PRC(rs, zs, ks)
    for z in tqdm.trange(1, z_max):
        for k in range(1, k_max):
            RZK_TABLE[0, z, k] = get_rzk_slow_PRC(0, z, k)

    print('Done. Table generation took %.1f seconds' % (time.time() - t0))


def compute_RZK_table_BDC(r_max, z_max, k_max):
    """
    Computes the logarithm of the combinatorial formula used for computing lower bounds on the capacity of the BDC.
    """
    t0 = time.time()
    global RZK_TABLE
    code_for_lower_bounds.src.log_arithmetics.LOG_FACTORIALS = np.array([0.0] + [log_factorial(k) for k in range(1, k_max + r_max + z_max + 1)])

    print(f'Generating cache table of size {r_max}x{z_max}x{k_max} (~%.1f bits)...' % (np.log2(r_max * z_max * k_max)))

    RZK_TABLE = np.zeros((r_max, z_max, k_max))
    rs = np.reshape(np.arange(1, r_max), (-1, 1, 1))
    zs = np.reshape(np.arange(1, z_max), (1, -1, 1))
    ks = np.reshape(np.arange(1, k_max), (1, 1, -1))
    RZK_TABLE[1:, 1:, 1:] = get_rzk_less_slow_BDC(rs, zs, ks)
    for z in tqdm.trange(1, z_max):
        for k in range(1, k_max):
            RZK_TABLE[0, z, k] = get_rzk_slow_BDC(0, z, k)

    print('Done. Table generation took %.1f seconds' % (time.time() - t0))
