import numpy as np
from scipy import stats
import tqdm
import time

BAA_EPSILON = 1E-30


def apply_penalty(alphas, L_i, l2):
    power = -L_i * l2
    log_alphas = np.log(alphas) + power
    mla = np.max(log_alphas)
    probs = np.exp(log_alphas - mla)
    probs /= np.sum(probs)
    return probs


def score_lambda2(alphas, L_i, L_target, l2):
    probs = apply_penalty(alphas, L_i, l2)
    return np.sign(L_target - np.dot(probs, L_i))


def find_lambda2_bs(alphas, L_i, L_target):
    lb = 0.0
    ub = 1.0
    while score_lambda2(alphas, L_i, L_target, ub) < 0:
        ub *= 2

    num_iterations = 100
    mid = (ub + lb) / 2
    for i in range(num_iterations):
        mid = (lb + ub) / 2
        sign = score_lambda2(alphas, L_i, L_target, mid)
        if np.abs(sign) < BAA_EPSILON:
            return mid, apply_penalty(alphas, L_i, mid)
        elif sign < 0:
            lb = mid
        else:
            ub = mid

    return mid, apply_penalty(alphas, L_i, mid)


def get_score(P_ji, Q, L_i, D_i, beta):
    Q = np.reshape(Q, (1, -1))
    denominator = np.reshape(np.sum(P_ji * Q, axis=1), (-1, 1))
    denominator[denominator < BAA_EPSILON] = BAA_EPSILON
    return (np.sum(Q * P_ji * np.log(P_ji / denominator)) - (beta * np.dot(Q, D_i))) / np.dot(Q, L_i)


def get_score_theoretical(P_ji, Q, Q_prev, L_target, D_i, beta):
    Q = np.reshape(Q, (1, -1))
    Q_prev = np.reshape(Q_prev, (1, -1))
    denominator = np.reshape(np.sum(P_ji * Q_prev, axis=1), (-1, 1))
    denominator[denominator < BAA_EPSILON] = BAA_EPSILON
    Q_prev = np.clip(Q_prev, BAA_EPSILON, 1)
    W_ji = (Q_prev * P_ji) / denominator
    return (np.sum(Q * P_ji * (np.log(W_ji) - np.log(Q))) - beta * np.dot(Q, D_i)) / L_target


def get_log_alphas(P_ji, Q, D_i, beta):
    Q = np.reshape(Q, (1, -1))
    denominator = np.reshape(np.sum(P_ji * Q, axis=1), (-1, 1))
    denominator[denominator < BAA_EPSILON] = BAA_EPSILON
    return np.sum(P_ji * np.log(Q * P_ji / denominator), axis=0) - beta * D_i


def do_baa_step(P_ji, Q, L_target, beta, L_i, D_i):
    log_alphas = get_log_alphas(P_ji, Q, D_i, beta)
    mla = np.max(log_alphas)
    alphas = np.exp(log_alphas - mla)
    l2, new_Q = find_lambda2_bs(alphas, L_i, L_target)
    return new_Q


def generate_BAA_params(l: float, BAA_N: int, verbose: bool = True, channel_type: str = 'PRC'):
    if verbose:
        print(f'Computing BAA transition probs {BAA_N - 1}x{BAA_N} (~2 ** %.1f)' % (np.log2(BAA_N * (BAA_N - 1))))
    t0 = time.time()
    I = np.arange(1, BAA_N)
    J = np.arange(0, BAA_N)
    IJ, JI = np.meshgrid(I, J)
    if channel_type == 'PRC':
        P_ji = np.clip(stats.poisson(IJ * l).pmf(JI), BAA_EPSILON, 1)
    elif channel_type == 'BDC':
        P_ji = np.clip(stats.binom(IJ, l).pmf(JI), BAA_EPSILON, 1)
    else:
        raise RuntimeError(f"Unknown channel_type {channel_type}")
    if verbose:
        print('That took %.1f seconds' % (time.time() - t0))
    D_i = P_ji[0, :]
    L_i = I
    return P_ji, D_i, L_i


def generate_optimized_distribution(l: float, L_target: float, beta: float, verbose: bool = True,
                                    step_limit: int = 100, delta: float = 0.005, BAA_N: int = 512,
                                    channel_type: str = 'PRC'):
    """
    Uses the weighted version of the Blahut-Arimoto algorithm to generate a potential distribution for an MD07-type code.
    """
    P_ji, D_i, L_i = generate_BAA_params(l, BAA_N, verbose, channel_type)
    Q = np.ones(BAA_N - 1)
    next_Q = Q
    d = 0.0
    steps = range(step_limit)
    if verbose:
        steps = tqdm.trange(step_limit)
    for _ in steps:
        next_Q = np.clip(do_baa_step(P_ji, Q, L_target, beta, L_i, D_i), 1E-100, None)
        d = np.max(np.log(Q / next_Q))
        if d < delta:
            break
        Q = next_Q
    if verbose:
        print(
            f'The expected score of this distribution is {get_score(P_ji, next_Q, L_i, D_i, beta)}, and it is at most {d} from optimal.')
    return next_Q
