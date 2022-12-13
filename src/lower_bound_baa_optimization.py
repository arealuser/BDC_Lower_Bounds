from typing import Optional

import numpy as np
from scipy import stats
import tqdm
import time

BAA_EPSILON = 1E-100


def apply_penalty(alphas, letter_costs, l2):
    power = -letter_costs * l2
    log_alphas = np.log(alphas) + power
    mla = np.max(log_alphas)
    probs = np.exp(log_alphas - mla)
    probs /= np.sum(probs)
    return probs


def score_lambda2(alphas, letter_costs, average_cost_target, l2):
    probs = apply_penalty(alphas, letter_costs, l2)
    return np.sign(average_cost_target - np.dot(probs, letter_costs))


def find_lambda2_bs(alphas, letter_costs, average_cost_target):
    lb = 0.0
    ub = 1.0
    while score_lambda2(alphas, letter_costs, average_cost_target, ub) < 0:
        ub *= 2

    num_iterations = 100
    mid = (ub + lb) / 2
    for i in range(num_iterations):
        mid = (lb + ub) / 2
        sign = score_lambda2(alphas, letter_costs, average_cost_target, mid)
        if np.abs(sign) < BAA_EPSILON:
            return mid, apply_penalty(alphas, letter_costs, mid)
        elif sign < 0:
            lb = mid
        else:
            ub = mid

    return mid, apply_penalty(alphas, letter_costs, mid)


def get_score(transition_matrix, input_distribution, letter_costs, deletion_probabilities, deletion_penalty):
    input_distribution = np.reshape(input_distribution, (1, -1))
    denominator = np.reshape(np.sum(transition_matrix * input_distribution, axis=1), (-1, 1))
    denominator[denominator < BAA_EPSILON] = BAA_EPSILON
    return (np.sum(input_distribution * transition_matrix * np.log(transition_matrix / denominator)) -
            (deletion_penalty * np.dot(input_distribution, deletion_probabilities))) \
           / np.dot(input_distribution, letter_costs)


def get_score_theoretical(transition_matrix, input_distribution, previous_input_distribution, average_cost_target,
                          deletion_probabilities, deletion_penalty):
    input_distribution = np.reshape(input_distribution, (1, -1))
    previous_input_distribution = np.reshape(previous_input_distribution, (1, -1))
    denominator = np.reshape(np.sum(transition_matrix * previous_input_distribution, axis=1), (-1, 1))
    denominator[denominator < BAA_EPSILON] = BAA_EPSILON
    previous_input_distribution = np.clip(previous_input_distribution, BAA_EPSILON, 1)
    W_ji = (previous_input_distribution * transition_matrix) / denominator
    return (np.sum(input_distribution * transition_matrix * (np.log(W_ji) - np.log(input_distribution))) -
            deletion_penalty * np.dot(input_distribution, deletion_probabilities)) / average_cost_target


def get_log_alphas(transition_matrix, input_distribution, deletion_probabilities, deletion_penalty):
    input_distribution = np.reshape(input_distribution, (1, -1))
    denominator = np.reshape(np.sum(transition_matrix * input_distribution, axis=1), (-1, 1))
    denominator[denominator < BAA_EPSILON] = BAA_EPSILON
    return np.sum(transition_matrix * np.log(input_distribution * transition_matrix / denominator),
                  axis=0) - deletion_penalty * deletion_probabilities


def do_baa_step(transition_matrix, input_distribution, average_cost_target, deletion_penalty, letter_costs,
                deletion_probabilities):
    log_alphas = get_log_alphas(transition_matrix, input_distribution, deletion_probabilities, deletion_penalty)
    max_log_alphas = np.max(log_alphas)
    alphas = np.exp(log_alphas - max_log_alphas)
    l2, next_input_distribution = find_lambda2_bs(alphas, letter_costs, average_cost_target)
    return next_input_distribution


def generate_BAA_params(channel_parameter: float, alphabet_size: int, verbose: bool = True, channel_type: str = 'PRC'):
    if verbose:
        print(f'Computing BAA transition probs {alphabet_size - 1}x{alphabet_size} (~2 ** %.1f)' %
              (np.log2(alphabet_size * (alphabet_size - 1))))
    t0 = time.time()
    input_index = np.arange(1, alphabet_size)
    output_index = np.arange(0, alphabet_size)
    input_indices, output_indices = np.meshgrid(input_index, output_index)
    if channel_type == 'PRC':
        transition_matrix = np.clip(stats.poisson(input_indices * channel_parameter).pmf(output_indices),
                                    BAA_EPSILON, 1)
    elif channel_type == 'BDC':
        transition_matrix = np.clip(stats.binom(input_indices, channel_parameter).pmf(output_indices), BAA_EPSILON, 1)
    else:
        raise RuntimeError(f"Unknown channel_type {channel_type}")
    if verbose:
        print('That took %.1f seconds' % (time.time() - t0))
    deletion_probabilities = transition_matrix[0, :]
    letter_costs = input_index
    return transition_matrix, deletion_probabilities, letter_costs


def generate_optimized_distribution(channel_parameter: float, average_cost_target: float, deletion_penalty: float,
                                    verbose: bool = True, step_limit: int = 100, delta: float = 0.005,
                                    alphabet_size: int = 512, channel_type: str = 'PRC',
                                    initial_distribution: Optional[np.ndarray] = None):
    """
    Uses a weighted version of the Blahut-Arimoto algorithm to generate a candidate distribution for an MD07-type code.
    """
    transition_matrix, deletion_probabilities, letter_costs = generate_BAA_params(channel_parameter, alphabet_size,
                                                                                  verbose, channel_type)
    if initial_distribution is None:
        input_distribution = np.ones(alphabet_size - 1)
    else:
        input_distribution = np.copy(initial_distribution)
    next_input_distribution = input_distribution
    d = 0.0
    steps = range(step_limit)
    if verbose:
        steps = tqdm.trange(step_limit)
    for _ in steps:
        next_input_distribution = np.clip(do_baa_step(transition_matrix, input_distribution, average_cost_target,
                                                      deletion_penalty, letter_costs, deletion_probabilities),
                                          BAA_EPSILON, None)
        d = np.max(np.log(input_distribution / next_input_distribution))
        if d < delta:
            break
        input_distribution = next_input_distribution
    if verbose:
        print(
            f'The expected score of this distribution is {get_score(transition_matrix, next_input_distribution, letter_costs, deletion_probabilities, deletion_penalty)}, and it is at most {d} from optimal.')
    return next_input_distribution
