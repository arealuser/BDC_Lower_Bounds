import time
import numpy as np
import copy
from typing import List, Tuple

import tqdm

LOG_FACTORIALS: np.ndarray
RZK_TABLE: np.ndarray

EPSILON = 1E-30


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
    return m + np.log(np.sum(np.exp(arr - m)))


def log_add(a, b):
    """
    Compute log(exp(a) + exp(b))
    """
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def log_sub(a, b):
    """
    Compute log(exp(a) - exp(b))
    """
    m = max(a, b)
    return m + np.log(np.exp(a - m) - np.exp(b - m))


def log_sub_arr(a, b):
    """
    Compute log(exp(a) - exp(b))
    """
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) - np.exp(b - m))


def get_rzk_slow(r, z, k):
    """
    Compute log((((r+z) ^ k) - (r ^ k)) / (k!))
    """
    if r == 0:
        return (k * np.log(r + z)) - LOG_FACTORIALS[k]
    return log_sub(k * np.log(r + z), k * np.log(r)) - LOG_FACTORIALS[k]


def get_rzk_less_slow(rs, zs, ks):
    """
    Compute log((((r+z) ^ k) - (r ^ k)) / (k!)), using numpy vectorization.
    """
    return log_sub_arr(ks * np.log(rs + zs), ks * np.log(rs)) - LOG_FACTORIALS[1:]


def get_rzk(r, z, k):
    """
    Get the value of log((((r+z) ^ k) - (r ^ k)) / (k!)) from a cache (instead of computing it).
    """
    return RZK_TABLE[r, z, k]


def get_log_transition_prob_rzkl(r, z, k, lam):
    """
    Return log(lam^k e^(-lam) ((r+z)^k - r^k) / k!)
    """
    return (k * np.log(lam)) + get_rzk(r, z, k) - lam * (r + z)


BASE_SEQUENCES = [
    [1], [2], [3],
    [1, 1], [1, 2], [2, 1],
    [1, 1, 1]
]


class RunDistribution:
    distribution: np.ndarray  # The distribution of the length of runs
    lam: float  # The lambda parameter of the PRC
    average_length: float  # The average length of a run in this distribution
    D: float  # The probability that a random run from this distribution will be deleted.

    ordered_distribution: List[
        Tuple[float, int]]  # An ordered array of the distribution that can be used to find all entries
    # with at least some given probability.
    r_dist: np.ndarray  # The distribution of the total lengths of combined runs due to deletions of intermediary runs.
    z_dist: np.ndarray  # The distribution of runs conditioned on them not being deleted.
    s_i_dist: np.ndarray  # The distribution of the length of a single run, conditioned on it being deleted.
    k_dist: np.ndarray

    def __init__(self, dist: np.ndarray, channel_param: float):
        self.k_dist = np.zeros(1)
        self.distribution = np.array(copy.copy(dist))
        self.distribution[self.distribution < EPSILON] = EPSILON
        self.lam = channel_param
        self.average_length = float(np.dot(np.arange(len(self.distribution)), self.distribution))
        self.D = float(np.dot(np.exp(-self.lam * np.arange(len(self.distribution))), self.distribution))
        self.ordered_distribution = [(np.log(p), i) for i, p in enumerate(self.distribution)]
        self.ordered_distribution.sort()
        self._compute_r_z_dists()

    def get_log_prob_seq(self, sequence: List[int]):
        base_log_prob = (np.log(self.D) * (len(sequence) - 1))
        log_probs = np.array([np.log(self.distribution[s]) for s in sequence])
        return base_log_prob + np.sum(log_probs) + np.log(1 - np.exp(-sequence[0] * self.lam))

    def get_runs_above_threshold(self, threshold: float) -> List[Tuple[float, int]]:
        if self.ordered_distribution[-1][0] < -threshold:
            return []
        ub = len(self.distribution) - 1
        lb = 0
        while True:
            if ub == lb:
                mid = ub
                break
            mid = (ub + lb) // 2
            if self.ordered_distribution[mid][0] < -threshold:
                lb = mid
                if lb == ub - 1:
                    mid = ub
                    break
            elif self.ordered_distribution[mid][0] > -threshold:
                ub = mid
                if ub == lb + 1:
                    if lb != 0 or self.ordered_distribution[0][0] < -threshold:
                        break
                    else:
                        mid = lb
                        break
            else:
                break

        return self.ordered_distribution[mid:]

    def _compute_r_z_dists(self):
        """
        Use dynamic programming to compute the distributions of the lengths of runs united by the channel,
            and of the lengths of runs conditioned on them not being deleted by the channel.
        """
        # An r_max \times r_max array that holds in index j,r the probability to have j runs unite
        #   and have a total length of r.
        r_max = RZK_TABLE.shape[0]
        pr_j_r = np.zeros((r_max, r_max))
        pr_j_r[0, 0] = 1 - self.D
        p0 = 1.
        for j in range(1, r_max):
            p0 *= self.D
            if p0 < EPSILON:
                break
            for r in range(r_max):
                dr = r - min(r, len(self.distribution) - 1)
                pr_j_r[j, r] = np.dot(pr_j_r[j - 1, dr:r], self.distribution[1:r + 1][::-1]) * self.D

        self.r_dist = np.sum(pr_j_r, axis=0)
        self.z_dist = self.distribution * (1 - np.exp(-self.lam * np.arange(len(self.distribution)))) / (1 - self.D)
        self.s_i_dist = self.distribution * np.exp(-self.lam * np.arange(len(self.distribution))) / self.D

    def compute_lower_bound(self, verbose: bool = False):
        """
        Computes a lower bound on the rate of the MD07 code based on this input distribution.
        """
        r_max, z_max, k_max = RZK_TABLE.shape
        k_probs = np.zeros(k_max)  # Will be used to store the distribution of the lengths of runs.
        combined_arl = ((
                (1 + self.D) / (1 - self.D))) * self.average_length  # The average total length of runs in a type
        first_term = -self.lam * combined_arl  # The first term of our lower bound which corresponds to the log(d^{r+z+s}) term in MD07's formula.
        second_term = np.log(
            self.lam) * self.lam * combined_arl  # The second term which corresponds to the log(\lambda^k) term in MD07's formula

        prs = np.reshape(self.r_dist, (-1, 1, 1))  # The probability of having a family with a given r
        rs = np.reshape(np.arange(r_max), (-1, 1, 1))  # The values of r

        effective_z_max = min(z_max,
                              len(self.z_dist))  # The distribution of z is bounded by both the size of our pre-cached data and the input distribution.

        pzs = np.reshape(self.z_dist[1:effective_z_max],
                         (1, -1, 1))  # The probability of having a family with a given z
        zs = np.reshape(np.arange(1, effective_z_max), (1, -1, 1))  # The values of z
        z_was_not_deleted = 1 - np.exp(
            -self.lam * zs)  # The probability that the run of length z was not deleted. Is used for conditioning in the probability of k.

        ks = np.reshape(np.arange(1, k_max), (1, 1, -1))  # The values of k
        lpk = (np.log(self.lam) * ks) + RZK_TABLE[:, 1:effective_z_max, 1:] - (self.lam * (
                rs + zs))  # logarithm of the probability of having a run of length k in the received codeword originate from a given family, neglecting (the important) conditioning on the z not being deleted.

        rzk_probs = np.exp(lpk) * prs * pzs / z_was_not_deleted  # The joint probability distribution of r, z and k
        k_probs[1:] = np.sum(rzk_probs, axis=(0, 1))  # The probabilities of run lengths on the output channel

        last_terms = np.sum(rzk_probs * RZK_TABLE[:, 1:effective_z_max, 1:])
        k_probs[k_probs < EPSILON] = EPSILON
        self.k_dist = k_probs
        zero_term = -np.dot(k_probs[1:], np.log(k_probs[1:] / np.sum(k_probs)))
        if verbose:
            print(f'{zero_term=}')
            print(f'{first_term=}')
            print(f'{second_term=}')
            print(f'{last_terms=}')
            print(f'{combined_arl=}')
        return ((zero_term + first_term + second_term + last_terms) / combined_arl) / np.log(2)


def compute_RZK_table_PRC(r_max, z_max, k_max):
    """
    Computes the logarithm of the combinatorial formula used for computing lower bounds on the capacity of the PRC.
    """
    t0 = time.time()
    global LOG_FACTORIALS, RZK_TABLE
    LOG_FACTORIALS = np.array([0.0] + [log_factorial(k) for k in range(1, k_max)])

    print(f'Generating cache table of size {r_max}x{z_max}x{k_max} (~%.1f bits)...' % (np.log2(r_max * z_max * k_max)))

    RZK_TABLE = np.zeros((r_max, z_max, k_max))
    rs = np.reshape(np.arange(1, r_max), (-1, 1, 1))
    zs = np.reshape(np.arange(1, z_max), (1, -1, 1))
    ks = np.reshape(np.arange(1, k_max), (1, 1, -1))
    RZK_TABLE[1:, 1:, 1:] = get_rzk_less_slow(rs, zs, ks)
    for z in tqdm.trange(1, z_max):
        for k in range(1, k_max):
            RZK_TABLE[0, z, k] = get_rzk_slow(0, z, k)

    print('Done. Table generation took %.1f seconds' % (time.time() - t0))
