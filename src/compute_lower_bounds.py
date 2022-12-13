import numpy as np
import copy
from typing import List, Tuple

from scipy import stats

import code_for_lower_bounds.src.rzk_tables

EPSILON = 1E-300

BASE_SEQUENCES = [
    [1], [2], [3],
    [1, 1], [1, 2], [2, 1],
    [1, 1, 1]
]


class RunDistribution:
    distribution: np.ndarray  # The distribution of the length of runs
    channel_parameter: float  # The parameter of the PRC / BDC channel.
    average_length: float  # The average length of a run in this distribution
    deletion_probability: float  # The probability that a random run from this distribution will be deleted.
    channel_type: str  # The type of the channel.

    ordered_distribution: List[
        Tuple[float, int]]  # An ordered array of the distribution that can be used to find all entries
    # with at least some given probability.
    r_dist: np.ndarray  # The distribution of the total lengths of combined runs due to deletions of intermediary runs.
    z_dist: np.ndarray  # The distribution of runs conditioned on them not being deleted.
    s_i_dist: np.ndarray  # The distribution of the length of a single run, conditioned on it being deleted.
    k_dist: np.ndarray  # The distribution on lengths of runs in the output channel.

    def __init__(self, dist: np.ndarray, channel_param: float, channel_type: str = 'PRC'):
        self.k_dist = np.zeros(1)  # Will be changed later on.

        self.distribution = np.array(copy.copy(dist))
        # Lower bound distribution values to avoid logarithm / division by 0.
        self.distribution = np.clip(self.distribution, EPSILON, None)
        self.channel_parameter = channel_param

        assert channel_type in {'PRC', 'BDC'}
        self.channel_type = channel_type

        # Compute parameters of the distribution.
        self.average_length = float(np.dot(np.arange(len(self.distribution)), self.distribution))
        self._compute_deletion_probability()
        self._compute_r_z_dists()

        # Compute a list of the possible inputs ordered by their likelihood.
        # This is used only for the inefficient legacy code for sampling high-likelihood patterns.
        self.ordered_distribution = [(np.log(p), i) for i, p in enumerate(self.distribution)]
        self.ordered_distribution.sort()

    def get_log_prob_seq(self, sequence: List[int]):
        """
        Given a sequence of input runs, returns the natural logarithm of the probability that it will occur as a result
            of this run distribution.
        """
        base_log_prob = (np.log(self.deletion_probability) * (len(sequence) - 1))
        log_probs = np.array([np.log(self.distribution[s]) for s in sequence])
        return base_log_prob + np.sum(log_probs) + np.log(1 - np.exp(-sequence[0] * self.channel_parameter))

    def _compute_deletion_probability(self):
        """
        Computes the probability that a run from the given distribution will be completely deleted by the channel.
        """
        if self.channel_type == 'PRC':
            self.deletion_probability = float(np.dot(
                np.exp(-self.channel_parameter * np.arange(len(self.distribution))),
                self.distribution))
        elif self.channel_type == 'BDC':
            self.deletion_probability = float(np.dot(
                self.channel_parameter ** np.arange(len(self.distribution)),
                self.distribution))

    def _get_runs_above_threshold(self, threshold: float) -> List[Tuple[float, int]]:
        """
        Legacy code.
        Returns all the runs whose likelihood is above a given threshold.
        """
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
        r_max = code_for_lower_bounds.src.rzk_tables.RZK_TABLE.shape[0]
        pr_j_r = np.zeros((r_max, r_max))
        pr_j_r[0, 0] = 1 - self.deletion_probability
        p0 = 1.
        for j in range(1, r_max):
            p0 *= self.deletion_probability
            if p0 < EPSILON:
                break
            for r in range(r_max):
                dr = r - min(r, len(self.distribution) - 1)
                pr_j_r[j, r] = np.dot(pr_j_r[j - 1, dr:r], self.distribution[1:r + 1][::-1]) * self.deletion_probability

        self.r_dist = np.sum(pr_j_r, axis=0)
        if self.channel_type == 'PRC':
            # The distribution of runs, conditioned on them not being completely deleted by the channel.
            self.z_dist = self.distribution * (1 - np.exp(-self.channel_parameter * np.arange(len(self.distribution)))) \
                          / (1 - self.deletion_probability)
            # The distribution of the length of a single run, conditioned on it being deleted.
            self.s_i_dist = self.distribution * np.exp(-self.channel_parameter * np.arange(len(self.distribution))) \
                            / self.deletion_probability
        elif self.channel_type == 'BDC':
            # The distribution of runs, conditioned on them not being completely deleted by the channel.
            self.z_dist = self.distribution * (1 - self.channel_parameter ** np.arange((len(self.distribution)))) \
                          / (1 - self.deletion_probability)
            # The distribution of the length of a single run, conditioned on it being deleted.
            self.s_i_dist = self.distribution * (self.channel_parameter ** np.arange(len(self.distribution))) \
                            / self.deletion_probability
        else:
            raise RuntimeError(f"Unsupported channel type {self.channel_type}.")

    def _compute_lower_bound_PRC(self, verbose: bool = False) -> float:
        r_max, z_max, k_max = code_for_lower_bounds.src.rzk_tables.RZK_TABLE.shape
        k_probs = np.zeros(k_max)  # Will be used to store the distribution of the lengths of runs.
        average_total_run_length = ((
                (1 + self.deletion_probability) / (
                1 - self.deletion_probability))) * self.average_length  # The average total length of runs in a type
        first_term = -self.channel_parameter * average_total_run_length  # The first term of our lower bound which corresponds to the log(d^{r+z+s}) term in MD07's formula.
        second_term = np.log(
            self.channel_parameter) * self.channel_parameter * average_total_run_length  # The second term which corresponds to the log(\lambda^k) term in MD07's formula

        prs = np.reshape(self.r_dist, (-1, 1, 1))  # The probability of having a family with a given r
        rs = np.reshape(np.arange(r_max), (-1, 1, 1))  # The values of r

        effective_z_max = min(z_max,
                              len(self.z_dist))  # The distribution of z is bounded by both the size of our pre-cached data and the input distribution.

        pzs = np.reshape(self.z_dist[1:effective_z_max],
                         (1, -1, 1))  # The probability of having a family with a given z
        zs = np.reshape(np.arange(1, effective_z_max), (1, -1, 1))  # The values of z
        z_was_not_deleted = 1 - np.exp(
            -self.channel_parameter * zs)  # The probability that the run of length z was not deleted. Is used for conditioning in the probability of k.

        ks = np.reshape(np.arange(1, k_max), (1, 1, -1))  # The values of k
        lpk = (np.log(self.channel_parameter) * ks) + code_for_lower_bounds.src.rzk_tables.RZK_TABLE[:,
                                                      1:effective_z_max, 1:] - \
              (self.channel_parameter * (
                      rs + zs))  # logarithm of the probability of having a run of length k in the received codeword originate from a given family, not taking into account the conditioning on the z not being deleted.

        rzk_probs = np.exp(lpk) * prs * pzs / z_was_not_deleted  # The joint probability distribution of r, z and k
        k_probs[1:] = np.sum(rzk_probs, axis=(0, 1))  # The probabilities of run lengths on the output channel

        last_terms = np.sum(rzk_probs * code_for_lower_bounds.src.rzk_tables.RZK_TABLE[:, 1:effective_z_max, 1:])
        k_probs[k_probs < EPSILON] = EPSILON
        self.k_dist = k_probs
        zero_term = -np.dot(k_probs[1:], np.log(k_probs[1:] / np.sum(k_probs)))
        if verbose:
            print(f'{zero_term=}')
            print(f'{first_term=}')
            print(f'{second_term=}')
            print(f'{last_terms=}')
            print(f'{average_total_run_length=}')
        return ((zero_term + first_term + second_term + last_terms) / average_total_run_length) / np.log(2)

    def _compute_lower_bound_BDC(self, verbose: bool = False) -> float:
        r_max, z_max, k_max = code_for_lower_bounds.src.rzk_tables.RZK_TABLE.shape
        k_probs = np.zeros(k_max)  # Will be used to store the distribution of the lengths of runs.
        average_total_run_length = ((1 + self.deletion_probability) / (1 - self.deletion_probability)) \
                                   * self.average_length  # The average total length of runs in a type

        # The first term corresponds to H(d) * sum_k P_k / (1-d) (the sum of (52) and (53) in the Appendix of MD06).
        first_term = stats.bernoulli(self.channel_parameter).entropy() * average_total_run_length

        # The second term corresponds to H(P)/\sum_j j P_j \cdot \sum_k k \P_k/(1-d)
        # (equal to (54) in the Appendix of MD06).
        second_term = stats.entropy(self.distribution) * average_total_run_length / self.average_length

        # TODO: Make sure the way we are computing this corresponds to a lower bound.
        prs = np.reshape(self.r_dist, (-1, 1, 1))  # The probability of having a family with a given r
        rs = np.reshape(np.arange(r_max), (-1, 1, 1))  # The values of r

        effective_z_max = min(z_max, len(self.z_dist))  # The distribution of z is bounded by both the size of our
        # pre-cached data and the input distribution.

        pzs = np.reshape(self.z_dist[1:effective_z_max],
                         (1, -1, 1))  # The probability of having a family with a given z
        zs = np.reshape(np.arange(1, effective_z_max), (1, -1, 1))  # The values of z
        # The probability that the run of length z will not be deleted.
        # This is used for conditioning in the probability of k.
        z_was_not_deleted = 1 - (self.channel_parameter ** zs)

        ks = np.reshape(np.arange(1, k_max), (1, 1, -1))  # The values of k
        # logarithm of the probability of having a run of length k in the received codeword originating
        # from a given family, not taking into account the conditioning on the z not being deleted.
        log_prob_k = (np.log(((1 - self.channel_parameter) / self.channel_parameter)) * ks) + \
                     code_for_lower_bounds.src.rzk_tables.RZK_TABLE[:, 1:effective_z_max, 1:] + \
                     (np.log(self.channel_parameter) * (rs + zs))
        self.log_prob_k = log_prob_k

        # The joint probability distribution of r, z and k
        rzk_probs = np.exp(log_prob_k) * prs * pzs / z_was_not_deleted
        self.rzk_probs = rzk_probs
        k_probs[1:] = np.sum(rzk_probs, axis=(0, 1))  # The probabilities of run lengths on the output channel

        # The third term corresponds to the average of log(over(r+z, k) - over(r, k)).
        third_term = -np.sum(rzk_probs *
                            np.clip(code_for_lower_bounds.src.rzk_tables.RZK_TABLE[:, 1:effective_z_max, 1:],
                                    np.log(EPSILON), None))
        k_probs = np.clip(k_probs, EPSILON, None)
        self.k_dist = k_probs
        # The 0th term is the entropy of the K distributions themselves.
        zero_term = -np.dot(k_probs[1:], np.log(k_probs[1:] / np.sum(k_probs)))

        if verbose:
            print(f'{zero_term=}')
            print(f'{first_term=}')
            print(f'{second_term=}')
            print(f'{third_term=}')
            print(f'{average_total_run_length=}')
        return ((zero_term - first_term - third_term) / average_total_run_length) / np.log(2)

    def compute_lower_bound(self, verbose: bool = False) -> float:
        """
        Computes a lower bound on the rate of the MD07 / MD06 code using their bound for the appropriate channel.
        """
        if verbose:
            print(f"Computing a lower bound on the given distribution for the {self.channel_type}.")
        if self.channel_type == 'PRC':
            return self._compute_lower_bound_PRC(verbose=verbose)
        elif self.channel_type == 'BDC':
            return self._compute_lower_bound_BDC(verbose)
        else:
            raise RuntimeError(f"Unknown channel type {self.channel_type}")
