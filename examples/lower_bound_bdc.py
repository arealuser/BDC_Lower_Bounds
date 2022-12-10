from typing import Tuple
import pickle

import matplotlib.pyplot as plt

from code_for_lower_bounds.src import compute_lower_bounds, lower_bound_baa_optimization, rzk_tables
import numpy as np

K_MAX = 256
R_MAX = 1024
Z_MAX = 1024


def params_to_distribution(params: Tuple[float, float, float], step_limit: int):
    """
    Uses the given hyper-parameters to optimize the distribution for the BDC with the BAA heuristic.
    """
    lam = params[0]
    l_val = params[1] / lam
    beta = params[2] * lam * l_val
    dist = lower_bound_baa_optimization.generate_optimized_distribution(l=lam,
                                                                        L_target=l_val,
                                                                        beta=beta,
                                                                        step_limit=step_limit,
                                                                        delta=1E-4,
                                                                        verbose=True,
                                                                        channel_type='BDC')
    return dist


if __name__ == '__main__':
    # print('Computing cache tables...')
    # rzk_tables.compute_RZK_table_BDC(R_MAX, Z_MAX, K_MAX)

    print('Generating optimized distribution...')

    dist2 = np.concatenate(([0], params_to_distribution((0.19, 7.72, 0.438), 10000)))
    dist2 = (1 / np.sum(dist2)) * dist2
    f = plt.figure(figsize=(5, 5), dpi=400)
    plt.plot(dist2)
    plt.xlabel('Run Length', fontsize='large')
    plt.ylabel('Frequency', fontsize='large')
    plt.tight_layout()
    plt.savefig('./distribution.pdf')
    # with open('temp2.pkl', 'wb') as f:
    #     pickle.dump(dist2, f)
    #
    # print('Computing a lower bound based on the optimized distribution...')
    # rd = compute_lower_bounds.RunDistribution(dist2, 0.19)
    # lb = rd.compute_lower_bound(verbose=True)
    # print(f'Generated a lower bound of {lb/rd.lam} for the capacity of the BDC')