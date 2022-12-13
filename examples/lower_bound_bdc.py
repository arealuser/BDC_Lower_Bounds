import time
from datetime import timedelta
from typing import Tuple
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import basinhopping

from code_for_lower_bounds.src import compute_lower_bounds, lower_bound_baa_optimization, rzk_tables
import numpy as np

K_MAX = 128
R_MAX = 1024
Z_MAX = 1024


def params_to_distribution(params: Tuple[float, float, float], step_limit: int):
    """
    Uses the given hyper-parameters to optimize the distribution for the BDC with the BAA heuristic.
    """
    lam = params[0]
    l_val = params[1] / lam
    beta = params[2] * lam * l_val
    dist = lower_bound_baa_optimization.generate_optimized_distribution(channel_parameter=lam,
                                                                        average_cost_target=l_val,
                                                                        deletion_penalty=beta,
                                                                        step_limit=step_limit,
                                                                        delta=1E-4,
                                                                        verbose=True,
                                                                        channel_type='BDC')
    return dist


def geometric_dist(md07_param):
    return stats.geom(1 - md07_param).pmf(np.arange(Z_MAX))


if __name__ == '__main__':
    # print('Computing cache tables...')
    rzk_tables.compute_RZK_table_BDC(R_MAX, Z_MAX, K_MAX)

    # Recreate the numbers in MD07:
    md07df = pd.DataFrame(columns=["d", "p", "rate"],
                          data=[[0.05, 0.53, 0.72829],
                                [0.10, 0.57, 0.56196],
                                [0.15, 0.62, 0.43918],
                                [0.20, 0.67, 0.34669],
                                [0.25, 0.72, 0.27588],
                                [0.30, 0.77, 0.22243],
                                [0.35, 0.81, 0.18101],
                                [0.40, 0.84, 0.14841],
                                [0.45, 0.87, 0.12286],
                                [0.50, 0.89, 0.10186],
                                [0.55, 0.91, 0.084323],
                                [0.60, 0.92, 0.069564],
                                [0.65, 0.93, 0.056858],
                                [0.70, 0.94, 0.045324],
                                [0.75, 0.96, 0.035984],
                                [0.80, 0.97, 0.027266],
                                [0.85, 0.98, 0.019380],
                                [0.90, 0.985, 0.012378],
                                [0.95, 0.993, 0.005741]])
    my_rates = np.zeros(len(md07df))
    new_rates = np.zeros(len(md07df))
    count = 0
    t0 = time.time()
    # for i in range(len(md07df) - 1, -1, -1):
    for i in range(16, 17):
        p = md07df["p"][i]
        d = md07df["d"][i]
        dist = geometric_dist(p)



        def get_dist(l_val: float, beta: float, verbose: bool = False) -> np.ndarray:
            dist2 = np.zeros(dist.shape)
            dist2[1:] = lower_bound_baa_optimization.generate_optimized_distribution(channel_parameter=d,
                                                                                     average_cost_target=l_val,
                                                                                     deletion_penalty=beta,
                                                                                     step_limit=100,
                                                                                     delta=1E-4,
                                                                                     verbose=verbose,
                                                                                     channel_type='BDC',
                                                                                     initial_distribution=dist[1:],
                                                                                     alphabet_size=len(dist))
            dist2 = dist2 / np.sum(dist2)
            return dist2


        def get_score(params: Tuple[float, float]) -> float:
            l_val, beta = params
            global count
            count += 1
            td = timedelta(seconds=time.time() - t0)
            print(f'get_score with {i=}, {l_val=}, {beta=}, {count=}, {str(td)}')
            if min(l_val, beta) < 0:
                return 100

            dist2 = get_dist(l_val, beta)
            rd2 = compute_lower_bounds.RunDistribution(dist2, d, 'BDC')
            lb2 = rd2.compute_lower_bound(False)
            print(f'{lb2=}')
            return -lb2


        #
        opt_obj = basinhopping(get_score, (np.dot(dist, np.arange(len(dist))), 15), niter=5)
        print(opt_obj)
        best_lb = -opt_obj.fun
        best_dist = get_dist(opt_obj.x[0], opt_obj.x[1])
        with open(f'results/dist_{i}.pkl', 'wb') as f:
            pickle.dump((opt_obj, best_dist), f)

        # best_dist = get_dist(43.67778036, 1383.37898488, True)
        # print(compute_lower_bounds.RunDistribution(best_dist, d, 'BDC').compute_lower_bound(True))
        fig = plt.figure(figsize=(8, 8), dpi=400)
        plt.plot(best_dist, label='Optimized Distribution')
        plt.plot(dist, label='MD07 Distribution')
        plt.ylabel('PMF', fontsize='x-large')
        plt.xlabel('Run Length', fontsize='x-large')
        plt.legend(fontsize='large')
        plt.title('Optimized Distribution', fontsize='xx-large')
        plt.tight_layout()
        plt.savefig(f'results/distribution_{i}.pdf')
        # plt.show()

        new_rates[i] = best_lb
        rd = compute_lower_bounds.RunDistribution(dist, d, 'BDC')
        lb = rd.compute_lower_bound(False)

        my_rates[i] = lb

        print(f"{lb=},\t{new_rates[i]=},\t{md07df['d'][i]=},\t{md07df['p'][i]=},\t{md07df['rate'][i]=}")

    md07df["check"] = my_rates
    md07df["new rate"] = new_rates

    print(md07df)
