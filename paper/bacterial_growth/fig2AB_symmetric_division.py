"""
Fig. 2A-B: symmetric cell division with Gamma-distributed division times.

    S -> S + S      division, Gamma(shape alpha), mean division time 1/r_div

A: scaled probability densities of the division time for several shape
   parameters alpha (alpha = 1 is the Markovian, exponential case).
B: growth of a population of 100 cells for each alpha, at a fixed mean
   division time. The more deterministic the division time (large alpha), the
   slower the growth and the more visible the synchronised generations.

The growth curves ship precomputed in ``results/`` (DelaySSA, 10 realisations);
set ``recompute = True`` to regenerate them (a few minutes).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # so that `import nomass` works without installing it
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data and output paths are relative to this script

import numpy as np
import matplotlib.pyplot as plt
import nomass
from nomass import distributions as dist

plt.rcParams.update({'font.size': 16})

PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']   # colour-blind safe

r_div = 1 / 71.6                       # mean division time 71.6 h (C. crescentus, 31 C)
alpha_list = [100, 10, 1, 0.5, 0.3]
Tend_of_alpha = {100: 550, 10: 550, 1: 350, 0.5: 250, 0.3: 125}   # simulated time [h] per alpha
N_start = 100
recompute = False


def main():
    # ---- A: scaled densities of the division time
    tau = np.linspace(0.01, 3, 1000)
    plt.figure(figsize=(6, 4.2))
    for ai, alpha in enumerate(alpha_list):
        pdf = dist.pdf(tau, 'Gamma', 1.0, alpha)
        pdf = pdf / (np.max(pdf[15:]) if alpha < 1 else np.max(pdf))   # diverging densities: scale on tau > 0.05
        plt.plot(tau, pdf, lw=2, label=r'$\alpha = %s$' % alpha, color=PALETTE[ai])
    plt.axvline(x=1, color='gray', lw=2, ls='--')
    plt.legend(fontsize=13)
    plt.xlabel(r'Time to division [$1/\lambda_{div}$]')
    plt.ylabel('Scaled PDF')
    plt.ylim(0, 1.1)
    plt.xlim(0, 3)
    plt.tight_layout()
    plt.show()

    # ---- B: population growth for each alpha
    plt.figure(figsize=(6, 4.2))
    for ai, alpha in enumerate(alpha_list):
        Tend = Tend_of_alpha[alpha]
        cache = 'results/pop_results_symmetric_%s_delay.npy' % alpha
        if recompute or not os.path.exists(cache):
            print('Simulating alpha = %s ...' % alpha)
            population = run_symmetric_division(alpha, Tend)
            np.save(cache, population)
        else:
            population = np.load(cache)
        time = np.linspace(0, Tend, population.shape[1]) * r_div
        plt.plot(time, population[:, :, 0].mean(axis=0), lw=2, label=r'$\alpha = %s$' % alpha, color=PALETTE[ai])
    plt.yscale('log')
    plt.ylim(80, 1.1e4)
    plt.legend(fontsize=13)
    plt.xlabel(r'Time [$1/\lambda_{div}$]')
    plt.ylabel('Population size')
    plt.tight_layout()
    plt.show()


def run_symmetric_division(alpha, Tend, N_simulations=10):
    division = nomass.Reaction_channel('S -> S + S', rate=r_div, shape_param=alpha, distribution='Gamma',
                                       method='Delay', name='Division')
    sim = nomass.Gillespie_simulation({'S': N_start}, [division], Tend=Tend, N_simulations=N_simulations,
                                      timepoints=100, unit='h')
    return sim.run_simulations(verbose=False)


if __name__ == '__main__':
    main()
