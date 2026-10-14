"""
Fig. S2: symmetric division A -> 2A with gamma-distributed division times.

A: rescaled densities of the division time for several shape parameters alpha.
B: simulated population growth (100 cells, mean division time 1) for several alpha,
   with the fitted exponential growth rate k.
C: analytical doubling time t2 = mu ln2 / (alpha (2^(1/alpha) - 1)) (Supplementary Eq. 14).
D: analytical growth rate k = ln2 / t2.

Simulated populations ship precomputed in ``results/NonMarkovian_Division.npy``
(array alpha x realisations x time x [A, total]); set ``recompute = True`` to
regenerate them (a minute).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # so that `import nomass` works without installing it
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data and output paths are relative to this script

from math import log
import numpy as np
import matplotlib.pyplot as plt
import nomass
from nomass import distributions as dist

plt.rcParams.update({'font.size': 14})

MU_DIV = 1.0                      # mean division time
NU = 2                            # A -> nu A
ALPHA_SIM = [1, 2, 10, 100]
TEND, N_SIMULATIONS, TIMEPOINTS = 5, 10, 100
recompute = False


def doubling_time(alpha, mu=MU_DIV, nu=NU):
    """Population doubling time of A -> nu A for gamma division times (Eq. 14)."""
    return mu * log(2) / (alpha * (nu ** (1.0 / alpha) - 1))


def main():
    # ---- A: rescaled densities
    plt.figure(figsize=(6, 3.5))
    tau = np.linspace(0.02, 3 * MU_DIV, 1000)
    for alpha in [0.1, 1, 10, 100]:
        pdf = dist.pdf(tau, 'Gamma', 1 / MU_DIV, alpha)
        plt.plot(tau, pdf / np.max(pdf), lw=2, label=r'$\alpha$ = %s' % alpha)
    plt.axvline(x=1, ls='--', color='black', lw=2, alpha=0.5)
    plt.yticks([0, 1], ['0', '1'])
    plt.ylabel('PDF (rescaled)')
    plt.xlabel(r'Time to division [$\mu_{div}$]')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # ---- B: simulated growth
    cache = 'results/NonMarkovian_Division.npy'
    if recompute or not os.path.exists(cache):
        results = []
        for alpha in ALPHA_SIM:
            print('Simulating alpha = %s ...' % alpha)
            division = nomass.Reaction_channel('A -> A + A', rate=1 / MU_DIV, shape_param=alpha, distribution='Gamma')
            sim = nomass.Gillespie_simulation({'A': 100}, [division], Tend=TEND, N_simulations=N_SIMULATIONS,
                                              timepoints=TIMEPOINTS)
            results.append(sim.run_simulations(verbose=False))
        np.save(cache, np.array(results))
    results = np.load(cache)
    time = np.linspace(0, TEND, results.shape[2])
    plt.figure(figsize=(6, 3.5))
    for alpha, population in zip(ALPHA_SIM, results):
        mean = population[:, :, -1].mean(axis=0)
        k = np.polyfit(time[len(time) // 2:], np.log(mean[len(time) // 2:]), 1)[0]
        line, = plt.plot(time, mean, lw=2, label=r'$\alpha$ = %s' % alpha)
        plt.text(time[-1], mean[-1], ' k = %.2f' % k, color=line.get_color(), fontsize=10, va='center')
    plt.yscale('log')
    plt.ylabel('Population size')
    plt.xlabel(r'Time [$\mu_{div}$]')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # ---- C, D: analytical doubling time and growth rate
    alpha = np.logspace(-1.2, 1.7, num=200)
    t2 = doubling_time(alpha)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
    ax1.plot(alpha, t2, color='blue', lw=2)
    ax1.axhline(y=log(2) / log(NU), ls='--', color='black', alpha=0.5)
    ax1.axhline(y=log(2) / (NU - 1), ls='--', color='black', alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_yticks([0, log(2), 1])
    ax1.set_yticklabels(['0', 'ln 2', '1'])
    ax1.set_ylabel(r'Doubling time / $\mu_{div}$')
    ax1.text(alpha[-1], 1, 'Deterministic ', ha='right', va='bottom', fontsize=10)
    ax1.text(alpha[-1], log(2), 'Markovian ', ha='right', va='bottom', fontsize=10)
    ax2.plot(alpha, log(2) / t2, color='blue', lw=2)
    ax2.axhline(y=log(NU), ls='--', color='black', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Shape parameter $\alpha$')
    ax2.set_ylabel('Growth rate $k$')
    for ax in (ax1, ax2):
        ax.set_xlim(alpha.min(), alpha.max())
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
