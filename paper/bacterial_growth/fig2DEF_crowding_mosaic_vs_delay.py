"""
Fig. 2D-F: asymmetric division of C. crescentus under crowding, MOSAIC vs DelaySSA.

    S -> S + Q      stalked cell division, Lognormal (CV 0.1), mean 71.6 h
    Q -> S          quiescent cell differentiation, Gamma (shape 3)

Crowding slows division: the division rate is r_div * K / (K + N), with N the
total population, and is updated after every event through ``update_function``.

MOSAIC evaluates the hazard at the current rate, so it follows the slowdown
exactly. DelaySSA draws the division time when the cell is born and cannot
revise it, so it keeps dividing at the rate that applied at birth and
overestimates the growth (E) and distorts the division-time distribution (F).

Results ship precomputed in ``results/`` (50 realisations per method); set
``recompute = True`` to regenerate them (about ten minutes).
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

r_div = 1 / 71.6              # free division rate [1/h]
r_diff = r_div / 0.4          # differentiation rate [1/h]
alpha_div = 0.1               # CV of the lognormal division time
alpha_diff = 3                # shape of the Gamma differentiation time
N_start = 100
K = 500                       # crowding constant
Tend = 3.5 / r_div / (K / (K + 500))
N_simulations = 50
recompute = False

C_MOSAIC, C_DELAY = 'red', 'blue'


def crowding(sim):
    """Called after every iteration: division slows down with the total population."""
    N = sim.reactant_population['S'] + sim.reactant_population['Q']
    sim.reaction_channel_list[0].rate = r_div * K / (K + N)


def run_crowding_model(method):
    division = nomass.Reaction_channel('S -> S + Q', rate=r_div * K / (K + N_start + 40), shape_param=alpha_div,
                                       distribution='Lognormal', method=method, name='Division: S -> S + Q')
    differentiation = nomass.Reaction_channel('Q -> S', rate=r_diff, shape_param=alpha_diff,
                                              distribution='Gamma', method='Delay', name='Differentiation: Q -> S')
    sim = nomass.Gillespie_simulation({'S': N_start, 'Q': 0}, [division, differentiation], Tend=Tend,
                                      N_simulations=N_simulations, timepoints=100, unit='h',
                                      update_function=crowding)
    population = sim.run_simulations(verbose=True)
    return population, np.asarray(division.wait_times)


def main():
    results = {}
    for method in ('MOSAIC', 'Delay'):
        pop_file = 'results/pop_results_%s.npy' % method
        wait_file = 'results/div_wait_times_%s.npy' % method
        if recompute or not (os.path.exists(pop_file) and os.path.exists(wait_file)):
            population, wait_times = run_crowding_model(method)
            np.save(pop_file, population)
            np.save(wait_file, wait_times)
        else:
            population, wait_times = np.load(pop_file), np.load(wait_file)
        results[method] = (population, wait_times)

    # ---- F: division-time distributions
    # Reference lognormal densities at the division rate of the beginning, middle and
    # end of the simulation (populations of ~140, ~490 and ~800 cells). Their heights
    # are scaled down to fit on the histogram axes.
    tau = np.linspace(1e-3, 350, 1000)
    plt.figure(figsize=(5.5, 4.2))
    for method, color, label in (('MOSAIC', C_MOSAIC, 'MOSAIC'), ('Delay', C_DELAY, 'DelaySSA')):
        plt.hist(results[method][1], alpha=0.3, bins=np.linspace(0, 350), density=True, label=label, color=color)
    for N, scale in ((N_start + 40, 3.3), (490, 2.5), (800, 2.3)):
        plt.plot(tau, dist.pdf(tau, 'Lognormal', r_div * K / (K + N), alpha_div) / scale, color='black', lw=1, ls='--')
    plt.xlim(40, 275)
    plt.ylim(0, 0.014)
    plt.yticks([0, 0.00625, 0.0125], ['0', '0.006', '0.012'])
    plt.xlabel('Division time [h]')
    plt.ylabel('PDF')
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()

    # ---- E: populations of stalked (solid) and quiescent (dashed) cells
    time = np.linspace(0, Tend, results['MOSAIC'][0].shape[1])
    plt.figure(figsize=(6, 4.2))
    for method, color, label in (('MOSAIC', C_MOSAIC, 'MOSAIC'), ('Delay', C_DELAY, 'DelaySSA')):
        population = results[method][0]
        for species, ls in ((0, '-'), (1, '--')):
            plt.plot(time, population[:, :, species].T, color=color, alpha=0.035, ls=ls)
            plt.plot(time, population[:, :, species].mean(axis=0), color=color, lw=2, ls=ls,
                     label=label if species == 0 else None)
    plt.xlabel('Time [h]')
    plt.ylabel('Cell population')
    plt.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    plt.show()

    for method in ('MOSAIC', 'Delay'):
        population = results[method][0]
        growth = np.polyfit(time[-30:], population[:, -30:, 0].mean(axis=0), 1)[0]
        print('%-7s final growth of S: %.2f cells/h' % (method, growth))


if __name__ == '__main__':
    main()
