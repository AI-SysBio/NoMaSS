"""
Fig. 2C: computational cost per event versus system size for each backend.

Benchmark system: N particles undergoing the transition S -> S with a
gamma-distributed waiting time (the population stays constant, so the cost per
event can be measured at a fixed system size). Each configuration is run until
about 500,000 events have been simulated.

Backends compared (one curve each):
    alpha = 100, 10      MOSAIC
    alpha = 1            standard Gillespie SSA
    alpha = 0.5, 0.3     Laplace Gillespie
    alpha = 1            DelaySSA (direct method, and Anderson's modified next
                         reaction method for the published measurements)

The measurements of the paper ship in ``results/computational_cost.npz`` and
are plotted without any post-processing. They were taken with the original
implementation on one machine; absolute values depend on the hardware and the
NoMaSS version. Set ``recompute = True`` to measure again with the NoMaSS
engine of this repository (several hours; NoMaSS implements the direct
DelaySSA variant only).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # so that `import nomass` works without installing it
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data and output paths are relative to this script

import time
import numpy as np
import matplotlib.pyplot as plt
import nomass

plt.rcParams.update({'font.size': 14})

PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', 'gray', 'black']
ALPHA_LIST = [100, 10, 1, 0.5, 0.3]
N_LIST = [10, 33, 100, 333, 1000, 3333, 10000, 33333]
N_SIMULATIONS = 5
TOTAL_EVENTS = 400000           # Tend is chosen so that N * Tend * rate ~ TOTAL_EVENTS
recompute = False


def method_for(alpha):
    if alpha == 1:
        return 'MOSAIC', 'Gillespie'
    return ('Laplace', 'Laplace') if alpha < 1 else ('MOSAIC', 'MOSAIC')


def measure(N, alpha, method):
    """Return (elapsed seconds, number of events) for N particles."""
    channel = nomass.Reaction_channel('S -> S', rate=1.0, shape_param=alpha, distribution='Gamma', method=method)
    sim = nomass.Gillespie_simulation({'S': N}, [channel], Tend=TOTAL_EVENTS / N, N_simulations=N_SIMULATIONS,
                                      timepoints=100)
    start = time.perf_counter()
    sim.run_simulations(verbose=False)
    return time.perf_counter() - start, channel.number_of_accepted_reactions


def compute():
    labels = [r'$\alpha = %s$ (%s)' % (a, method_for(a)[1]) for a in ALPHA_LIST] + ['DelaySSA [direct]']
    seconds = np.zeros((len(labels), len(N_LIST)))
    events = np.zeros_like(seconds)
    for j, N in enumerate(N_LIST):
        for i, alpha in enumerate(ALPHA_LIST):
            print('N = %d, alpha = %s' % (N, alpha))
            seconds[i, j], events[i, j] = measure(N, alpha, method_for(alpha)[0])
        print('N = %d, DelaySSA' % N)
        seconds[-1, j], events[-1, j] = measure(N, 1, 'Delay')
    np.savez('results/computational_cost_nomass.npz', N=np.array(N_LIST), labels=np.array(labels),
             seconds=seconds, events=events)
    return 'results/computational_cost_nomass.npz'


def main():
    if recompute:
        source = compute()
    else:
        source = 'results/computational_cost.npz'
    data = np.load(source)
    N, labels, seconds, events = data['N'], data['labels'], data['seconds'], data['events']

    plt.figure(figsize=(7, 4.5))
    for i, label in enumerate(labels):
        cost = seconds[i] / events[i]
        plt.plot(N, cost, 'o--', color=PALETTE[i % len(PALETTE)], label=str(label))
        plt.fill_between(N, cost * 0.9, cost * 1.1, color='gray', alpha=0.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('System size')
    plt.ylabel('Runtime / event [s]')
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
