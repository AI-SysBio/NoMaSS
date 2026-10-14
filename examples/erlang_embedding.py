"""
Hidden Markov embedding: a Gamma waiting time as a chain of exponential stages.

A Gamma distribution with integer shape k and mean 1/r is the sum of k
exponential stages of rate k*r (an Erlang distribution). The non-Markovian
reaction A -> B can therefore be embedded in a purely Markovian chain

    A -> A_1 -> A_2 -> ... -> A_{k-1} -> B

at the cost of k - 1 hidden species. This script simulates both descriptions
and overlays the population of B: the curves agree within the Monte Carlo
noise. The embedding only exists for integer shapes (phase-type
distributions generalise it), which is why NoMaSS simulates the
distribution directly.

Run with:  python examples/erlang_embedding.py
"""
import numpy as np
import matplotlib.pyplot as plt
import nomass

r, k = 1.0, 5
settings = dict(Tend=4, N_simulations=20, timepoints=100, unit='h')

# 1) non-Markovian channel simulated directly
direct = nomass.Reaction_channel('A -> B', rate=r, shape_param=k, distribution='Gamma')
sim_direct = nomass.Gillespie_simulation({'A': 500, 'B': 0}, [direct], **settings)
pop_direct = sim_direct.run_simulations(verbose=False)

# 2) Markovian embedding with k - 1 hidden stages
stages = ['A'] + ['A_%d' % i for i in range(1, k)] + ['B']
chain = [nomass.Reaction_channel('%s -> %s' % (stages[i], stages[i + 1]), rate=k * r) for i in range(k)]
N_init = {s: 0 for s in stages}
N_init['A'] = 500
sim_chain = nomass.Gillespie_simulation(N_init, chain, **settings)
pop_chain = sim_chain.run_simulations(verbose=False)

t = np.linspace(0, settings['Tend'], settings['timepoints'])
B_direct = pop_direct[:, :, 1].mean(axis=0)
B_chain = pop_chain[:, :, -2].mean(axis=0)
print('Max difference of the mean B population: %.1f cells (out of 500)' % np.max(np.abs(B_direct - B_chain)))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(t, B_direct, lw=3, label='Gamma(k=%d) channel, MOSAIC' % k)
ax.plot(t, B_chain, lw=2, ls='--', label='%d exponential stages' % k)
ax.set_xlabel('Time [h]')
ax.set_ylabel('Population of B')
ax.legend()
plt.show()
