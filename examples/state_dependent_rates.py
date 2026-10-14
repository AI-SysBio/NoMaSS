"""
State-dependent waiting times: logistic growth of a dividing population.

The division rate is updated after every iteration through ``update_function``
so that it decreases linearly with the population size (carrying capacity K).

Delay-based channels draw the waiting time when the reactant enters the
channel, so they cannot follow a rate that changes afterwards. Use MOSAIC or
Laplace Gillespie for state-dependent kinetics, as done here.

Run with:  python examples/state_dependent_rates.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so that `import nomass` works without installing it
import nomass


class param:
    Tend = 25
    N_simulations = 10
    timepoints = 100
    unit = 'h'


r0 = 1.0      # division rate of an isolated cell
K = 200       # carrying capacity

division = nomass.Reaction_channel(param, rate=r0, shape_param=5, distribution='Gamma', method='MOSAIC',
                                   reactants=['Cell'], products=['Cell', 'Cell'], name='Division')


def update_rates(sim):
    """Called after each iteration: rates may depend on the current state."""
    N = sim.reactant_population['Cell']
    sim.reaction_channel_list[0].rate = max(r0 * (1 - N / K), 1e-9)
    # sim.reaction_channel_list[0].shape_param can be changed the same way


sim = nomass.Gillespie_simulation({'Cell': 5}, param, [division])
populations = sim.run_simulations(update_function=update_rates, verbose=False)

print('Final population (expected close to K = %d): %s' % (K, populations[:, -1, 0]))
sim.plot_populations(reactant_list=['Cell'])
