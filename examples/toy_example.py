"""
Quick start: a small reaction network mixing the three simulation methods.

    A -> B          Gamma waiting time,   simulated with DelaySSA
    B -> C + A      Weibull waiting time, simulated with MOSAIC (rejection)
    C -> 0          Gamma (shape < 1),    simulated with Laplace Gillespie
    A + B -> 0      two reactants:        Poisson process

Run with:  python examples/toy_example.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so that `import nomass` works without installing it
import nomass


class param:
    Tend = 10            # length of the simulation
    N_simulations = 20   # number of independent realisations
    timepoints = 100     # number of recorded time points (plots only)
    unit = 'h'           # time unit (plots only)


# Reaction channels: one waiting-time distribution and one method each
r1 = nomass.Reaction_channel(param, rate=1, shape_param=20, distribution='Gamma', method='Delay',
                             reactants=['A'], products=['B'], name='A -> B')
r2 = nomass.Reaction_channel(param, rate=4, shape_param=5, distribution='Weibull', method='MOSAIC',
                             reactants=['B'], products=['C', 'A'], name='B -> C + A')
r3 = nomass.Reaction_channel(param, rate=0.03, shape_param=0.5, distribution='Gamma', method='Laplace',
                             reactants=['C'], products=[], name='C -> 0')
r4 = nomass.Reaction_channel(param, rate=0.03,  # two reactants: Poisson process
                             reactants=['A', 'B'], products=[], name='A + B -> 0')

# Initial population (every species must be listed)
N_init = {'A': 300, 'B': 0, 'C': 0}

# Build and run the model
sim = nomass.Gillespie_simulation(N_init, param, [r1, r2, r3, r4])
print(sim)
populations = sim.run_simulations(verbose=True)   # array (N_simulations, timepoints, species + total)

# Plot the results
sim.plot_inter_event_time_distribution()
sim.plot_populations()
