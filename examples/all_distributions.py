"""
A cycle of eight reactions, one per supported waiting-time distribution.

    A -> B  Exponential      E -> F  Lognormal
    B -> C  Gamma            F -> G  Cauchy
    C -> D  Weibull          G -> H  Pareto
    D -> E  Gaussian         H -> A  Lomax

No ``method`` is given, so NoMaSS picks the default for each channel (MOSAIC
when the hazard is bounded, Laplace Gillespie otherwise). The printed model
shows the method chosen for each channel.

Run with:  python examples/all_distributions.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so that `import nomass` works without installing it
import nomass

species = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
distributions = [('Exponential', 1), ('Gamma', 15), ('Weibull', 2.5), ('Gaussian', 0.2),
                 ('Lognormal', 0.3), ('Cauchy', 0.1), ('Pareto', 3), ('Lomax', 3)]

channels = []
for i, (distribution, shape) in enumerate(distributions):
    reactant, product = species[i], species[(i + 1) % len(species)]
    channels.append(nomass.Reaction_channel('%s -> %s' % (reactant, product), rate=1, shape_param=shape,
                                            distribution=distribution,
                                            name='%s: %s -> %s' % (distribution, reactant, product)))

N_init = {s: 0 for s in species}
N_init['A'] = 200

sim = nomass.Gillespie_simulation(N_init, channels, Tend=15, N_simulations=20, unit='h', print_warnings=True)
print(sim)
sim.run_simulations(verbose=False)
sim.plot_inter_event_time_distribution()
sim.plot_populations(figsize=(8, 4))
