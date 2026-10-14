"""
NoMaSS: Non-Markovian Stochastic Simulation.

A Gillespie-based framework to simulate reaction networks whose inter-event
times follow arbitrary distributions, mixing three exact methods per reaction
channel: MOSAIC (rejection), Laplace Gillespie and DelaySSA.

Quick start::

    import nomass

    r1 = nomass.Reaction_channel('A -> B', rate=1, shape_param=20, distribution='Gamma')
    sim = nomass.Gillespie_simulation({'A': 300, 'B': 0}, [r1], Tend=10, N_simulations=20, unit='h')
    populations = sim.run_simulations()
    sim.plot_populations()

Reference: Pelissier A., Phan M., Beerenwinkel N., Rodriguez Martinez M.
"Gillespie-based simulation and inference for non-Markovian stochastic reaction
networks", Briefings in Bioinformatics 27(5), bbag486 (2026). doi:10.1093/bib/bbag486
"""

from .simulation import Reaction_channel, Reactant, Gillespie_simulation, parse_reaction, PALETTE
from .distributions import DISTRIBUTIONS, METHODS
from . import distributions

__version__ = '1.0.0'

__all__ = ['Reaction_channel', 'Reactant', 'Gillespie_simulation', 'parse_reaction',
           'distributions', 'DISTRIBUTIONS', 'METHODS', 'PALETTE', '__version__']
