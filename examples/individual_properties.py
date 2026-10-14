"""
Individual properties: cells that remember their generation and size.

Every individual is a ``nomass.Reactant`` object and can carry arbitrary
attributes. The ``on_reaction`` hook is called after each accepted reaction
with the consumed and produced individuals, which is where properties are
set or inherited. Here each division halves the mother's size and
increments the generation of both daughters; death keeps the generation of
the dying cell for a final histogram.

Run with:  python examples/individual_properties.py
"""
import numpy as np
import matplotlib.pyplot as plt
import nomass

division = nomass.Reaction_channel('Cell -> Cell + Cell', rate=1.0, shape_param=8,
                                   distribution='Gamma', name='Division')
death = nomass.Reaction_channel('Cell -> 0', rate=0.3, name='Death')

generations_at_death = []


def inherit(sim, channel, consumed, produced):
    """Hook called after each accepted reaction."""
    if channel is division:
        mother = consumed[0]
        for daughter in produced:
            daughter.generation = mother.generation + 1
            daughter.size = mother.size / 2
    elif channel is death:
        generations_at_death.append(consumed[0].generation)


sim = nomass.Gillespie_simulation({'Cell': 20}, [division, death], Tend=6, N_simulations=1, unit='h',
                                  on_reaction=inherit)

# Initial individuals are created by reinitialize_pop(); give them properties
# through the hook as well, by running the initialisation first:
sim.reinitialize_pop()
for cell in sim.reactant_list['Cell']:
    cell.generation = 0
    cell.size = 1.0

# run_simulations() would reinitialise the population (and lose the
# properties above), so run the single realisation directly:
sim.run_simulation()

generations = [cell.generation for cell in sim.reactant_list['Cell']]
sizes = [cell.size for cell in sim.reactant_list['Cell']]
print('%d cells alive, generation %d to %d, mean size %.3f'
      % (len(generations), min(generations), max(generations), np.mean(sizes)))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
bins = np.arange(0, max(generations + generations_at_death) + 2) - 0.5
ax.hist(generations, bins=bins, alpha=0.7, label='alive at Tend')
ax.hist(generations_at_death, bins=bins, alpha=0.7, label='died before Tend')
ax.set_xlabel('Generation')
ax.set_ylabel('Number of cells')
ax.legend()
plt.show()
