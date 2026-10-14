# Interacting with SBML files

The Systems Biology Markup Language (SBML) is an XML-based standard for computational models in systems biology. NoMaSS can read and write models in this format through the optional module `nomass.sbml`, which needs two extra libraries:

```bash
pip install python-libsbml simplesbml
```

## Saving a NoMaSS model as SBML

Use `get_model_in_SBML()` on any `Gillespie_simulation` (or `nomass.sbml.model_to_sbml(sim)`). Rate laws are written as `r * reactant_1 * reactant_2 ...`, since NoMaSS propensities are always proportional to the reactant counts. SBML only describes Markovian kinetics, so the distribution, shape parameter, simulation method and identity transfer of each channel are stored in the `local_params` of the reaction's kinetic law (`distribution_index`, `shape_param`, `method_index`, `transfer_identity`; see `nomass/sbml.py` for the index tables).

```python
import nomass

# define param, N_init and the reaction channels as usual
sim = nomass.Gillespie_simulation(N_init, param, reaction_channel_list)

sbml_model = sim.get_model_in_SBML()
with open("model.sbml", "w") as f:
    f.write(sbml_model.toSBML())
```

`translate_to_sbml.py` writes the stem-cell differentiation model of the paper this way (`stem_cell_model.sbml`).

## Loading an SBML model into NoMaSS

```python
from nomass.sbml import model_from_sbml

class param:
    Tend = 200
    N_simulations = 20
    unit = 'h'
    timepoints = 100

sim = model_from_sbml('your_model.sbml', param, verbose=True)
sim.run_simulations(verbose=False)
sim.plot_inter_event_time_distribution()
sim.plot_populations()
```

`load_and_simulate_sbml.py` does this for `stem_cell_model.sbml`.

The importer is mainly meant for files written by NoMaSS. Many models of the [BioModels database](https://www.ebi.ac.uk/biomodels/) can be loaded, but only rate laws proportional to the reactant amounts are supported; the importer prints a warning for every reaction whose rate law it cannot map. Species given as concentrations are rescaled to integer amounts, because NoMaSS tracks every individual explicitly.
