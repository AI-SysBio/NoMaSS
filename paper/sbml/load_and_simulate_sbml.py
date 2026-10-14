"""
Load a model from an SBML file and simulate it with NoMaSS.

Requires the optional dependencies:  pip install python-libsbml simplesbml
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # so that `import nomass` works without installing it
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # file paths are relative to this script

from nomass.sbml import model_from_sbml


class params:
    Tend = 200           # length of the simulation
    N_simulations = 20
    unit = 'h'           # unit of time (used for plotting only)
    timepoints = 100     # number of time points to record (used for plotting only)


def main():
    sbml_file = 'stem_cell_model.sbml'
    sim = model_from_sbml(sbml_file, params, verbose=True)

    print('Running %s simulations for Tend = %s%s ...' % (params.N_simulations, params.Tend, params.unit))
    sim.run_simulations(params.Tend, verbose=False)
    sim.plot_inter_event_time_distribution()
    sim.plot_populations()


if __name__ == '__main__':
    main()
