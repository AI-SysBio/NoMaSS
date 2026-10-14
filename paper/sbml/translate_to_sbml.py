"""
Build the stem-cell differentiation model of the paper and write it as SBML.

    ESC -> EPI   differentiation (Gamma)
    EPI -> NPC   differentiation (Gamma)

Requires the optional dependencies:  pip install python-libsbml simplesbml
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # so that `import nomass` works without installing it
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # file paths are relative to this script

import nomass


class param:
    Tend = 170
    unit = 'h'
    N_simulations = 20
    timepoints = 100


def main():
    r_diffAB, alpha_diffAB = 0.0339, 27.60
    r_diffBC, alpha_diffBC = 0.0164, 39.29

    N_init = {'ESC': 100, 'EPI': 0, 'NPC': 0}

    EE = nomass.Reaction_channel(param, rate=r_diffAB, shape_param=alpha_diffAB, distribution='Gamma',
                                 reactants=['ESC'], products=['EPI'], name='Differentiation: ESC -> EPI')
    EN = nomass.Reaction_channel(param, rate=r_diffBC, shape_param=alpha_diffBC, distribution='Gamma',
                                 reactants=['EPI'], products=['NPC'], name='Differentiation: EPI -> NPC')

    sim = nomass.Gillespie_simulation(N_init, param, [EE, EN])
    print(sim)

    sbml_model = sim.get_model_in_SBML()
    print(sbml_model.toSBML())
    with open('stem_cell_model.sbml', 'w') as f:
        f.write(sbml_model.toSBML())


if __name__ == '__main__':
    main()
