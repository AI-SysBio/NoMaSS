"""
Import and export of NoMaSS models in the SBML format.

SBML only describes Markovian rate laws, so the non-Markovian information of a
channel (distribution, shape parameter, simulation method, identity transfer)
is stored in the ``local_params`` of the kinetic law of each reaction.

Requires the optional dependencies ``python-libsbml`` and ``simplesbml``::

    pip install python-libsbml simplesbml
"""

import copy
import re

import numpy as np

from . import distributions as dist
from .simulation import Reaction_channel, Gillespie_simulation

__all__ = ['model_to_sbml', 'model_from_sbml', 'DISTRIBUTION_INDEX', 'METHOD_INDEX']

# Stable integer codes written in the SBML files (do not reorder).
DISTRIBUTION_INDEX = ['Exponential', 'Weibull', 'Gaussian', 'Gamma', 'Lognormal', 'Cauchy', 'Pareto', 'Lomax']
METHOD_INDEX = ['MOSAIC', 'Laplace', 'Delay']


def _simplesbml():
    try:
        import simplesbml
    except ImportError as e:  # pragma: no cover
        raise ImportError('SBML support needs the optional dependencies: '
                          'pip install python-libsbml simplesbml') from e
    return simplesbml


def model_to_sbml(simulation):
    """Return a ``simplesbml.SbmlModel`` describing a ``Gillespie_simulation``.

    Rate laws are ``r_i * reactant_1 * reactant_2 ...`` (NoMaSS propensities are
    always proportional to the reactant counts). Write the file with
    ``model.toSBML()``.
    """
    simplesbml = _simplesbml()
    model = simplesbml.SbmlModel()
    model.addCompartment(1, comp_id='comp')
    for species, n_init in simulation.reactant_population_init.items():
        model.addSpecies(species, n_init, comp='comp')

    used_ids = set()
    for ri, channel in enumerate(simulation.reaction_channel_list):
        local_params = {
            'shape_param': channel.shape_param,
            'distribution_index': DISTRIBUTION_INDEX.index(channel.distribution),
            'method_index': METHOD_INDEX.index(channel.method),
            'transfer_identity': int(bool(channel.transfer_identity)),
        }
        rate_label = 'r%d' % ri
        model.addParameter(rate_label, channel.rate)
        rate_law = ' * '.join([rate_label] + list(channel.reactants))

        rxn_id = re.sub(r'[^A-Za-z0-9_]', '_', channel.name.replace('->', '_to_').replace('+', '_and_'))
        rxn_id = re.sub(r'_+', '_', rxn_id).strip('_')
        if not rxn_id or rxn_id[0].isdigit() or rxn_id in used_ids:
            rxn_id = 'reaction_%d' % ri
        used_ids.add(rxn_id)
        model.addReaction(list(channel.reactants), list(channel.products), rate_law,
                          local_params=local_params, rxn_id=rxn_id)
    return model


def model_from_sbml(sbml_file, param, verbose=False):
    """Build a ``Gillespie_simulation`` from an SBML file.

    Mainly meant for files written by ``model_to_sbml``. Generic SBML models are
    imported on a best-effort basis: only rate laws proportional to the reactant
    amounts are supported, species given as concentrations are rescaled to
    integer amounts, and a warning is printed when a rate law cannot be mapped.
    """
    simplesbml = _simplesbml()
    if verbose:
        print('Reading SBML model from %s ...\n' % sbml_file)
    model = simplesbml.loadSBMLFile(sbml_file)

    species = sorted(model.getListOfFloatingSpecies() + model.getListOfBoundarySpecies(), key=len)[::-1]
    reactions = sorted(model.getListOfReactionIds(), key=len)[::-1]
    compartments = sorted(model.getListOfCompartmentIds(), key=len)[::-1]
    functions = sorted(model.getListOfFunctionIds(), key=len)[::-1]
    parameters = sorted(model.getListOfParameterIds(), key=len)[::-1]

    # ---- initial populations
    N_init = {}
    concentration = np.zeros(len(species))
    amount = np.zeros(len(species))
    for si, entity in enumerate(species):
        if model.isAmount(entity):
            amount[si] = int(model.getSpeciesInitialAmount(entity))
        elif model.isConcentration(entity):
            concentration[si] = model.getSpeciesInitialConcentration(entity)
        else:
            N_init[entity] = 0
    if np.sum(concentration) > 0:
        concentration = concentration / np.min(concentration[concentration > 0]) * 10
        for si, entity in enumerate(species):
            if model.isConcentration(entity):
                N_init[entity] = int(concentration[si])
    if np.sum(amount) > 0:
        if np.max(amount) > 100000:
            print(' Some species have very large initial amounts; they are rescaled so that '
                  'the smallest non-zero amount is 10.')
            amount = amount / np.min(amount[amount > 0]) * 10
        for si, entity in enumerate(species):
            if model.isAmount(entity):
                N_init[entity] = int(amount[si])
    if verbose:
        print('  Initial population:', N_init)

    # ---- reaction channels
    reaction_channel_list = []
    for ri, reaction_id in enumerate(reactions):
        reactants = [model.getReactant(reaction_id, k) for k in range(model.getNumReactants(reaction_id))]
        products = [model.getProduct(reaction_id, k) for k in range(model.getNumProducts(reaction_id))]
        if verbose:
            print('\n  Reaction %s: %s -> %s' % (reaction_id, reactants, products))

        local_params = _local_params(model, ri)
        rate = _rate_from_law(model.getRateLaw(reaction_id), model, species, functions, compartments,
                              parameters, local_params, len(reactants), reaction_id, verbose)

        shape_param = local_params.get('shape_param', 1)
        distribution = DISTRIBUTION_INDEX[int(local_params.get('distribution_index', 0))]
        method = METHOD_INDEX[int(local_params['method_index'])] if 'method_index' in local_params else None
        transfer_identity = bool(local_params.get('transfer_identity', False))

        channel = Reaction_channel(param, rate=rate, shape_param=shape_param, distribution=distribution,
                                   method=method, reactants=reactants, products=products,
                                   name=reaction_id, transfer_identity=transfer_identity)
        reaction_channel_list.append(channel)

    for channel in reaction_channel_list:
        for s in channel.reactants + channel.products:
            N_init.setdefault(s, 0)

    simulation = Gillespie_simulation(N_init, param, reaction_channel_list=reaction_channel_list)
    if verbose:
        print('\nImported', simulation)
    return simulation


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _local_params(model, ri):
    reaction = model.getModel().getListOfReactions()[ri]
    return {p.getId(): p.getValue() for p in reaction.getKineticLaw().getListOfParameters()}


def _expand_functions(rate_law, func, body, arguments):
    full_call = func + rate_law.split(func)[1].split(')')[0] + ')'
    call_args = rate_law.split(func)[1].split(')')[0].replace('(', '').replace(' ', '').split(',')
    for formal, actual in zip(arguments, call_args):
        body = body.replace(formal, actual)
    return rate_law.replace(full_call, body)


def _rate_from_law(rate_law, model, species, functions, compartments, parameters,
                   local_params, n_reactants, reaction_id, verbose):
    for func in functions:
        if func in rate_law:
            rate_law = _expand_functions(rate_law, func, model.getFunctionBody(func),
                                         model.getListOfArgumentsInUserFunction(func))
    if verbose:
        print('   rate law =', rate_law)

    for comp in compartments:
        rate_law = rate_law.replace(comp, '1')  # compartment sizes are taken as 1

    # count species appearing in the rate law
    probe = copy.copy(rate_law)
    n_law = 0
    for entity in species:
        if entity in probe:
            probe = probe.replace(entity, '')
            n_law += 1
    if n_law != n_reactants:
        print('\n   WARNING: NoMaSS only supports rate laws proportional to the reactant amounts.')
        print('            The imported model will NOT correspond to the original SBML model')
        print('            for reaction %s (rate law = %s)\n' % (reaction_id, rate_law))

    for entity in species:
        rate_law = rate_law.replace(entity, '1')
    for pid in sorted(local_params, key=len)[::-1]:
        rate_law = rate_law.replace(pid, str(local_params[pid]))
    for pid in parameters:
        rate_law = rate_law.replace(pid, str(model.getParameterValue(pid)))

    try:
        rate = float(eval(rate_law, {'__builtins__': {}}, {}))
    except Exception as e:
        raise ValueError('Could not evaluate the rate law "%s" of reaction %s: %s' % (rate_law, reaction_id, e))
    if verbose:
        print('   rate = %s' % rate)
    return rate
