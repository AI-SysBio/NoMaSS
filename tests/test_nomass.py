"""
Tests of the NoMaSS engine.

Statistical tests compare the waiting times produced by each method with the
theoretical distribution (Kolmogorov-Smirnov). Seeds are fixed, so failures
are deterministic.
"""
import os
import random
import runpy
import warnings

import numpy as np
import pytest
import scipy.stats as st

import nomass
from nomass import distributions as D
from nomass.simulation import _WeightedPool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    ('Exponential', 1.0), ('Gamma', 4.0), ('Gamma', 0.5), ('Weibull', 2.5), ('Weibull', 0.6),
    ('Gaussian', 0.2), ('Lognormal', 0.3), ('Lognormal', 0.1), ('Cauchy', 0.1),
    ('Pareto', 3.0), ('Lomax', 3.0),
]
COMBOS = [(d, a, m) for d, a in CASES for m in nomass.METHODS if D.method_supported(m, d, a)]


@pytest.mark.parametrize('distribution,shape,method', COMBOS)
def test_waiting_times_follow_the_theoretical_distribution(distribution, shape, method):
    channel = nomass.Reaction_channel('A -> B', rate=1.0, shape_param=shape, distribution=distribution,
                                      method=method)
    sim = nomass.Gillespie_simulation({'A': 2000, 'B': 0}, [channel], Tend=30, timepoints=50, seed=1)
    sim.run_simulations(verbose=False)

    waits = np.asarray(channel.wait_times)
    assert len(waits) >= 1900, 'almost every reactant should have reacted before Tend'
    frozen = D.scipy_distribution(distribution, 1.0, shape)
    pvalue = st.kstest(waits, lambda x: 1.0 - frozen.sf(x)).pvalue
    assert pvalue > 1e-3, 'KS test rejects the theoretical distribution (p=%.2e)' % pvalue


def test_methods_can_be_mixed_in_one_model():
    r1 = nomass.Reaction_channel('A -> B', rate=1, shape_param=20, distribution='Gamma', method='Delay')
    r2 = nomass.Reaction_channel('B -> C + A', rate=4, shape_param=5, distribution='Weibull', method='MOSAIC')
    r3 = nomass.Reaction_channel('C -> 0', rate=0.03, shape_param=0.5, distribution='Gamma', method='Laplace')
    r4 = nomass.Reaction_channel('A + B -> 0', rate=0.03)
    sim = nomass.Gillespie_simulation({'A': 300, 'B': 0, 'C': 0}, [r1, r2, r3, r4],
                                      Tend=10, N_simulations=3, timepoints=100, unit='h', seed=3)
    pop = sim.run_simulations(verbose=False)

    assert pop.shape == (3, 100, 4)
    assert np.all(pop[:, :, -1] == pop[:, :, :-1].sum(axis=2))
    assert np.all(pop[:, 0, 0] == 300)
    for ch in (r1, r2, r3, r4):
        assert ch.number_of_accepted_reactions > 0
    assert r2.number_of_rejected_reactions > 0  # MOSAIC rejects
    assert r1.number_of_rejected_reactions == 0  # Delay never rejects
    assert abs(np.mean(r1.wait_times) - 1.0) < 0.05  # A -> B is only consumed by r1 and r4

    # plots work headless and return their figures
    fig, ax = sim.plot_populations(show=False)
    assert ax.get_ylabel() == 'Population'
    fig, ax = sim.plot_populations_single(show=False)
    figures = sim.plot_inter_event_time_distribution(show=False, verbose=False)
    assert len(figures) == 3  # the Poisson channel has no waiting-time distribution


def test_reaction_strings():
    assert nomass.parse_reaction('A + B -> C') == (['A', 'B'], ['C'])
    assert nomass.parse_reaction('0 -> A') == ([], ['A'])
    assert nomass.parse_reaction('A -> 0') == (['A'], [])
    assert nomass.parse_reaction(' -> A') == ([], ['A'])
    assert nomass.parse_reaction('2 A -> B') == (['A', 'A'], ['B'])
    assert nomass.parse_reaction('Gon -> Gon + mRNA') == (['Gon'], ['Gon', 'mRNA'])
    ch = nomass.Reaction_channel('A + B -> C', rate=2)
    assert ch.reactants == ['A', 'B'] and ch.products == ['C'] and ch.name == 'A + B -> C'
    ch = nomass.Reaction_channel(reactants=['A'], products=[], rate=1)
    assert ch.name == 'A -> 0'
    with pytest.raises(ValueError, match='exactly one'):
        nomass.parse_reaction('A + B')
    with pytest.raises(ValueError, match='either'):
        nomass.Reaction_channel('A -> B', reactants=['A'])


def test_legacy_param_object_is_still_accepted():
    class param:
        Tend = 5
        N_simulations = 2
        timepoints = 20
        unit = 'h'

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ch = nomass.Reaction_channel(param, rate=1, name='A -> B')
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
    ch.reactants, ch.products = ['A'], ['B']
    sim = nomass.Gillespie_simulation({'A': 50, 'B': 0}, param)
    sim.reaction_channel_list = [ch]
    assert sim.Tend == 5 and sim.N_simulations == 2 and sim.unit == 'h' and sim.param.timepoints == 20
    pop = sim.run_simulations(param.Tend, verbose=False)
    assert pop.shape == (2, 20, 3)


def test_update_function_implements_state_dependent_rates():
    K = 200

    def update(sim):
        N = sim.reactant_population['A']
        sim.reaction_channel_list[0].rate = max(1.0 * (1 - N / K), 1e-9)

    for method, shape in (('MOSAIC', 5), ('Laplace', 0.5)):
        division = nomass.Reaction_channel('A -> A + A', rate=1, shape_param=shape, distribution='Gamma',
                                           method=method)
        sim = nomass.Gillespie_simulation({'A': 10}, [division], Tend=25, N_simulations=2, timepoints=50,
                                          seed=5, update_function=update)
        pop = sim.run_simulations(verbose=False)
        assert np.all(np.abs(pop[:, -1, 0] - K) <= 10), method


def test_rates_changed_by_update_function_are_reset_for_each_realisation():
    # the initial 100 delays of every realisation must be drawn at the initial rate,
    # not at the rate left behind by update_function in the previous realisation
    r = 1 / 71.6

    def halve(sim):
        sim.reaction_channel_list[0].rate = r / 2

    ch = nomass.Reaction_channel('S -> S + Q', rate=r, distribution='Exponential', method='Delay')
    sim = nomass.Gillespie_simulation({'S': 100, 'Q': 0}, [ch], Tend=600, N_simulations=4,
                                      update_function=halve, seed=2)
    pop = sim.run_simulations(verbose=False)
    # renewal process: first interval mean 71.6 h, later ones 143.2 h -> 469 events in 600 h
    assert np.all(np.abs(pop[:, -1, 1] - 469) < 60), pop[:, -1, 1]
    assert ch.rate == r  # restored after the last realisation

    # Laplace: the initial Poisson rates are drawn in the same place; realisations must agree
    ch = nomass.Reaction_channel('S -> S + Q', rate=r, shape_param=0.5, distribution='Gamma', method='Laplace')
    sim = nomass.Gillespie_simulation({'S': 100, 'Q': 0}, [ch], Tend=600, N_simulations=4,
                                      update_function=halve, seed=2)
    pop = sim.run_simulations(verbose=False)
    final = pop[:, -1, 1]
    assert np.std(final) < 0.15 * np.mean(final), final
    assert ch.rate == r


def test_on_reaction_hook_receives_individuals():
    division = nomass.Reaction_channel('Cell -> Cell + Cell', rate=1, shape_param=5, distribution='Gamma')
    death = nomass.Reaction_channel('Cell -> 0', rate=0.2)
    seen = {'div': 0, 'death': 0}

    def hook(sim, channel, consumed, produced):
        if channel is division:
            assert len(consumed) == 1 and len(produced) == 2
            assert isinstance(consumed[0], nomass.Reactant)
            assert consumed[0] not in sim.reactant_list['Cell']       # the mother is gone
            assert all(p in sim.reactant_list['Cell'] for p in produced)
            for daughter in produced:
                daughter.generation = consumed[0].generation + 1
            seen['div'] += 1
        else:
            assert len(consumed) == 1 and produced == []
            seen['death'] += 1

    sim = nomass.Gillespie_simulation({'Cell': 20}, [division, death], Tend=4, seed=11, on_reaction=hook)
    sim.reinitialize_pop()
    for cell in sim.reactant_list['Cell']:
        cell.generation = 0
    sim.run_simulation()
    assert seen['div'] > 0 and seen['death'] > 0
    assert seen['div'] == division.number_of_accepted_reactions
    assert max(c.generation for c in sim.reactant_list['Cell']) >= 2


def test_transfer_identity_keeps_the_clocks_of_other_channels():
    on = nomass.Reaction_channel('Goff -> Gon', rate=1, shape_param=5, distribution='Gamma')
    off = nomass.Reaction_channel('Gon -> Goff', rate=1, shape_param=5, distribution='Gamma')
    prod = nomass.Reaction_channel('Gon -> Gon + mRNA', rate=10, transfer_identity=True)
    deg = nomass.Reaction_channel('mRNA -> 0', rate=1)
    kept = []

    def hook(sim, channel, consumed, produced):
        if channel is prod:
            kept.append(consumed[0] is produced[0])

    sim = nomass.Gillespie_simulation({'Goff': 200, 'Gon': 0, 'mRNA': 0}, [on, off, prod, deg],
                                      Tend=50, timepoints=50, unit='min', seed=7, on_reaction=hook)
    pop = sim.run_simulations(verbose=False)

    # if the Gon clock were reset at every transcription event (10x faster than
    # deactivation), the deactivation waiting time would be much shorter than 1
    assert abs(np.mean(off.wait_times) - 1.0) < 0.1
    assert pop[0, -1, 0] + pop[0, -1, 1] == 200  # genes are conserved
    assert kept and all(kept)                     # same individual before and after


def test_weighted_pool_samples_proportionally_to_weights():
    random.seed(0)
    pool = _WeightedPool()
    for key, w in [('a', 1.0), ('b', 2.0), ('c', 3.0), ('d', 4.0)]:
        pool.add(key, w)
    pool.remove('b')
    pool.set('c', 6.0)        # weights now a:1, c:6, d:4
    pool.add('e', 9.0)        # reuses b's slot -> a:1, c:6, d:4, e:9
    pool.rescale(2.0)
    assert abs(pool.total() - 40.0) < 1e-9
    assert len(pool) == 4 and 'b' not in pool
    counts = {k: 0 for k in 'acde'}
    n = 40000
    for _ in range(n):
        counts[pool.sample()] += 1
    for key, expected in [('a', 1), ('c', 6), ('d', 4), ('e', 9)]:
        assert abs(counts[key] / n - expected / 20) < 0.01, key


def test_default_method_selection():
    assert nomass.Reaction_channel('A -> B', distribution='Gamma', shape_param=5).method == 'MOSAIC'
    assert nomass.Reaction_channel('A -> B', distribution='Gamma', shape_param=0.5).method == 'Laplace'
    assert nomass.Reaction_channel('A -> B', distribution='Weibull', shape_param=0.5).method == 'Laplace'
    assert nomass.Reaction_channel('A -> B', distribution='Lognormal', shape_param=0.5).method == 'MOSAIC'
    assert nomass.Reaction_channel('A -> B', distribution='weib', shape_param=1).distribution == 'Exponential'


def test_invalid_models_raise_clear_errors():
    with pytest.raises(ValueError, match='Laplace'):
        nomass.Reaction_channel('A -> B', distribution='Gaussian', shape_param=0.2, method='Laplace')
    with pytest.raises(ValueError, match='MOSAIC'):
        nomass.Reaction_channel('A -> B', distribution='Gamma', shape_param=0.5, method='MOSAIC')
    with pytest.raises(ValueError, match='Unsupported distribution'):
        nomass.Reaction_channel('A -> B', distribution='foo')
    with pytest.raises(ValueError, match='negative'):
        nomass.Reaction_channel('A -> B', rate=-1)
    with pytest.raises(ValueError, match="missing from N_init"):
        nomass.Gillespie_simulation({'A': 1}, [nomass.Reaction_channel('A -> Z')]).run_simulations(verbose=False)
    with pytest.raises(ValueError, match='No reaction channel'):
        nomass.Gillespie_simulation({'A': 1}).run_simulations(verbose=False)
    with pytest.raises(TypeError, match='Unexpected keyword'):
        nomass.Gillespie_simulation({'A': 1}, Tend=1, foo=2)


def test_laplace_mixing_densities_reproduce_the_survival_functions():
    rng = np.random.default_rng(0)
    for distribution, rate, shape in [('Gamma', 1.0, 0.5), ('Weibull', 2.0, 0.6), ('Lomax', 1.0, 2.5)]:
        lam = np.array([D.laplace_rate(distribution, rate, shape, rng) for _ in range(50000)])
        tau = rng.exponential(1.0 / lam)
        frozen = D.scipy_distribution(distribution, rate, shape)
        pvalue = st.kstest(tau, lambda x: 1.0 - frozen.sf(x)).pvalue
        assert pvalue > 1e-3, distribution


def test_hazards_match_pdf_over_survival():
    for distribution, shape in CASES:
        frozen = D.scipy_distribution(distribution, 2.0, shape)
        for t in (0.2, 0.5, 1.0, 2.0):
            ref = float(frozen.pdf(t) / frozen.sf(t))
            if ref > 0:
                assert abs(D.hazard(t, distribution, 2.0, shape) - ref) <= 1e-2 * ref + 1e-9


@pytest.mark.parametrize('script', ['toy_example.py', 'all_distributions.py', 'state_dependent_rates.py',
                                    'individual_properties.py', 'erlang_embedding.py'])
def test_examples_run(script, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *a, **k: plt.close('all'))
    runpy.run_path(os.path.join(ROOT, 'examples', script), run_name='__main__')
