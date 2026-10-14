"""
Tests of the NoMaSS engine.

Statistical tests compare the waiting times produced by each method with the
theoretical distribution (Kolmogorov-Smirnov). Seeds are fixed, so failures
are deterministic.
"""
import os
import runpy

import numpy as np
import pytest
import scipy.stats as st

import nomass
from nomass import distributions as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Param:
    Tend = 30
    N_simulations = 1
    timepoints = 50
    unit = 'h'


CASES = [
    ('Exponential', 1.0), ('Gamma', 4.0), ('Gamma', 0.5), ('Weibull', 2.5), ('Weibull', 0.6),
    ('Gaussian', 0.2), ('Lognormal', 0.3), ('Lognormal', 0.1), ('Cauchy', 0.1),
    ('Pareto', 3.0), ('Lomax', 3.0),
]
COMBOS = [(d, a, m) for d, a in CASES for m in nomass.METHODS if D.method_supported(m, d, a)]


@pytest.mark.parametrize('distribution,shape,method', COMBOS)
def test_waiting_times_follow_the_theoretical_distribution(distribution, shape, method):
    channel = nomass.Reaction_channel(Param, rate=1.0, shape_param=shape, distribution=distribution,
                                      method=method, reactants=['A'], products=['B'], name='A -> B')
    sim = nomass.Gillespie_simulation({'A': 2000, 'B': 0}, Param, [channel], seed=1)
    sim.run_simulations(verbose=False)

    waits = np.asarray(channel.wait_times)
    assert len(waits) >= 1900, 'almost every reactant should have reacted before Tend'
    frozen = D.scipy_distribution(distribution, 1.0, shape)
    pvalue = st.kstest(waits, lambda x: 1.0 - frozen.sf(x)).pvalue
    assert pvalue > 1e-3, 'KS test rejects the theoretical distribution (p=%.2e)' % pvalue


def test_methods_can_be_mixed_in_one_model():
    class P:
        Tend = 10
        N_simulations = 3
        timepoints = 100
        unit = 'h'

    r1 = nomass.Reaction_channel(P, rate=1, shape_param=20, distribution='Gamma', method='Delay',
                                 reactants=['A'], products=['B'], name='A -> B')
    r2 = nomass.Reaction_channel(P, rate=4, shape_param=5, distribution='Weibull', method='MOSAIC',
                                 reactants=['B'], products=['C', 'A'], name='B -> C + A')
    r3 = nomass.Reaction_channel(P, rate=0.03, shape_param=0.5, distribution='Gamma', method='Laplace',
                                 reactants=['C'], products=[], name='C -> 0')
    r4 = nomass.Reaction_channel(P, rate=0.03, reactants=['A', 'B'], products=[], name='A + B -> 0')
    sim = nomass.Gillespie_simulation({'A': 300, 'B': 0, 'C': 0}, P, [r1, r2, r3, r4], seed=3)
    pop = sim.run_simulations(verbose=False)

    assert pop.shape == (3, 100, 4)
    assert np.all(pop[:, :, -1] == pop[:, :, :-1].sum(axis=2))
    assert np.all(pop[:, 0, 0] == 300)
    for ch in (r1, r2, r3, r4):
        assert ch.number_of_accepted_reactions > 0
    assert r2.number_of_rejected_reactions > 0  # MOSAIC rejects
    assert r1.number_of_rejected_reactions == 0  # Delay never rejects
    assert abs(np.mean(r1.wait_times) - 1.0) < 0.05  # A -> B is only consumed by r1 and r4

    # plotting works headless
    sim.plot_populations(show=False)
    sim.plot_populations_single(show=False)
    sim.plot_inter_event_time_distribution(show=False, verbose=False)


def test_update_function_implements_state_dependent_rates():
    class P:
        Tend = 25
        N_simulations = 2
        timepoints = 50
        unit = 'h'

    K = 200

    def update(sim):
        N = sim.reactant_population['A']
        sim.reaction_channel_list[0].rate = max(1.0 * (1 - N / K), 1e-9)

    for method, shape in (('MOSAIC', 5), ('Laplace', 0.5)):
        division = nomass.Reaction_channel(P, rate=1, shape_param=shape, distribution='Gamma', method=method,
                                           reactants=['A'], products=['A', 'A'], name='division')
        sim = nomass.Gillespie_simulation({'A': 10}, P, [division], seed=5)
        pop = sim.run_simulations(update_function=update, verbose=False)
        assert np.all(np.abs(pop[:, -1, 0] - K) <= 10), method


def test_transfer_identity_keeps_the_clocks_of_other_channels():
    class P:
        Tend = 50
        N_simulations = 1
        timepoints = 50
        unit = 'min'

    on = nomass.Reaction_channel(P, rate=1, shape_param=5, distribution='Gamma',
                                 reactants=['Goff'], products=['Gon'], name='on')
    off = nomass.Reaction_channel(P, rate=1, shape_param=5, distribution='Gamma',
                                  reactants=['Gon'], products=['Goff'], name='off')
    prod = nomass.Reaction_channel(P, rate=10, reactants=['Gon'], products=['Gon', 'mRNA'],
                                   transfer_identity=True, name='prod')
    deg = nomass.Reaction_channel(P, rate=1, reactants=['mRNA'], products=[], name='deg')
    sim = nomass.Gillespie_simulation({'Goff': 200, 'Gon': 0, 'mRNA': 0}, P, [on, off, prod, deg], seed=7)
    pop = sim.run_simulations(verbose=False)

    # if the Gon clock were reset at every transcription event (10x faster than
    # deactivation), the deactivation waiting time would be much shorter than 1
    assert abs(np.mean(off.wait_times) - 1.0) < 0.1
    assert pop[0, -1, 0] + pop[0, -1, 1] == 200  # genes are conserved


def test_default_method_selection():
    assert nomass.Reaction_channel(Param, distribution='Gamma', shape_param=5).method == 'MOSAIC'
    assert nomass.Reaction_channel(Param, distribution='Gamma', shape_param=0.5).method == 'Laplace'
    assert nomass.Reaction_channel(Param, distribution='Weibull', shape_param=0.5).method == 'Laplace'
    assert nomass.Reaction_channel(Param, distribution='Lognormal', shape_param=0.5).method == 'MOSAIC'
    assert nomass.Reaction_channel(Param, distribution='weib', shape_param=1).distribution == 'Exponential'


def test_invalid_models_raise_clear_errors():
    with pytest.raises(ValueError, match='Laplace'):
        nomass.Reaction_channel(Param, distribution='Gaussian', shape_param=0.2, method='Laplace')
    with pytest.raises(ValueError, match='MOSAIC'):
        nomass.Reaction_channel(Param, distribution='Gamma', shape_param=0.5, method='MOSAIC')
    with pytest.raises(ValueError, match='Unsupported distribution'):
        nomass.Reaction_channel(Param, distribution='foo')
    with pytest.raises(ValueError, match='negative'):
        nomass.Reaction_channel(Param, rate=-1)
    with pytest.raises(ValueError, match="missing from N_init"):
        ch = nomass.Reaction_channel(Param, reactants=['A'], products=['Z'])
        nomass.Gillespie_simulation({'A': 1}, Param, [ch]).run_simulations(verbose=False)
    with pytest.raises(ValueError, match='No reaction channel'):
        nomass.Gillespie_simulation({'A': 1}, Param).run_simulations(verbose=False)


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


@pytest.mark.parametrize('script', ['toy_example.py', 'all_distributions.py',
                                    'state_dependent_rates.py', 'creation_and_interaction.py'])
def test_examples_run(script, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *a, **k: plt.close('all'))
    runpy.run_path(os.path.join(ROOT, 'examples', script), run_name='__main__')
