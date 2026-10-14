# Changelog

## 1.0.0 (2026-10)

First release of NoMaSS on PyPI, accompanying the article in *Briefings in Bioinformatics* 27(5), bbag486 (2026).

* Three exact simulation methods selectable per reaction channel: MOSAIC (rejection), Laplace Gillespie and DelaySSA.
* Eight inter-event time distributions: Exponential, Gamma, Weibull, Gaussian, Lognormal, Cauchy, Pareto and Lomax, all parameterised by a nominal rate and a dimensionless shape.
* Reactions written as strings (`'A + B -> C'`), keyword simulation parameters, `update_function` for state-dependent rates and `on_reaction` for individual properties.
* Plotting methods return their matplotlib figures and axes.
* Scripts, data and cached results reproducing the figures of the paper in `paper/`.

Compared with the earlier REGIR code: the delay and Laplace back-ends are new, the Laplace selection uses a Fenwick tree, channel parameters changed by `update_function` are reset for every realisation, and the Lognormal waiting time is parameterised by its mean in every back-end.
