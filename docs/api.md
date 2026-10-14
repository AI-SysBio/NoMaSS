# NoMaSS API reference

```python
import nomass
```

The package exposes three classes, one helper and the `distributions` module.
Every class and function also carries a docstring (`help(nomass.Gillespie_simulation)`).

## `Reaction_channel`

```python
nomass.Reaction_channel(reaction=None, rate=1.0, shape_param=1.0, distribution='Exponential',
                        method=None, name='', reactants=None, products=None, transfer_identity=False)
```

One reaction channel with its waiting-time distribution.

| Argument | Meaning |
| --- | --- |
| `reaction` | Reaction string such as `'A + B -> C'`, `'0 -> A'`, `'A -> 0'`, `'2 A -> B'`. Alternatively give `reactants` and `products` as lists of species names. |
| `rate` | Nominal rate λ: inverse of the mean waiting time (median for Cauchy, Pareto and Lomax). |
| `shape_param` | Dimensionless shape α (see the table below). Ignored for the exponential distribution. |
| `distribution` | `'Exponential'`, `'Gamma'`, `'Weibull'`, `'Gaussian'`, `'Lognormal'`, `'Cauchy'`, `'Pareto'` or `'Lomax'`; short aliases (`'exp'`, `'norm'`, `'lognorm'`, `'weib'`, ...) are accepted. |
| `method` | `'MOSAIC'`, `'Laplace'` or `'Delay'`. `None` picks MOSAIC when the hazard is bounded and Laplace Gillespie otherwise. |
| `name` | Label for printouts and plots, defaults to the reaction string. |
| `transfer_identity` | When a product is also a reactant, keep the same individual (and the clocks of its other channels) instead of creating a new one. |

Channels with zero or several reactants are Poisson processes with propensity `rate × product of the reactant counts`.

Attributes updated during a simulation: `wait_times` (list of waiting times of the accepted reactions), `number_of_accepted_reactions`, `number_of_rejected_reactions` (MOSAIC only). `rate` and `shape_param` can be changed from an `update_function`.

Shape parameter per distribution:

| Distribution | `rate` | `shape_param` | MOSAIC | Laplace | Delay |
| --- | --- | --- | :-: | :-: | :-: |
| Exponential | 1 / mean | unused | ✓ | ✓ | ✓ |
| Gamma | 1 / mean | shape α | α ≥ 1 | α ≤ 1 | ✓ |
| Weibull | 1 / mean | shape k | k ≥ 1 | k ≤ 1 | ✓ |
| Gaussian | 1 / mean | std / mean | ✓ | | ✓ |
| Lognormal | 1 / mean | std / mean | ✓ | | ✓ |
| Cauchy | 1 / median | scale / median | ✓ | | ✓ |
| Pareto (type I) | 1 / median | tail index α | ✓ | | ✓ |
| Lomax (Pareto II) | 1 / median | tail index α | ✓ | ✓ | ✓ |

An unsupported combination raises `ValueError` at construction.

## `Gillespie_simulation`

```python
nomass.Gillespie_simulation(N_init, reaction_channel_list=None, Tend=10.0, N_simulations=1,
                            timepoints=100, unit='', min_ratio=20, print_warnings=False, seed=None,
                            on_reaction=None, update_function=None, max_population=1_000_000)
```

| Argument | Meaning |
| --- | --- |
| `N_init` | `dict` species name → initial count. Every species used by a channel must be present (0 if absent initially). |
| `reaction_channel_list` | List of `Reaction_channel`; can also be set later through the attribute of the same name. |
| `Tend` | Length of each realisation. |
| `N_simulations` | Number of independent realisations. |
| `timepoints` | Number of recorded time points per realisation. |
| `unit` | Time unit, used in plot labels. |
| `min_ratio` | MOSAIC: the total propensity is kept at least `min_ratio` times above every channel's rate bound (accuracy criterion of the paper). |
| `seed` | Seed of the random number generators. |
| `on_reaction` | Hook `on_reaction(sim, channel, consumed, produced)` called after every accepted reaction with the `Reactant` objects involved. |
| `update_function` | Hook `update_function(sim)` called after every iteration; may change the `rate` or `shape_param` of any channel. |
| `max_population` | Stop a realisation once the total population exceeds this value. |

### Methods

`run_simulations(Tend=None, verbose=True, update_function=None, seed=None, on_reaction=None)`
: Runs `N_simulations` realisations and returns an array of shape `(N_simulations, timepoints, n_species + 1)`; species are in the order of `N_init` and the last column is the total. Also stored in `population_compiled`. Channel statistics (`wait_times`, counts) are reset at the start and accumulated over all realisations.

`reinitialize_pop()` and `run_simulation(Tend=None)`
: Lower-level pair to run a single realisation, for example to set properties on the initial individuals between the two calls (see `examples/individual_properties.py`).

`plot_populations(reactant_list=None, log_scale=False, color_list=None, figsize=(6.2, 4.2), show=True, ax=None)`
: Every realisation as thin lines and the mean as thick lines. Returns `(fig, ax)`.

`plot_populations_single(show=True, ax=None)`
: Populations of the last realisation. Returns `(fig, ax)`.

`plot_inter_event_time_distribution(..., show=True, verbose=True)`
: Histogram of the simulated waiting times of each single-reactant channel against the fitted and theoretical densities. Returns a list of `(fig, ax)`.

`compute_propensities()` and `perform_reaction(mu, reactant_id=None)`
: Internals meant to be overridden in a subclass for kinetic laws that are not proportional to the reactant counts.

### State during a realisation

| Attribute | Content |
| --- | --- |
| `t` | Current time. |
| `reactant_population` | `dict` species → current count. |
| `reactant_list` | `dict` species → list of `Reactant` objects currently alive. |
| `reactant_times` | `dict` species → channel name → `{id: time the clock started}`. |
| `reaction_channel_list` | The channels, whose `rate` and `shape_param` may be modified by `update_function`. |

## `Reactant`

```python
nomass.Reactant(ID=None, **properties)
```

One individual. Attributes can be attached freely (`cell.size = 2.0`), typically from an `on_reaction` hook; they travel with the object when `transfer_identity=True`.

## `parse_reaction`

```python
nomass.parse_reaction('A + B -> C')   # (['A', 'B'], ['C'])
```

## `nomass.distributions`

Functions used by the engine, available for analysis and plotting:

| Function | Purpose |
| --- | --- |
| `hazard(t, distribution, rate, shape)` | Instantaneous rate ψ(t)/S(t). |
| `pdf(t, distribution, rate, shape)` | Theoretical density of the waiting time. |
| `sample(distribution, rate, shape, size=None, rng=None)` | Random waiting times. |
| `laplace_rate(distribution, rate, shape, rng=None)` | A Poisson rate drawn from the Laplace mixing density (Exponential, Gamma α ≤ 1, Weibull α ≤ 1, Lomax). |
| `scipy_distribution(distribution, rate, shape)` | Frozen `scipy.stats` distribution with the NoMaSS parametrisation (Gaussian and Cauchy truncated to t ≥ 0). |
| `mean`, `fit`, `hazard_upper_bound`, `hazard_is_bounded`, `hazard_is_increasing`, `laplace_supported`, `default_method`, `method_supported` | Helpers behind the method selection and the MOSAIC rate bounds. |

Constants: `nomass.DISTRIBUTIONS`, `nomass.METHODS`, `nomass.PALETTE`, `nomass.__version__`.
