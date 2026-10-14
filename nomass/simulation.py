"""
Core of NoMaSS: reaction channels, reactants and the stochastic simulation
engine mixing the three non-Markovian Gillespie methods.

* MOSAIC  (rejection based, Pelissier et al. 2026): every reactant of a channel
  is proposed at the maximum rate ``rmax`` of the channel and accepted with
  probability ``hazard(t) / rmax``.
* Laplace Gillespie (Masuda & Rocha 2018): each reactant carries its own
  Poisson rate drawn from the mixing density of the inter-event time
  distribution; the simulation is then an ordinary Gillespie simulation.
* DelaySSA (Anderson 2007): the waiting time of each reactant is drawn when it
  enters the channel and the event is scheduled in a priority queue.

The three methods can be mixed freely in the same model, one method per
reaction channel.
"""

import heapq
import random
import warnings
from math import log, isfinite

import numpy as np
import matplotlib.pyplot as plt

from . import distributions as dist

__all__ = ['Reaction_channel', 'Reactant', 'Gillespie_simulation']

# seaborn "deep" palette, so that figures look like the ones of the paper
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
           '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

MAX_POPULATION = 1_000_000


class _counts:
    channel_numbers = 0


class Reaction_channel:
    """A reaction channel ``reactants -> products`` with a waiting time distribution.

    Parameters
    ----------
    param_simulation : object
        Simulation parameters (``Tend``, ``N_simulations``, ``timepoints``, ``unit``).
        Kept as first positional argument for backward compatibility; may be ``None``.
    rate : float
        Nominal rate: the inverse of the mean waiting time (median for Cauchy,
        Pareto and Lomax).
    shape_param : float
        Shape parameter of the distribution (see ``nomass.distributions``).
    distribution : str
        'Exponential', 'Gamma', 'Weibull', 'Gaussian', 'Lognormal', 'Cauchy',
        'Pareto' or 'Lomax' (aliases such as 'exp', 'norm', 'lognorm' are accepted).
    method : str or None
        'MOSAIC', 'Laplace' or 'Delay'. ``None`` selects MOSAIC when the
        hazard is bounded and Laplace Gillespie otherwise.
    name : str
        Label used in printouts and plots.
    reactants, products : list of str
        Species names. Non-Markovian waiting times are only defined for
        channels with exactly one reactant; other channels are Poisson processes.
    transfer_identity : bool
        If a product is also a reactant, keep the same individual (its clocks
        in the other channels are preserved) instead of creating a new one.
    """

    def __init__(self, param_simulation=None, rate=1.0, shape_param=1.0,
                 distribution='Exponential', method=None, name='',
                 reactants=None, products=None, transfer_identity=False):
        _counts.channel_numbers += 1
        name = name if name else 'Reaction %d' % _counts.channel_numbers

        if rate < 0:
            raise ValueError("Reaction '%s': the rate cannot be negative" % name)

        self.param = param_simulation
        self.reactants = list(reactants) if reactants else []
        self.products = list(products) if products else []
        self.rate = float(rate)
        self.shape_param = float(shape_param)
        self.name = name
        self.transfer_identity = transfer_identity

        self.distribution = dist.canonical_distribution(distribution)
        if rate == 0 or (self.distribution in ('Gamma', 'Weibull') and shape_param == 1):
            self.distribution = 'Exponential'
        if self.distribution != 'Exponential' and shape_param <= 0:
            raise ValueError("Reaction '%s': the shape parameter must be positive" % name)

        if method is None:
            self.method = dist.default_method(self.distribution, self.shape_param)
        else:
            self.method = dist.canonical_method(method)
            if not dist.method_supported(self.method, self.distribution, self.shape_param):
                raise ValueError(
                    "Reaction '%s': the %s method does not support a %s distribution "
                    "with shape parameter %g. Supported combinations are listed in the "
                    "README (MOSAIC needs a bounded hazard, Laplace needs a completely "
                    "monotone survival function; Delay supports every distribution)."
                    % (name, self.method, self.distribution, self.shape_param))

        self.rate_function = dist.hazard_function(self.distribution)
        self.rmax_fixed = not dist.hazard_is_increasing(self.distribution, self.shape_param)
        self.rmax = self.rate
        self.temp_rmax = self.rate

        # statistics
        self.wait_times = []
        self.number_of_rejected_reactions = 0
        self.number_of_accepted_reactions = 0

    # ----------------------------------------------------------------------
    def reset_statistics(self):
        self.wait_times = []
        self.number_of_rejected_reactions = 0
        self.number_of_accepted_reactions = 0

    def is_poisson(self):
        """True when the channel is simulated as a plain Poisson process."""
        return len(self.reactants) != 1 or self.distribution == 'Exponential'

    def hazard(self, t):
        """Instantaneous rate of one reactant that entered the channel ``t`` ago."""
        return dist.hazard(t, self.distribution, self.rate, self.shape_param)

    def sample_delay(self, rng=None, size=None):
        return dist.sample(self.distribution, self.rate, self.shape_param, size=size, rng=rng)

    def sample_laplace_rate(self, rng=None):
        return dist.laplace_rate(self.distribution, self.rate, self.shape_param, rng=rng)

    def _prepare(self, Tend):
        """Precompute the fixed rate bound used by MOSAIC."""
        if self.rmax_fixed:
            self.rmax = dist.hazard_upper_bound(self.distribution, self.rate, self.shape_param, Tend)
        else:
            self.rmax = self.rate
        self.temp_rmax = self.rmax

    def __repr__(self):
        return "Reaction_channel(%r)" % self.name

    def __str__(self):
        s = '\n    %s' % self.name
        s += '\n      reactants = %s' % self.reactants
        s += '\n      products = %s' % self.products
        s += '\n      rate = %s' % self.rate
        if self.distribution != 'Exponential':
            s += '\n      shape_parameter = %s' % self.shape_param
        s += '\n      distribution = %s' % self.distribution
        if not self.is_poisson():
            s += '\n      method = %s' % self.method
        if self.transfer_identity:
            s += '\n      transfer_identity = True'
        return s


class Reactant:
    """One individual of a species. Extend it to attach individual properties."""

    __slots__ = ('id',)

    def __init__(self, ID=None):
        self.id = ID

    def __repr__(self):
        return 'Reactant(%s)' % self.id


class Gillespie_simulation:
    """Non-Markovian stochastic simulation of a reaction network.

    Parameters
    ----------
    N_init : dict
        Initial population, species name -> count. Every species used by the
        reaction channels must appear here (with 0 if initially absent).
    param : object
        Simulation parameters: ``Tend`` (length), ``N_simulations`` (number of
        repetitions), ``timepoints`` (number of recorded time points) and
        ``unit`` (time unit, for plots).
    reaction_channel_list : list of Reaction_channel, optional
        Can also be set later through the ``reaction_channel_list`` attribute.
    min_ratio : float
        MOSAIC only: the total propensity is kept at least ``min_ratio`` times
        above every channel's ``rmax`` so that the rate bounds are refreshed often
        enough (see Pelissier et al. 2026).
    print_warnings : bool
        Print numerical warnings during the simulation.
    seed : int, optional
        Seed of the random number generators (both ``random`` and NumPy).
    """

    def __init__(self, N_init, param, reaction_channel_list=None, min_ratio=10,
                 print_warnings=False, seed=None):
        self.param = param
        self.param.min_ratio = min_ratio
        self.param.print_warnings = print_warnings
        self.reaction_channel_list = list(reaction_channel_list) if reaction_channel_list else []
        self.reactant_population_init = dict(N_init)
        self.update_function = None
        self.population_compiled = None
        self.Tend = getattr(param, 'Tend', None)
        self.seed = seed
        self.rng = None
        self._warned = set()

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def run_simulations(self, Tend=None, verbose=True, update_function=None, seed=None):
        """Run ``param.N_simulations`` simulations and return the recorded populations.

        Returns an array of shape ``(N_simulations, timepoints, n_species + 1)``;
        the last column is the total population. Species are in the order of
        ``N_init``.

        ``update_function(simulation)`` is called after every iteration and may
        change the ``rate`` or ``shape_param`` of any channel to implement
        state-dependent kinetics.
        """
        if Tend is None:
            Tend = self.param.Tend
        self.Tend = Tend
        self.update_function = update_function
        self._validate()

        if seed is None:
            seed = self.seed
        if seed is not None:
            random.seed(seed)
        self.rng = np.random.default_rng(random.getrandbits(64))

        for channel in self.reaction_channel_list:
            channel._prepare(Tend)
            channel.reset_statistics()

        N_sim = int(getattr(self.param, 'N_simulations', 1))
        timepoints = int(getattr(self.param, 'timepoints', 100))
        population = np.empty((N_sim, timepoints, len(self.reactant_population_init) + 1))
        for ni in range(N_sim):
            if verbose:
                print("   Simulation %d / %d ..." % (ni + 1, N_sim))
            self.reinitialize_pop()
            self.run_simulation(Tend)
            population[ni] = self.population_t

        self.population_compiled = population
        return population

    def get_populations(self):
        return self.population_t

    def get_reactant_list(self):
        return self.reactant_list

    def get_channel_waiting_times(self):
        return [channel.wait_times for channel in self.reaction_channel_list]

    def get_model_in_SBML(self):
        """Export the model to a ``simplesbml`` model (requires ``simplesbml``)."""
        from .sbml import model_to_sbml
        return model_to_sbml(self)

    # ----------------------------------------------------------------------
    # Validation and initialisation
    # ----------------------------------------------------------------------
    def _warn(self, key, message):
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(message, stacklevel=3)

    def _validate(self):
        if not self.reaction_channel_list:
            raise ValueError('No reaction channel defined: set reaction_channel_list first')
        names = set()
        for channel in self.reaction_channel_list:
            if channel.name in names:
                raise ValueError("Two reaction channels share the name '%s'" % channel.name)
            names.add(channel.name)
            for species in channel.reactants + channel.products:
                if species not in self.reactant_population_init:
                    raise ValueError(
                        "Species '%s' used by reaction '%s' is missing from N_init "
                        "(add it with an initial population of 0)" % (species, channel.name))
            if channel.transfer_identity and len(set(channel.products)) < len(channel.products):
                raise ValueError(
                    "Reaction '%s': transfer_identity=True is ambiguous when a product "
                    "appears several times" % channel.name)
            if len(channel.reactants) != 1 and channel.distribution != 'Exponential':
                self._warn(('poisson', channel.name),
                           "Reaction '%s' has %d reactants: non-Markovian waiting times are "
                           "only defined for single-reactant channels, it is simulated as a "
                           "Poisson process." % (channel.name, len(channel.reactants)))

    def reinitialize_pop(self):
        """Reset the populations to the initial condition (statistics are kept)."""
        self.t = 0.0
        self._next_id = 0
        self._seq = 0
        channels = self.reaction_channel_list
        self.channel_index = {ch.name: ci for ci, ch in enumerate(channels)}

        # channels that depend on the individual clocks of a given species
        self.channels_of = {name: [] for name in self.reactant_population_init}
        for ch in channels:
            if len(ch.reactants) == 1:
                self.channels_of[ch.reactants[0]].append(ch)

        self.reactant_population = dict(self.reactant_population_init)
        self.reactant_list = {name: [] for name in self.reactant_population_init}
        self._index = {name: {} for name in self.reactant_population_init}
        self.reactant_times = {name: {ch.name: {} for ch in self.channels_of[name]}
                               for name in self.reactant_population_init}
        self._lam = {ch.name: {} for ch in channels if ch.method == 'Laplace'}
        self._lam_sum = {ch.name: 0.0 for ch in channels if ch.method == 'Laplace'}
        self._lam_params = {ch.name: (ch.rate, ch.shape_param) for ch in channels if ch.method == 'Laplace'}
        self._delay_token = {ch.name: {} for ch in channels if ch.method == 'Delay'}
        self._delay_pool = {}
        self._heap = []

        for name, n in self.reactant_population_init.items():
            self.reactant_population[name] = 0
            for _ in range(int(n)):
                self._add_reactant(name)

    # ----------------------------------------------------------------------
    # Individual bookkeeping
    # ----------------------------------------------------------------------
    def _new_reactant(self):
        self._next_id += 1
        return Reactant(self._next_id)

    def _add_reactant(self, name, reactant=None):
        if reactant is None:
            reactant = self._new_reactant()
        lst = self.reactant_list[name]
        self._index[name][reactant.id] = len(lst)
        lst.append(reactant)
        self.reactant_population[name] += 1
        for ch in self.channels_of[name]:
            self._start_clock(ch, name, reactant.id)
        return reactant

    def _remove_reactant(self, name, reactant_id):
        lst = self.reactant_list[name]
        idx = self._index[name].pop(reactant_id)
        last = lst.pop()
        if idx < len(lst):
            lst[idx] = last
            self._index[name][last.id] = idx
        self.reactant_population[name] -= 1
        for ch in self.channels_of[name]:
            self._stop_clock(ch, name, reactant_id)

    def _start_clock(self, ch, name, reactant_id):
        """(Re)start the clock of one individual in one channel at the current time."""
        times = self.reactant_times[name][ch.name]
        times.pop(reactant_id, None)      # move to the end: dict order == age order
        times[reactant_id] = self.t
        if ch.method == 'Laplace' and ch.distribution != 'Exponential':
            lam = self._lam[ch.name]
            old = lam.pop(reactant_id, 0.0)
            new = ch.sample_laplace_rate(self.rng)
            lam[reactant_id] = new
            self._lam_sum[ch.name] += new - old
        elif ch.method == 'Laplace':
            lam = self._lam[ch.name]
            old = lam.pop(reactant_id, 0.0)
            lam[reactant_id] = ch.rate
            self._lam_sum[ch.name] += ch.rate - old
        elif ch.method == 'Delay':
            if ch.rate > 0:
                t_event = self.t + self._draw_delay(ch)
                self._delay_token[ch.name][reactant_id] = t_event
                self._seq += 1
                heapq.heappush(self._heap, (t_event, self._seq, self.channel_index[ch.name], reactant_id))
            else:
                self._delay_token[ch.name].pop(reactant_id, None)

    def _draw_delay(self, ch):
        """Waiting time for a Delay channel, drawn in batches for speed.

        The batch is discarded when ``rate`` or ``shape_param`` changed
        (e.g. through ``update_function``).
        """
        pool = self._delay_pool.get(ch.name)
        if pool is None or pool[0] != (ch.rate, ch.shape_param) or pool[2] >= len(pool[1]):
            n = max(64, min(4096, 2 * self.reactant_population.get(ch.reactants[0], 0)))
            pool = [(ch.rate, ch.shape_param), ch.sample_delay(self.rng, size=n), 0]
            self._delay_pool[ch.name] = pool
        value = float(pool[1][pool[2]])
        pool[2] += 1
        return value

    def _stop_clock(self, ch, name, reactant_id):
        self.reactant_times[name][ch.name].pop(reactant_id, None)
        if ch.method == 'Laplace':
            old = self._lam[ch.name].pop(reactant_id, None)
            if old is not None:
                self._lam_sum[ch.name] -= old
        elif ch.method == 'Delay':
            self._delay_token[ch.name].pop(reactant_id, None)

    def _next_delay(self):
        """Return the earliest valid scheduled delay (time, channel index, id) or None."""
        heap = self._heap
        while heap:
            t_event, _, ci, reactant_id = heap[0]
            ch = self.reaction_channel_list[ci]
            if self._delay_token[ch.name].get(reactant_id) == t_event:
                return t_event, ci, reactant_id
            heapq.heappop(heap)  # stale entry: the reactant was consumed or rescheduled
        return None

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------
    def run_simulation(self, Tend):
        """Run one realisation until ``Tend`` (call ``reinitialize_pop`` first)."""
        timepoints = int(getattr(self.param, 'timepoints', 100))
        n_species = len(self.reactant_population)
        self.Tend = Tend
        self.population_t = np.zeros((timepoints, n_species + 1))
        record_times = np.linspace(0.0, Tend, timepoints, endpoint=False)
        ti = 0
        update_function = self.update_function

        if update_function is not None:
            update_function(self)

        while self.t < Tend:
            while ti < timepoints and self.t >= record_times[ti]:
                self.population_t[ti, :-1] = list(self.reactant_population.values())
                ti += 1

            propensity = self.compute_propensities()
            a0 = float(np.sum(propensity))
            next_delay = self._next_delay()

            if a0 <= 0 and next_delay is None:
                break
            if sum(self.reactant_population.values()) > MAX_POPULATION:
                self._warn('explosion', 'More than %d reactants: the simulation is stopped early'
                           % MAX_POPULATION)
                break

            tau = log(1.0 / random.random()) / a0 if a0 > 0 else float('inf')

            if next_delay is not None and next_delay[0] <= self.t + tau:
                t_event, ci, reactant_id = next_delay
                heapq.heappop(self._heap)
                self.t = t_event
                if self.t >= Tend:
                    break
                self.perform_reaction(ci, reactant_id=reactant_id)
            else:
                self.t += tau
                if self.t >= Tend:
                    break
                r2 = random.random() * a0
                mu, p_sum = 0, propensity[0]
                while p_sum < r2:
                    mu += 1
                    p_sum += propensity[mu]
                self.perform_reaction(mu)

            if update_function is not None:
                update_function(self)

        # fill the remaining time points with the final population
        self.population_t[ti:, :-1] = list(self.reactant_population.values())
        self.population_t[:, -1] = np.sum(self.population_t[:, :-1], axis=1)
        self.t = Tend

    def compute_propensities(self):
        """Propensity of every channel at the current time (MOSAIC uses ``N * rmax``)."""
        channels = self.reaction_channel_list
        propensity = np.zeros(len(channels))
        for ci, ch in enumerate(channels):
            if len(ch.reactants) != 1:
                n = 1.0
                for species in ch.reactants:
                    n *= self.reactant_population[species]
                ch.rmax = ch.temp_rmax = ch.rate
                propensity[ci] = n * ch.rate
            elif ch.method == 'Delay':
                propensity[ci] = 0.0  # handled by the event queue
            elif ch.distribution == 'Exponential' or ch.method == 'MOSAIC':
                self._update_rmax(ch)
                propensity[ci] = self.reactant_population[ch.reactants[0]] * ch.rmax
            else:  # Laplace
                self._refresh_laplace(ch)
                propensity[ci] = max(self._lam_sum[ch.name], 0.0)

        # keep the total propensity well above every rmax so that the bounds
        # of increasing hazards are refreshed often enough (MOSAIC only)
        a0 = float(np.sum(propensity))
        min_ratio = self.param.min_ratio
        for ci, ch in enumerate(channels):
            if propensity[ci] > 0 and ch.method == 'MOSAIC' and not ch.is_poisson():
                ratio = a0 / ch.rmax
                if ratio < min_ratio:
                    ch.temp_rmax = ch.rmax * min_ratio / ratio
                    propensity[ci] = self.reactant_population[ch.reactants[0]] * ch.temp_rmax
        return propensity

    def _update_rmax(self, ch):
        """Upper bound of the hazard over the current population of the channel."""
        if ch.distribution == 'Exponential':
            ch.rmax = ch.temp_rmax = ch.rate
            return
        if ch.rmax_fixed:
            if ch.rate != getattr(ch, '_rmax_rate', None):
                ch._prepare(self.Tend)
                ch._rmax_rate = ch.rate
            ch.temp_rmax = ch.rmax
            return
        species = ch.reactants[0]
        times = self.reactant_times[species][ch.name]
        rmax = ch.rate
        if times:
            t_oldest = self.t - next(iter(times.values()))
            h = ch.rate_function(t_oldest, ch.rate, ch.shape_param)
            if not isfinite(h):
                if self.param.print_warnings:
                    print('Warning: hazard is %s at t = %.2e for reaction %s' % (h, t_oldest, ch.name))
                h = 5 * ch.rate
            rmax = max(rmax, h)
        ch.rmax = ch.temp_rmax = rmax

    def _refresh_laplace(self, ch):
        """Keep the drawn Poisson rates consistent with a rate changed by ``update_function``."""
        old_rate, old_shape = self._lam_params[ch.name]
        if ch.rate == old_rate and ch.shape_param == old_shape:
            return
        lam = self._lam[ch.name]
        if ch.shape_param == old_shape and old_rate > 0:
            factor = ch.rate / old_rate
            for key in lam:
                lam[key] *= factor
            self._lam_sum[ch.name] *= factor
        else:
            for key in lam:
                lam[key] = ch.sample_laplace_rate(self.rng)
            self._lam_sum[ch.name] = float(sum(lam.values()))
        self._lam_params[ch.name] = (ch.rate, ch.shape_param)

    # ----------------------------------------------------------------------
    # Reactions
    # ----------------------------------------------------------------------
    def perform_reaction(self, mu, reactant_id=None):
        """Attempt reaction ``mu``; returns True if it was accepted.

        ``reactant_id`` is given for scheduled (Delay) events, which are always accepted.
        """
        ch = self.reaction_channel_list[mu]
        Dt = 0.0
        chosen = {}

        if len(ch.reactants) != 1:
            # Poisson process: propensity is exact, always accept
            for species in ch.reactants:
                if species in chosen:
                    continue
                lst = self.reactant_list[species]
                if not lst:
                    return False
                chosen[species] = lst[random.randrange(len(lst))].id
        else:
            species = ch.reactants[0]
            times = self.reactant_times[species][ch.name]
            if reactant_id is not None:
                if reactant_id not in times:
                    return False
                Dt = self.t - times[reactant_id]
            elif ch.method == 'Laplace' and ch.distribution != 'Exponential':
                lam = self._lam[ch.name]
                ids = list(lam.keys())
                reactant_id = random.choices(ids, weights=list(lam.values()))[0]
                Dt = self.t - times[reactant_id]
            else:
                lst = self.reactant_list[species]
                reactant_id = lst[random.randrange(len(lst))].id
                Dt = self.t - times[reactant_id]
                if ch.distribution != 'Exponential':
                    h = ch.rate_function(Dt, ch.rate, ch.shape_param)
                    if not isfinite(h):
                        if self.param.print_warnings:
                            print('Warning: hazard is %s at t = %.2e for reaction %s' % (h, Dt, ch.name))
                        h = ch.temp_rmax
                    if h < ch.temp_rmax * random.random():
                        ch.number_of_rejected_reactions += 1
                        return False
                elif ch.rate < ch.temp_rmax * random.random():
                    ch.number_of_rejected_reactions += 1
                    return False
            chosen[species] = reactant_id

        # ---- accepted: update populations
        ch.number_of_accepted_reactions += 1
        if len(ch.reactants) == 1:
            ch.wait_times.append(Dt)

        kept = {}
        for species in ch.reactants:
            if ch.transfer_identity and species in ch.products and species not in kept:
                kept[species] = chosen[species]
            else:
                self._remove_reactant(species, chosen[species])

        for species in ch.products:
            if species in kept:
                # same individual: only the clock of this channel restarts
                self._start_clock(ch, species, kept.pop(species))
            else:
                self._add_reactant(species)
        return True

    # ----------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------
    def _unit_label(self):
        unit = getattr(self.param, 'unit', '')
        return ' [%s]' % unit if unit else ''

    def plot_populations_single(self, show=True):
        """Plot the species populations of the last realisation."""
        timepoints = self.population_t.shape[0]
        time_points = np.linspace(0, self.Tend, timepoints)
        plt.figure(figsize=(7, 4))
        for ri, reactant in enumerate(self.reactant_population.keys()):
            plt.plot(time_points, self.population_t[:, ri], lw=2, color=PALETTE[ri % 10], label=reactant)
        plt.xlabel('Time%s' % self._unit_label())
        plt.ylabel('Population')
        plt.legend()
        if show:
            plt.show()

    def plot_populations(self, reactant_list=None, log_scale=False, color_list=None,
                         figsize=(6.2, 4.2), show=True):
        """Plot every realisation (thin lines) and the mean (thick line) of each species."""
        if self.population_compiled is None:
            raise RuntimeError('Run run_simulations() before plotting')
        species = list(self.reactant_population_init.keys()) + ['Total']
        if reactant_list is None:
            reactant_list = species
        color_list = list(color_list) if color_list else []
        while len(color_list) < len(reactant_list):
            color_list.append(PALETTE[(len(color_list) + 1) % 10])

        population = self.population_compiled
        time_points = np.linspace(0, self.Tend, population.shape[1])
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': 16})
        k = 0
        for ri, name in enumerate(species):
            if name not in reactant_list:
                continue
            for i in range(population.shape[0]):
                plt.plot(time_points, population[i, :, ri], lw=0.3, alpha=0.1, color=PALETTE[0])
            plt.plot(time_points, population[:, :, ri].mean(axis=0), lw=3, color=color_list[k], label=name)
            k += 1
        plt.xlabel('Time%s' % self._unit_label())
        plt.ylabel('Population')
        plt.legend()
        if log_scale:
            plt.yscale('log')
        if show:
            plt.show()

    def plot_inter_event_time_distribution(self, color_list=None, plot_fitted=True, plot_theory=True,
                                           theory_color='green', fitted_color='red', bins=None,
                                           figsize=(6.2, 4.2), fontsize=16, lw=2, show=True, verbose=True):
        """Histogram of the simulated waiting times of each channel against theory.

        The theoretical curve is the distribution the channel would have if it
        were the only process acting on its reactant; competing channels on
        the same species make the simulated distribution differ, which is expected.
        """
        N_sim = max(int(getattr(self.param, 'N_simulations', 1)), 1)
        for ri, ch in enumerate(self.reaction_channel_list):
            wait_times = np.asarray(ch.wait_times, dtype=float)
            if verbose:
                print("\n Reaction channel %s:" % ch.name)
            if len(wait_times) == 0:
                if verbose:
                    print('   This reaction never occurred')
                continue
            if len(ch.reactants) != 1:
                if verbose:
                    print('   This channel fired %.2f times per simulation on average.' % (len(wait_times) / N_sim))
                    print('   No waiting time distribution is defined for channels without a unique reactant.')
                continue

            color = PALETTE[(ri + 4) % 10] if color_list is None else color_list[ri]
            nbins = 30 if bins is None else bins
            plt.figure(figsize=figsize)
            plt.rcParams.update({'font.size': fontsize})
            plt.hist(wait_times, bins=nbins, color=color, density=True, edgecolor='black')
            plt.xlabel("Time before reaction%s" % self._unit_label())
            plt.ylabel("PDF")
            plt.title(ch.name)
            tmax = np.max(wait_times)
            t = np.linspace(tmax / 100000, tmax, 10000)

            fitted = None
            if plot_fitted and len(wait_times) > 2:
                try:
                    fitted, fitted_pdf = dist.fit(ch.distribution, wait_times)
                    plt.plot(t, fitted_pdf(t), fitted_color, lw=lw, label='Fitted')
                except Exception:  # fitting is cosmetic; never fail the plot
                    fitted = None
            if plot_theory:
                plt.plot(t, dist.pdf(t, ch.distribution, ch.rate, ch.shape_param),
                         theory_color, lw=lw, linestyle='--', label='Theory')
            plt.legend()
            plt.xlim(0, (tmax if isinstance(nbins, int) else np.max(nbins)) * 1.03)
            if show:
                plt.show()

            if verbose:
                unit = getattr(self.param, 'unit', '')
                print("   Method: %s" % ch.method)
                print("   Observed rate %.3f vs nominal %.3f" % (1 / np.mean(wait_times), ch.rate))
                print("   Mean waiting time %.2f%s vs %.2f%s expected" %
                      (np.mean(wait_times), unit, 1 / ch.rate, unit))
                print("   Std of waiting times %.2f%s" % (np.std(wait_times), unit))
                if fitted is not None:
                    print("   Fitted scipy parameters: %s" % ', '.join('%.3g' % p for p in fitted))
                print("   Accepted reactions = %d" % ch.number_of_accepted_reactions)
                if ch.method == 'MOSAIC':
                    print("   Rejected reactions = %d" % ch.number_of_rejected_reactions)
        if verbose:
            print()
            print('  Notes:')
            print('  - The theoretical curve assumes the channel is the only process acting on')
            print('    its reactant. Competing channels or products that are also reactants')
            print('    change the simulated distribution: this is expected, not an error.')
            print('  - Out of steady state, a growing population biases waiting times towards')
            print('    short values because new reactants are continuously created with Dt = 0.')
            print('  - Increase Tend or N_simulations if the histogram is too noisy.')

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "Gillespie_simulation(%d species, %d channels)" % (
            len(self.reactant_population_init), len(self.reaction_channel_list))

    def __str__(self):
        s = 'NoMaSS model:\n'
        s += '\n  Initial population:\n    %s\n' % self.reactant_population_init
        s += '\n  Reaction channels:'
        for ch in self.reaction_channel_list:
            s += '    %s\n' % ch
        return s
