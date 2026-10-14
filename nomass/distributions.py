"""
Inter-event time distributions supported by NoMaSS.

Every distribution is parametrised by a nominal ``rate`` (the inverse of the
mean waiting time, or of the median when the mean is undefined) and a
dimensionless ``shape`` parameter. This module provides, for each distribution:

* ``hazard(t, rate, shape)``      instantaneous rate at time ``t`` since the
                                   reactant entered the channel (used by MOSAIC)
* ``pdf(t, rate, shape)``         theoretical density (used for plotting)
* ``sample(rate, shape, size)``   random waiting times (used by DelaySSA)
* ``laplace_rate(rate, shape)``   a random Poisson rate drawn from the mixing
                                   density of the Laplace Gillespie algorithm
                                   (only for completely monotone survival
                                   functions)

See Table S1 of Pelissier et al., Brief. Bioinform. 27(5), 2026 for the
parametrisations.
"""

from math import log, gamma as gamma_fn, exp, pi, sqrt, erf, atan, sin, inf, isfinite

import numpy as np
import scipy.stats as stat
from scipy.special import gammainc

# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

_ALIASES = {
    'exponential': 'Exponential', 'exp': 'Exponential', 'poisson': 'Exponential',
    'gamma': 'Gamma', 'gam': 'Gamma',
    'weibull': 'Weibull', 'weib': 'Weibull',
    'gaussian': 'Gaussian', 'normal': 'Gaussian', 'norm': 'Gaussian',
    'lognormal': 'Lognormal', 'lognorm': 'Lognormal', 'log-normal': 'Lognormal',
    'cauchy': 'Cauchy', 'cau': 'Cauchy',
    'pareto': 'Pareto', 'pareto1': 'Pareto', 'power-law': 'Pareto', 'powerlaw': 'Pareto',
    'lomax': 'Lomax', 'pareto2': 'Lomax',
}

DISTRIBUTIONS = ('Exponential', 'Gamma', 'Weibull', 'Gaussian', 'Lognormal',
                 'Cauchy', 'Pareto', 'Lomax')

METHODS = ('MOSAIC', 'Laplace', 'Delay')

_METHOD_ALIASES = {
    'mosaic': 'MOSAIC', 'rejection': 'MOSAIC', 'regir': 'MOSAIC',
    'laplace': 'Laplace', 'laplace gillespie': 'Laplace', 'lga': 'Laplace',
    'delay': 'Delay', 'delayssa': 'Delay', 'delay-ssa': 'Delay', 'nrm': 'Delay',
}


def canonical_distribution(name):
    """Return the canonical distribution name for any accepted alias."""
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            "Unsupported distribution '%s'. Supported: %s" % (name, ', '.join(DISTRIBUTIONS)))
    return _ALIASES[key]


def canonical_method(name):
    """Return the canonical method name for any accepted alias."""
    key = str(name).strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(
            "Unsupported method '%s'. Supported: %s" % (name, ', '.join(METHODS)))
    return _METHOD_ALIASES[key]


# --------------------------------------------------------------------------
# Parameter conversions (rate, shape) -> native scipy parameters
# --------------------------------------------------------------------------

def _gamma_params(rate, shape):
    """Gamma with mean 1/rate and shape alpha: returns (alpha, scale)."""
    return shape, 1.0 / (shape * rate)


def _weibull_params(rate, shape):
    """Weibull with mean 1/rate and shape k: returns (k, scale)."""
    return shape, 1.0 / (rate * gamma_fn(1.0 + 1.0 / shape))


def _gaussian_params(rate, shape):
    """Normal with mean 1/rate and std = shape * mean: returns (mu, sigma)."""
    mu = 1.0 / rate
    return mu, shape * mu


def _lognormal_params(rate, shape):
    """Lognormal with mean 1/rate and std = shape * mean: returns (mu, sigma) of log."""
    sigma2 = log(1.0 + shape ** 2)
    mu = log(1.0 / rate) - sigma2 / 2.0
    return mu, sqrt(sigma2)


def _cauchy_params(rate, shape):
    """Cauchy with median 1/rate and scale = shape * median: returns (x0, gamma)."""
    x0 = 1.0 / rate
    return x0, shape * x0


def _pareto_params(rate, shape):
    """Pareto type I with median 1/rate and tail index alpha: returns (alpha, x_min)."""
    return shape, 1.0 / (rate * 2.0 ** (1.0 / shape))


def _lomax_params(rate, shape):
    """Lomax (Pareto II) with median 1/rate and tail index alpha: returns (alpha, kappa)."""
    return shape, 1.0 / (rate * (2.0 ** (1.0 / shape) - 1.0))


def scipy_distribution(distribution, rate, shape):
    """Return a frozen ``scipy.stats`` distribution for (distribution, rate, shape).

    Gaussian and Cauchy are truncated to non-negative waiting times, which is
    the distribution actually produced by a hazard-based simulation.
    """
    d = canonical_distribution(distribution)
    if d == 'Exponential':
        return stat.expon(scale=1.0 / rate)
    if d == 'Gamma':
        a, s = _gamma_params(rate, shape)
        return stat.gamma(a, scale=s)
    if d == 'Weibull':
        k, s = _weibull_params(rate, shape)
        return stat.weibull_min(k, scale=s)
    if d == 'Gaussian':
        mu, sigma = _gaussian_params(rate, shape)
        return stat.truncnorm(-mu / sigma, inf, loc=mu, scale=sigma)
    if d == 'Lognormal':
        mu, sigma = _lognormal_params(rate, shape)
        return stat.lognorm(sigma, scale=exp(mu))
    if d == 'Cauchy':
        x0, g = _cauchy_params(rate, shape)
        return _TruncatedCauchy(x0, g)
    if d == 'Pareto':
        a, xm = _pareto_params(rate, shape)
        return stat.pareto(a, scale=xm)
    if d == 'Lomax':
        a, kappa = _lomax_params(rate, shape)
        return stat.lomax(a, scale=kappa)
    raise ValueError(d)


class _TruncatedCauchy:
    """Cauchy distribution conditioned on t >= 0 (minimal frozen-distribution API)."""

    def __init__(self, x0, g):
        self.x0, self.g = x0, g
        self._F0 = stat.cauchy.cdf(0.0, loc=x0, scale=g)
        self._norm = 1.0 - self._F0

    def pdf(self, t):
        t = np.asarray(t, dtype=float)
        return np.where(t >= 0, stat.cauchy.pdf(t, self.x0, self.g) / self._norm, 0.0)

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        return np.where(t >= 0, stat.cauchy.sf(t, self.x0, self.g) / self._norm, 1.0)

    def rvs(self, size=None, random_state=None):
        rng = np.random if random_state is None else random_state
        u = rng.uniform(self._F0, 1.0, size=size)
        return stat.cauchy.ppf(u, self.x0, self.g)

    def mean(self):
        return inf


# --------------------------------------------------------------------------
# Hazard functions  h(t) = pdf(t) / survival(t)
# (fast closed forms; used at every MOSAIC step)
# --------------------------------------------------------------------------

def exponential_hazard(t, rate, shape=None):
    return rate


def gamma_hazard(t, rate, shape):
    alpha = shape
    if t <= 0:
        return 0.0 if alpha > 1 else (rate if alpha == 1 else inf)
    beta = alpha * rate
    pdf = (beta ** alpha) * (t ** (alpha - 1)) * exp(-beta * t) / gamma_fn(alpha)
    sf = 1.0 - gammainc(alpha, beta * t)
    return pdf / sf if sf > 0 else inf


def weibull_hazard(t, rate, shape):
    k, s = _weibull_params(rate, shape)
    if t <= 0:
        return 0.0 if k > 1 else (rate if k == 1 else inf)
    return (k / s) * (t / s) ** (k - 1)


def gaussian_hazard(t, rate, shape):
    mu, sigma = _gaussian_params(rate, shape)
    if t > mu + 7 * sigma:
        # asymptotic form: the cdf is numerically 1 in this region
        return (t - mu) / sigma ** 2
    pdf = exp(-0.5 * ((t - mu) / sigma) ** 2) / (sigma * sqrt(2 * pi))
    sf = 0.5 * (1.0 - erf((t - mu) / (sigma * sqrt(2))))
    return pdf / sf if sf > 0 else inf


def lognormal_hazard(t, rate, shape):
    if t <= 0:
        return 0.0
    mu, sigma = _lognormal_params(rate, shape)
    z = (log(t) - mu) / sigma
    pdf = exp(-0.5 * z * z) / (t * sigma * sqrt(2 * pi))
    sf = 0.5 * (1.0 - erf(z / sqrt(2)))
    return pdf / sf if sf > 0 else inf


def cauchy_hazard(t, rate, shape):
    x0, g = _cauchy_params(rate, shape)
    pdf = 1.0 / (pi * g * (1.0 + ((t - x0) / g) ** 2))
    sf = 0.5 - atan((t - x0) / g) / pi
    return pdf / sf if sf > 0 else inf


def pareto_hazard(t, rate, shape):
    a, xm = _pareto_params(rate, shape)
    return a / t if t >= xm else 0.0


def lomax_hazard(t, rate, shape):
    a, kappa = _lomax_params(rate, shape)
    return a / (kappa + max(t, 0.0))


_HAZARDS = {
    'Exponential': exponential_hazard,
    'Gamma': gamma_hazard,
    'Weibull': weibull_hazard,
    'Gaussian': gaussian_hazard,
    'Lognormal': lognormal_hazard,
    'Cauchy': cauchy_hazard,
    'Pareto': pareto_hazard,
    'Lomax': lomax_hazard,
}


def hazard_function(distribution):
    """Return the hazard function ``h(t, rate, shape)`` of a distribution."""
    return _HAZARDS[canonical_distribution(distribution)]


def hazard(t, distribution, rate, shape):
    """Evaluate the hazard with a slow but robust fallback for numerical overflow."""
    h = _HAZARDS[canonical_distribution(distribution)](t, rate, shape)
    if isfinite(h):
        return h
    frozen = scipy_distribution(distribution, rate, shape)
    try:
        h = float(np.exp(frozen.logpdf(t) - frozen.logsf(t)))
    except AttributeError:
        h = float(frozen.pdf(t) / frozen.sf(t))
    return h


# --------------------------------------------------------------------------
# Hazard shape information used by MOSAIC to bound the rate
# --------------------------------------------------------------------------

def hazard_is_bounded(distribution, shape):
    """True if the hazard is finite for all t >= 0 (requirement of MOSAIC)."""
    d = canonical_distribution(distribution)
    if d in ('Gamma', 'Weibull'):
        return shape >= 1
    return True


def hazard_is_increasing(distribution, shape):
    """True if the hazard is monotonically non-decreasing in t.

    For such distributions the maximum rate over a population is the hazard of
    the reactant that has been waiting the longest. Lognormal hazards are not
    monotone, but for a small relative standard deviation the maximum lies far
    in the tail, so they are treated as increasing (see Pelissier et al.).
    """
    d = canonical_distribution(distribution)
    if d in ('Gamma', 'Weibull'):
        return shape >= 1
    if d == 'Gaussian':
        return True
    if d == 'Lognormal':
        return shape < 0.25
    return False


def hazard_upper_bound(distribution, rate, shape, Tend):
    """A fixed upper bound of the hazard on [0, Tend] for non-increasing hazards."""
    d = canonical_distribution(distribution)
    if d == 'Exponential':
        return rate
    if d == 'Pareto':
        a, xm = _pareto_params(rate, shape)
        return a / xm
    if d == 'Lomax':
        a, kappa = _lomax_params(rate, shape)
        return a / kappa
    # Cauchy, Lognormal with large shape: scan numerically
    f = _HAZARDS[d]
    t = np.linspace(0.0, Tend, num=1000)
    values = np.array([f(ti, rate, shape) for ti in t])
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else rate


# --------------------------------------------------------------------------
# Laplace Gillespie: completely monotone survival functions as mixtures of
# exponentials.  psi(tau) = int p(lambda) exp(-lambda tau) dlambda
# --------------------------------------------------------------------------

def laplace_supported(distribution, shape):
    """True if the distribution can be simulated with the Laplace Gillespie algorithm."""
    d = canonical_distribution(distribution)
    if d == 'Exponential' or d == 'Lomax':
        return True
    if d in ('Gamma', 'Weibull'):
        return shape <= 1
    return False


def _positive_stable(alpha, rng):
    """Kanter's sampler of a positive stable variable S with E[exp(-u S)] = exp(-u^alpha)."""
    u = rng.uniform(0.0, pi)
    w = rng.exponential(1.0)
    a = (sin((1.0 - alpha) * u) * sin(alpha * u) ** (alpha / (1.0 - alpha))
         / sin(u) ** (1.0 / (1.0 - alpha)))
    return (a / w) ** ((1.0 - alpha) / alpha)


def laplace_rate(distribution, rate, shape, rng=None):
    """Draw a Poisson rate from the mixing density of the survival function.

    * Exponential: deterministic ``rate``
    * Gamma (alpha <= 1): lambda = beta / B,  B ~ Beta(alpha, 1 - alpha), beta = alpha * rate
    * Weibull (alpha <= 1): lambda = S / scale,  S positive alpha-stable
    * Lomax: lambda ~ Gamma(alpha, scale = 1 / kappa)
    """
    rng = np.random if rng is None else rng
    d = canonical_distribution(distribution)
    if d == 'Exponential' or ((d in ('Gamma', 'Weibull')) and shape == 1):
        return float(rate)
    if d == 'Gamma':
        if shape > 1:
            raise ValueError('Laplace Gillespie requires a Gamma shape parameter <= 1')
        beta = shape * rate
        return beta / rng.beta(shape, 1.0 - shape)
    if d == 'Weibull':
        if shape > 1:
            raise ValueError('Laplace Gillespie requires a Weibull shape parameter <= 1')
        _, s = _weibull_params(rate, shape)
        return _positive_stable(shape, rng) / s
    if d == 'Lomax':
        a, kappa = _lomax_params(rate, shape)
        return rng.gamma(a, 1.0 / kappa)
    raise ValueError("The Laplace Gillespie method does not support the %s distribution" % d)


# --------------------------------------------------------------------------
# Sampling and theoretical density
# --------------------------------------------------------------------------

def sample(distribution, rate, shape, size=None, rng=None):
    """Draw ``size`` waiting times (a float if ``size`` is None)."""
    frozen = scipy_distribution(distribution, rate, shape)
    out = frozen.rvs(size=1 if size is None else size, random_state=rng)
    return float(out[0]) if size is None else np.asarray(out, dtype=float)


def pdf(t, distribution, rate, shape):
    """Theoretical probability density of the waiting time at ``t``."""
    return scipy_distribution(distribution, rate, shape).pdf(np.asarray(t, dtype=float))


def mean(distribution, rate, shape):
    """Theoretical mean waiting time (inf when undefined)."""
    return float(scipy_distribution(distribution, rate, shape).mean())


def fit(distribution, samples):
    """Fit the native scipy parameters to observed waiting times (for plotting)."""
    d = canonical_distribution(distribution)
    samples = np.asarray(samples, dtype=float)
    if d == 'Exponential':
        P = stat.expon.fit(samples, floc=0)
        return P, lambda t: stat.expon.pdf(t, *P)
    if d == 'Gamma':
        P = stat.gamma.fit(samples, floc=0)
        return P, lambda t: stat.gamma.pdf(t, *P)
    if d == 'Weibull':
        P = stat.weibull_min.fit(samples, floc=0)
        return P, lambda t: stat.weibull_min.pdf(t, *P)
    if d == 'Gaussian':
        P = stat.norm.fit(samples)
        return P, lambda t: stat.norm.pdf(t, *P)
    if d == 'Lognormal':
        P = stat.lognorm.fit(samples, floc=0)
        return P, lambda t: stat.lognorm.pdf(t, *P)
    if d == 'Cauchy':
        P = stat.cauchy.fit(samples)
        return P, lambda t: stat.cauchy.pdf(t, *P)
    if d == 'Pareto':
        P = stat.pareto.fit(samples, floc=0)
        return P, lambda t: stat.pareto.pdf(t, *P)
    if d == 'Lomax':
        P = stat.lomax.fit(samples, floc=0)
        return P, lambda t: stat.lomax.pdf(t, *P)
    raise ValueError(d)


def default_method(distribution, shape):
    """Pick the recommended simulation method for a distribution.

    MOSAIC whenever the hazard is bounded, Laplace Gillespie for the heavy
    early-time hazards of Gamma/Weibull with shape < 1.
    """
    if hazard_is_bounded(distribution, shape):
        return 'MOSAIC'
    return 'Laplace'


def method_supported(method, distribution, shape):
    """Check whether ``method`` can simulate ``distribution`` with this shape."""
    m = canonical_method(method)
    if m == 'MOSAIC':
        return hazard_is_bounded(distribution, shape)
    if m == 'Laplace':
        return laplace_supported(distribution, shape)
    return True  # Delay works for every distribution
