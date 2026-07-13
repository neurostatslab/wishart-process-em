r"""Observation (noise) models for the Wishart process.

The Wishart process places a smooth prior on a *latent linear predictor*

.. math::

    \eta(x) \;=\; \mu(x) + G(x)\, v, \qquad v \sim \mathcal N(0, I_Q),

whose covariance :math:`\Sigma(x) = G(x) G(x)^\top` is the object of interest.
A :class:`Likelihood` maps this latent predictor to observations ``y``.  Three
models are provided:

* :class:`Gaussian` -- ``y ~ N(mu(x), Sigma(x))``.  This is *conjugate*: the
  latent ``v`` integrates out analytically, so no Monte-Carlo is needed.  It is
  the model used throughout the original paper for (pre-processed) neural data.
* :class:`Poisson` -- ``y_n ~ Poisson(softplus(eta_n))``.  Suited to raw spike
  counts.  The latent must be integrated numerically (QMC + Laplace).
* :class:`NegativeBinomial` -- ``y_n ~ NB(mean=softplus(eta_n), r_n)`` with a
  per-neuron dispersion ``r_n``.  Captures over-dispersed counts.

Each likelihood carries only *static* configuration; any trainable parameters
(e.g. the NB dispersion, or an optional Gaussian observation-noise variance)
live in the model parameter pytree and are passed in as ``lik_params``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jxr
from jax.scipy.special import gammaln

__all__ = ["Likelihood", "Gaussian", "Poisson", "NegativeBinomial", "get_likelihood"]

_RATE_FLOOR = 1e-6  # keeps rates strictly positive for log-likelihoods


def _softplus_rate(eta: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(eta) + _RATE_FLOOR


class Likelihood(abc.ABC):
    """Abstract observation model mapping a latent predictor ``eta`` to ``y``.

    Subclasses are frozen dataclasses so they can be used as static arguments to
    ``jax.jit``.  Trainable per-model parameters are threaded through the
    ``lik_params`` argument (a pytree, possibly ``None``).
    """

    #: Whether the latent ``v`` integrates out in closed form (Gaussian only).
    conjugate_gaussian: bool = False

    def init_params(self, num_neurons: int, key: jax.Array):
        """Return initial trainable likelihood parameters (default: ``None``)."""
        return None

    def log_prior(self, lik_params) -> jnp.ndarray:
        """Log-prior over the trainable likelihood parameters."""
        return jnp.asarray(0.0)

    @abc.abstractmethod
    def log_prob(self, lik_params, y: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
        """Log ``p(y | eta)`` for a single trial, summed over neurons."""

    @abc.abstractmethod
    def observe(self, key: jax.Array, lik_params, eta: jnp.ndarray) -> jnp.ndarray:
        """Sample observations ``y`` given the latent predictor ``eta``."""

    @abc.abstractmethod
    def mean(self, lik_params, eta: jnp.ndarray) -> jnp.ndarray:
        """Expected observation ``E[y | eta]``."""


@dataclass(frozen=True)
class Gaussian(Likelihood):
    r"""Multivariate-Gaussian observations ``y ~ N(mu(x), Sigma(x))``.

    Conjugate: fitting uses the exact marginal likelihood via
    :meth:`marginal_log_prob`; no latent sampling is required.  An optional
    homoscedastic observation-noise variance can be added on top of
    :math:`\Sigma(x)` by enabling ``learn_obs_noise``.

    Parameters
    ----------
    learn_obs_noise : bool
        If ``True``, a scalar observation-noise variance is learned and added to
        the diagonal of the covariance.
    """

    learn_obs_noise: bool = False
    conjugate_gaussian: bool = True

    def init_params(self, num_neurons, key):
        if self.learn_obs_noise:
            # softplus(-4) ~ 0.018 : small initial observation noise.
            return {"log_obs_var": jnp.asarray(-4.0)}
        return None

    def obs_var(self, lik_params) -> jnp.ndarray:
        if lik_params is None:
            return jnp.asarray(0.0)
        return jax.nn.softplus(lik_params["log_obs_var"])

    def marginal_log_prob(self, lik_params, y, mean, cov) -> jnp.ndarray:
        """Exact log ``N(y; mean, cov + obs_var * I)`` for one trial."""
        n = mean.shape[0]
        cov = cov + self.obs_var(lik_params) * jnp.eye(n)
        return jax.scipy.stats.multivariate_normal.logpdf(y, mean, cov)

    def log_prob(self, lik_params, y, eta):
        # Diagonal fallback (used only if a caller opts into latent sampling for
        # a Gaussian model); the conjugate path is preferred.
        var = jnp.maximum(self.obs_var(lik_params), _RATE_FLOOR)
        return jnp.sum(jax.scipy.stats.norm.logpdf(y, eta, jnp.sqrt(var)))

    def observe(self, key, lik_params, eta):
        var = self.obs_var(lik_params)
        noise = jnp.sqrt(var) * jxr.normal(key, eta.shape)
        return eta + noise

    def mean(self, lik_params, eta):
        return eta


@dataclass(frozen=True)
class Poisson(Likelihood):
    r"""Poisson-count observations ``y_n ~ Poisson(softplus(eta_n))``.

    The softplus inverse-link keeps rates positive while behaving linearly for
    large ``eta`` (unlike ``exp``, which can overflow).  Non-conjugate: the
    latent ``v`` is integrated numerically.
    """

    def log_prob(self, lik_params, y, eta):
        rate = _softplus_rate(eta)
        return jnp.sum(jax.scipy.stats.poisson.logpmf(y, rate))

    def observe(self, key, lik_params, eta):
        return jxr.poisson(key, _softplus_rate(eta)).astype(eta.dtype)

    def mean(self, lik_params, eta):
        return _softplus_rate(eta)


@dataclass(frozen=True)
class NegativeBinomial(Likelihood):
    r"""Over-dispersed counts ``y_n ~ NB(mean=softplus(eta_n), r_n)``.

    Uses the mean/dispersion parameterisation with per-neuron dispersion
    :math:`r_n > 0`; the variance is :math:`\mu + \mu^2 / r`.  As
    :math:`r \to \infty` the model approaches :class:`Poisson`.
    """

    def init_params(self, num_neurons, key):
        # softplus(log_r) ~ 1 initially (moderate over-dispersion).
        return {"log_r": jnp.zeros(num_neurons)}

    def _dispersion(self, lik_params) -> jnp.ndarray:
        return jax.nn.softplus(lik_params["log_r"]) + _RATE_FLOOR

    def log_prior(self, lik_params):
        # Weak standard-normal prior on the unconstrained dispersion.
        return jnp.sum(jax.scipy.stats.norm.logpdf(lik_params["log_r"]))

    def _log_prob_vec(self, r, y, mu):
        # NB log-pmf in mean/dispersion form.
        return (
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1.0)
            + r * (jnp.log(r) - jnp.log(r + mu))
            + y * (jnp.log(mu) - jnp.log(r + mu))
        )

    def log_prob(self, lik_params, y, eta):
        r = self._dispersion(lik_params)
        mu = _softplus_rate(eta)
        return jnp.sum(self._log_prob_vec(r, y, mu))

    def observe(self, key, lik_params, eta):
        # Gamma-Poisson mixture: y | g ~ Poisson(g), g ~ Gamma(r, mu/r).
        r = self._dispersion(lik_params)
        mu = _softplus_rate(eta)
        k_gam, k_pois = jxr.split(key)
        g = jxr.gamma(k_gam, r) * (mu / r)
        return jxr.poisson(k_pois, g).astype(eta.dtype)

    def mean(self, lik_params, eta):
        return _softplus_rate(eta)


_REGISTRY = {
    "gaussian": Gaussian,
    "poisson": Poisson,
    "negative_binomial": NegativeBinomial,
    "nb": NegativeBinomial,
}


def get_likelihood(name: str | Likelihood, **kwargs) -> Likelihood:
    """Look up a likelihood by name (e.g. ``"poisson"``) or pass one through."""
    if isinstance(name, Likelihood):
        return name
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown likelihood {name!r}; choose from {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)
