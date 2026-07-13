r"""Marginal-likelihood estimators for the Wishart process.

Fitting the model means maximising the *marginal* likelihood, in which the
per-trial latent :math:`v` is integrated out:

.. math::

    p(y \mid x, \theta) \;=\; \int p\big(y \mid \mu(x) + G(x) v\big)\,
                                    \mathcal N(v; 0, I_Q)\, \mathrm d v .

Three estimators are provided, selected per trial:

* **conjugate** -- for the Gaussian likelihood the integral is exact:
  :math:`y \sim \mathcal N(\mu(x), \Sigma(x))`.
* **qmc** -- a randomised quasi-Monte-Carlo estimate that draws :math:`v` from
  the prior; ``logsumexp`` of the per-sample log-likelihoods gives an
  importance-weighted (IWAE-style) estimate of the log marginal.  This is the
  cheap, differentiable objective used during training.
* **laplace** -- a lower-variance multiple-importance-sampling estimate whose
  proposal is an equal mixture of the prior and a Laplace approximation to the
  latent posterior.  It costs a per-trial Newton solve, so it is intended for
  accurate *evaluation* (e.g. held-out log-likelihood) rather than every
  gradient step.

Every estimator operates on a single trial and is vectorised over a dataset
with ``jax.vmap`` inside :func:`dataset_marginal_loglike`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jxr

from .optim import newton_minimize

__all__ = [
    "marginal_loglike_point_conjugate",
    "marginal_loglike_point_qmc",
    "marginal_loglike_point_laplace",
    "dataset_marginal_loglike",
]

_LOG_2PI = jnp.log(2.0 * jnp.pi)


def marginal_loglike_point_conjugate(model, params, y, x) -> jnp.ndarray:
    """Exact log marginal likelihood for a conjugate (Gaussian) likelihood."""
    mu = model.mean_at(params, x)
    cov = model.cov_at(params, x)
    return model.likelihood.marginal_log_prob(params.lik, y, mu, cov)


def marginal_loglike_point_qmc(model, params, y, x, key, lattice) -> jnp.ndarray:
    r"""Randomised-QMC estimate of the log marginal likelihood for one trial.

    Draws latent samples ``v`` from the prior via the lattice, forms
    ``eta = mu + G v`` and returns ``logsumexp_s log p(y | eta_s) - log S``.
    """
    mu = model.mean_at(params, x)
    g = model.sqrt_factor_at(params, x)  # (N, Q)
    v = lattice.gaussian_points(key)  # (S, Q)
    eta = mu[None, :] + v @ g.T  # (S, N)
    log_p = jax.vmap(lambda e: model.likelihood.log_prob(params.lik, y, e))(eta)
    return jax.scipy.special.logsumexp(log_p) - jnp.log(v.shape[0])


def marginal_loglike_point_laplace(model, params, y, x, key, lattice) -> jnp.ndarray:
    r"""Laplace-proposal multiple-importance-sampling estimate for one trial.

    The proposal is an equal mixture of the prior ``N(0, I)`` and the Laplace
    approximation ``N(m, H^{-1})`` to the latent posterior, where ``m`` is the
    posterior mode and ``H`` its Hessian.  Uses the balance heuristic across the
    two components.
    """
    mu = model.mean_at(params, x)
    g = model.sqrt_factor_at(params, x)  # (N, Q)
    q_dim = g.shape[1]

    def neg_joint(v):
        eta = mu + g @ v
        log_lik = model.likelihood.log_prob(params.lik, y, eta)
        log_prior = jnp.sum(jax.scipy.stats.norm.logpdf(v))
        return -(log_lik + log_prior)

    res = newton_minimize(neg_joint, jnp.zeros(q_dim))
    m, lh = res.x, res.hess_chol  # H = lh @ lh.T ; Cov = H^{-1}

    k_prior, k_lap = jxr.split(key)
    v_prior = lattice.gaussian_points(k_prior)  # (S, Q) ~ N(0, I)
    eps = lattice.gaussian_points(k_lap)  # (S, Q)
    # v = m + lh^{-T} eps  ->  Cov(v) = (lh lh^T)^{-1} = H^{-1}
    v_lap = m[None, :] + jax.scipy.linalg.solve_triangular(lh.T, eps.T, lower=False).T
    v_all = jnp.concatenate([v_prior, v_lap], axis=0)  # (2S, Q)

    # Component densities.
    log_prior = jnp.sum(jax.scipy.stats.norm.logpdf(v_all), axis=1)
    diff = v_all - m[None, :]
    quad = jnp.sum((diff @ lh) ** 2, axis=1)  # (v-m)^T H (v-m)
    log_det_cov = -2.0 * jnp.sum(jnp.log(jnp.diagonal(lh)))
    log_lap = -0.5 * (q_dim * _LOG_2PI + log_det_cov + quad)

    # Mixture proposal density (balance heuristic).
    log_q = jnp.logaddexp(jnp.log(0.5) + log_prior, jnp.log(0.5) + log_lap)

    eta_all = mu[None, :] + v_all @ g.T
    log_lik = jax.vmap(lambda e: model.likelihood.log_prob(params.lik, y, e))(eta_all)

    log_weights = log_lik + log_prior - log_q
    return jax.scipy.special.logsumexp(log_weights) - jnp.log(v_all.shape[0])


def _resolve_method(model, method: str) -> str:
    if method == "auto":
        return "conjugate" if model.likelihood.conjugate_gaussian else "qmc"
    if method == "conjugate" and not model.likelihood.conjugate_gaussian:
        raise ValueError("conjugate method requires a Gaussian likelihood")
    return method


def dataset_marginal_loglike(
    model,
    params,
    Y: jnp.ndarray,
    X: jnp.ndarray,
    key: jax.Array | None = None,
    lattice=None,
    method: str = "auto",
) -> jnp.ndarray:
    """Total marginal log-likelihood over a dataset ``(Y, X)``.

    Parameters
    ----------
    model : WishartProcessModel
    params : WPParams
    Y : jnp.ndarray
        Observations, shape ``(T, N)``.
    X : jnp.ndarray
        Conditions, shape ``(T, D)`` (or ``(T,)`` for 1-D).
    key : jax.Array, optional
        PRNG key; required for the ``qmc`` and ``laplace`` methods.
    lattice : QMCLattice, optional
        Lattice used for Monte-Carlo integration; required for ``qmc`` /
        ``laplace``.  Its ``dim`` must equal ``model.latent_dim``.
    method : {"auto", "conjugate", "qmc", "laplace"}
        Estimator to use.  ``"auto"`` picks ``conjugate`` for Gaussian models and
        ``qmc`` otherwise.
    """
    X = jnp.atleast_2d(X.T).T if X.ndim == 1 else jnp.asarray(X)
    Y = jnp.asarray(Y)
    resolved = _resolve_method(model, method)

    if resolved == "conjugate":
        vals = jax.vmap(lambda y, x: marginal_loglike_point_conjugate(model, params, y, x))(
            Y, X
        )
        return jnp.sum(vals)

    if lattice is None:
        raise ValueError(f"method={resolved!r} requires a QMCLattice")
    if lattice.dim != model.latent_dim:
        raise ValueError(
            f"lattice.dim ({lattice.dim}) must equal model.latent_dim "
            f"({model.latent_dim})"
        )
    if key is None:
        raise ValueError(f"method={resolved!r} requires a PRNG key")

    point_fn = {
        "qmc": marginal_loglike_point_qmc,
        "laplace": marginal_loglike_point_laplace,
    }[resolved]

    keys = jxr.split(key, Y.shape[0])
    vals = jax.vmap(lambda y, x, k: point_fn(model, params, y, x, k, lattice))(Y, X, keys)
    return jnp.sum(vals)
