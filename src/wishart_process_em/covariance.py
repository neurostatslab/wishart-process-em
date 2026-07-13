r"""Parameter container and covariance assembly for the Wishart process.

The covariance model follows the "full family" of Nejatbakhsh, Garon &
Williams (2023, sec. 2.3):

.. math::

    \Sigma(x) \;=\; L \big( U(x) U(x)^\top + \Lambda(x) \big) L^\top ,

where

* :math:`U(x) \in \mathbb R^{N \times P}` is a low-rank factor whose columns are
  weight-space GPs, scaled by a learnable per-latent (ARD) scale ``sZ``;
* :math:`\Lambda(x)` is an optional non-negative diagonal (the low-rank *plus
  diagonal* / ``WPlrd`` extension), each entry a softplus-transformed GP;
* :math:`L` is an optional learnable lower-triangular *scale matrix*, capturing
  condition-independent covariance and typically initialised from the Cholesky
  factor of the grand empirical covariance.

For sampling and for the non-Gaussian marginal likelihood we use the square-root
factor :math:`G(x)` with :math:`\Sigma(x) = G(x) G(x)^\top`, so that the latent
predictor is :math:`\eta(x) = \mu(x) + G(x) v` with :math:`v \sim N(0, I_Q)`.
When the diagonal is disabled, :math:`Q = P` and the latent integral has the
small dimension that makes QMC efficient.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["WPParams", "softplus_inverse", "scale_tril_from_raw", "raw_from_scale_tril"]


class WPParams(NamedTuple):
    """Trainable parameters of a :class:`~wishart_process_em.model.WishartProcessModel`.

    Optional components (``diag_w``, ``scale_raw``, ``lik``) are ``None`` when the
    corresponding feature is disabled; JAX treats ``None`` as an empty subtree,
    so those components simply receive no gradients.
    """

    mean_w: jnp.ndarray  # (N, M) mean-function GP weights
    mean_b: jnp.ndarray  # (N,) per-neuron mean bias
    factor_w: jnp.ndarray  # (N, P, M) covariance-factor GP weights
    log_latent_scale: jnp.ndarray  # (P,) unconstrained per-latent (ARD) scale
    diag_w: Any = None  # (N, M) optional diagonal-GP weights
    scale_raw: Any = None  # (N, N) optional unconstrained scale matrix
    lik: Any = None  # optional likelihood parameters


def softplus_inverse(y: jnp.ndarray) -> jnp.ndarray:
    """Inverse of ``softplus``; maps a positive value to its raw pre-image."""
    y = jnp.asarray(y)
    # log(exp(y) - 1), computed stably.
    return y + jnp.log(-jnp.expm1(-y))


def scale_tril_from_raw(scale_raw: jnp.ndarray) -> jnp.ndarray:
    """Build a lower-triangular matrix with positive diagonal from a raw matrix.

    The strictly-lower entries are taken as-is; the diagonal is passed through
    ``softplus`` to guarantee positivity (and hence a valid Cholesky factor).
    """
    strictly_lower = jnp.tril(scale_raw, k=-1)
    diag = jax.nn.softplus(jnp.diagonal(scale_raw))
    return strictly_lower + jnp.diag(diag)


def raw_from_scale_tril(tril: jnp.ndarray) -> jnp.ndarray:
    """Inverse of :func:`scale_tril_from_raw` for initialisation from a Cholesky."""
    strictly_lower = jnp.tril(tril, k=-1)
    diag_raw = softplus_inverse(jnp.diagonal(tril))
    return strictly_lower + jnp.diag(diag_raw)
