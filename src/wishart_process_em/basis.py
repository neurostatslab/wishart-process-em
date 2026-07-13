r"""Fourier basis helpers backed by :mod:`nemos.basis`.

This package does not define its own basis class: a
:class:`~wishart_process_em.model.WishartProcessModel` consumes a **nemos basis
object directly** (typically :class:`nemos.basis.FourierEval`).  The two helpers
here cover the only Wishart-process-specific needs on top of nemos:

* :func:`fourier_basis` -- construct a nemos Fourier basis with the defaults this
  package expects (inputs on the unit box, no constant/intercept term, which is
  handled by an explicit bias in the model).
* :func:`fourier_feature_scale` -- the Gaussian-process spectral scaling
  :math:`\tau_k = \sqrt{S(2\pi k)}` that turns a plain Fourier basis into a
  *weight-space GP*: with this scaling a standard-normal prior on the weights
  induces a stationary GP whose smoothness is set by the spectral density (see
  :mod:`wishart_process_em.kernels`).

Any nemos evaluation basis works with the model; the spectral scaling only
applies to Fourier bases (those exposing ``masked_frequencies``).  Inputs are
assumed to be scaled to the unit box ``[0, 1] ** num_dims``; because Fourier
features are modes on the torus, periodic conditions wrap around naturally.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import nemos.basis as nb
import numpy as np

__all__ = ["fourier_basis", "fourier_feature_scale"]

_TWO_PI = 2.0 * np.pi


def fourier_basis(
    max_freq: int,
    num_dims: int = 1,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> nb.FourierEval:
    """Construct a nemos Fourier basis with this package's defaults.

    Enumerates integer frequencies ``{0, ..., max_freq - 1}`` per dimension,
    drops the constant (DC) term, and takes the Cartesian product across
    dimensions.  ``bounds`` fixes the input period (default: the unit box), which
    is required so that single-point evaluation under ``jax.vmap`` is well
    defined.

    Parameters
    ----------
    max_freq : int
        Number of integer frequencies per input dimension.
    num_dims : int, optional
        Dimensionality of the condition space.
    bounds : tuple, optional
        Input range mapped onto one period, per dimension.

    Returns
    -------
    nemos.basis.FourierEval
    """
    if max_freq < 1:
        raise ValueError("max_freq must be >= 1")
    if num_dims < 1:
        raise ValueError("num_dims must be >= 1")
    nd_bounds = bounds if num_dims == 1 else tuple(bounds for _ in range(num_dims))
    return nb.FourierEval(
        frequencies=max_freq,
        ndim=num_dims,
        bounds=nd_bounds,
        frequency_mask="no-intercept",
    )


def fourier_feature_scale(
    basis: nb.FourierEval, spectral_density: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    r"""Per-feature Gaussian-process scaling ``sqrt(S(|2*pi*k|**2))``.

    Returns a vector aligned with the basis's feature layout ``[cos-block,
    sin-block]``, so that multiplying the (nemos) features by it yields a
    weight-space GP under a standard-normal weight prior.

    Parameters
    ----------
    basis : nemos Fourier basis
        Must expose ``masked_frequencies`` (shape ``(num_dims, n_combos)``).
    spectral_density : Callable
        Maps squared angular frequency ``|omega|**2`` to the density value
        (see :mod:`wishart_process_em.kernels`).
    """
    if not hasattr(basis, "masked_frequencies"):
        raise TypeError(
            "fourier_feature_scale requires a Fourier basis exposing "
            "`masked_frequencies`"
        )
    freqs = np.asarray(basis.masked_frequencies)  # (num_dims, n_combos)
    sq_omega = jnp.asarray((_TWO_PI**2) * np.sum(freqs**2, axis=0))
    tau = jnp.sqrt(jnp.asarray(spectral_density(sq_omega)))
    return jnp.concatenate([tau, tau])  # cos-block, sin-block
