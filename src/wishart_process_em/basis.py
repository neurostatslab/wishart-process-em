r"""Finite (weight-space) Fourier basis, backed by :mod:`nemos.basis`.

The Wishart process places Gaussian-process priors on the mean function
:math:`\mu(x)` and on the columns of the covariance factor :math:`U(x)`.  Each
scalar GP is represented in *weight space* by a truncated Fourier feature
expansion,

.. math::

    f(x) \;=\; \sum_{m} w_m \, \phi_m(x),
    \qquad w_m \sim \mathcal N(0, 1),

where the features :math:`\phi_m` are sine/cosine functions scaled by the square
root of the kernel's spectral density.  With this scaling, a standard-normal
prior on the weights induces a stationary GP whose smoothness is set by the
spectral density (see :mod:`wishart_process_em.kernels`).

The Fourier feature functions themselves -- frequency enumeration, the
N-dimensional Cartesian product, evaluation, and input-range handling -- are
provided by :class:`nemos.basis.FourierEval`.  This module only adds the
Wishart-process-specific piece: the Gaussian-process spectral scaling that
turns nemos's plain Fourier basis into a weight-space GP.

Inputs ``x`` are assumed to be scaled to the unit box ``[0, 1] ** num_dims``.
Because the features are Fourier modes on the torus, periodic conditions (e.g.
grating orientation, reach angle) wrap around naturally; non-periodic
conditions should be mapped into ``[0, 1]`` with a little headroom.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import nemos.basis as nb
import numpy as np

__all__ = ["TruncatedFourierBasis"]

_TWO_PI = 2.0 * np.pi


def _spectral_amplitudes(
    frequencies: np.ndarray, spectral_density: Callable[[jnp.ndarray], jnp.ndarray]
) -> np.ndarray:
    """``sqrt(spectral_density(|omega|**2))`` for integer frequency columns."""
    sq_omega = jnp.asarray((_TWO_PI**2) * np.sum(frequencies**2, axis=0))
    return np.asarray(jnp.sqrt(jnp.asarray(spectral_density(sq_omega))))


class TruncatedFourierBasis:
    r"""Truncated Fourier-feature basis for scalar Gaussian processes.

    A function drawn from this basis has the form

    .. math::

        f(x) = \sum_k \tau_k \big( w^{\cos}_k \cos(2\pi k \cdot x)
                                 + w^{\sin}_k \sin(2\pi k \cdot x) \big),

    where the integer frequency vectors :math:`k` range over the Cartesian
    product ``{0, ..., max_freq - 1} ** num_dims`` (excluding the all-zero mode,
    handled by an explicit bias in the model) and :math:`\tau_k =
    \sqrt{S(2\pi k)}` with :math:`S` the kernel spectral density.  Frequencies
    whose amplitude falls below ``tol`` times the largest are pruned.

    Frequency enumeration and evaluation are delegated to
    :class:`nemos.basis.FourierEval`.

    Parameters
    ----------
    max_freq : int
        Number of integer frequencies enumerated *per input dimension*.
    num_dims : int
        Dimensionality of the input / condition space ``x``.
    spectral_density : Callable
        Maps squared angular frequency ``|omega|**2`` to the spectral density
        value.  See :mod:`wishart_process_em.kernels` for constructors.
    tol : float, optional
        Relative amplitude threshold below which frequencies are discarded.
    bounds : tuple, optional
        Input range mapped onto one period, per dimension.  Defaults to
        ``(0.0, 1.0)`` (inputs assumed pre-scaled to the unit box).

    Attributes
    ----------
    num_features : int
        Length ``M`` of the feature vector returned by :meth:`features`.
    """

    def __init__(
        self,
        max_freq: int,
        num_dims: int,
        spectral_density: Callable[[jnp.ndarray], jnp.ndarray],
        tol: float = 1e-5,
        bounds: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if max_freq < 1:
            raise ValueError("max_freq must be >= 1")
        if num_dims < 1:
            raise ValueError("num_dims must be >= 1")

        nd_bounds = bounds if num_dims == 1 else tuple(bounds for _ in range(num_dims))

        # Enumerate the full frequency grid via nemos, then keep the non-constant
        # frequencies whose spectral amplitude exceeds the tolerance.  nemos owns
        # the grid; we only decide which frequencies matter for this GP prior.
        full = nb.FourierEval(
            frequencies=max_freq, ndim=num_dims, bounds=nd_bounds, frequency_mask="all"
        )
        grid = np.asarray(full.masked_frequencies)  # (num_dims, max_freq ** num_dims)
        amplitudes = _spectral_amplitudes(grid, spectral_density)
        is_constant = np.all(grid == 0, axis=0)
        thresh = tol * amplitudes[~is_constant].max()
        keep = (~is_constant) & (amplitudes >= thresh)

        self._basis = nb.FourierEval(
            frequencies=max_freq,
            ndim=num_dims,
            bounds=nd_bounds,
            frequency_mask=keep.reshape((max_freq,) * num_dims),
        )
        kept = np.asarray(self._basis.masked_frequencies)
        tau = _spectral_amplitudes(kept, spectral_density)
        # Feature layout is [cos-block, sin-block]; scale both by tau.
        self._tau = jnp.asarray(np.concatenate([tau, tau]))

        self.num_features = int(self._tau.shape[0])
        self.num_dims = num_dims
        self.max_freq = max_freq
        self.tol = tol
        self.spectral_density = spectral_density

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the GP-scaled feature vector at a single input ``x``.

        ``x`` has shape ``(num_dims,)`` (or is a scalar when ``num_dims == 1``).
        Returns a vector of shape ``(num_features,)``.  Traceable, so it composes
        with :func:`jax.vmap`/:func:`jax.grad` (used by the model and by the
        Fisher-information diagnostics).
        """
        x = jnp.atleast_1d(x)
        cols = tuple(x[d : d + 1] for d in range(self.num_dims))
        return self._tau * self._basis.evaluate(*cols)[0]

    def compute_features(self, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the GP-scaled design matrix for a batch of inputs.

        ``X`` has shape ``(T,)`` or ``(T, num_dims)``; returns ``(T, num_features)``.
        """
        X = jnp.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        cols = tuple(X[:, d] for d in range(self.num_dims))
        return self._basis.compute_features(*cols) * self._tau[None, :]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TruncatedFourierBasis(max_freq={self.max_freq}, "
            f"num_dims={self.num_dims}, num_features={self.num_features})"
        )
