r"""Finite (weight-space) basis approximations to Gaussian processes.

The Wishart process priors of Nejatbakhsh, Garon & Williams (2023) place
Gaussian-process priors on the mean function :math:`\mu(x)` and on the columns
of the covariance factor :math:`U(x)`.  Rather than working with a full kernel
matrix, this package approximates each scalar GP in *weight space* using a
truncated Fourier feature expansion:

.. math::

    f(x) \;=\; \sum_{m} w_m \, \phi_m(x),
    \qquad w_m \sim \mathcal N(0, 1),

where the (deterministic) features :math:`\phi_m` are sine/cosine functions
scaled by the square root of the kernel's spectral density.  With this scaling,
the induced prior over :math:`f` approximates a stationary GP whose smoothness is
controlled by the spectral density (see :mod:`wishart_process_em.kernels`).

Working in weight space turns the (otherwise nonparametric) GP into a finite,
differentiable parametric model: the free parameters are simply the weights
:math:`w`, on which we place a standard-normal prior.  This is what makes the
marginal-likelihood / EM style inference in this package tractable.

Inputs ``x`` are assumed to be scaled to the unit box ``[0, 1] ** num_dims``.
Because the features are Fourier modes on the torus, periodic conditions (e.g.
grating orientation, reach angle) wrap around naturally.  Non-periodic
conditions should be mapped into ``[0, 1]`` with a little headroom.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

__all__ = ["TruncatedFourierBasis"]


class TruncatedFourierBasis:
    r"""Truncated Fourier-feature basis for scalar Gaussian processes.

    A function drawn from this basis has the form

    .. math::

        f(x) = \sum_k \tau_k \big( w^{\sin}_k \sin(2\pi k \cdot x)
                                 + w^{\cos}_k \cos(2\pi k \cdot x) \big),

    where the integer frequency vectors :math:`k` range over
    ``{0, ..., max_freq - 1} ** num_dims`` (excluding the constant mode) and
    :math:`\tau_k = \sqrt{S(2\pi k)}` with :math:`S` the kernel spectral
    density.  Low-amplitude features (``tau`` below ``tol`` times the largest)
    are pruned for efficiency.

    Parameters
    ----------
    max_freq : int
        Number of integer frequencies enumerated *per input dimension*.
    num_dims : int
        Dimensionality of the input / condition space ``x``.
    spectral_density : Callable
        Maps squared angular frequency ``|omega|**2`` (shape ``(n_freq,)``) to
        the spectral density value.  See :mod:`wishart_process_em.kernels` for
        constructors (e.g. squared-exponential).  Larger values at high
        frequency yield rougher functions.
    tol : float, optional
        Relative threshold below which features are discarded.

    Attributes
    ----------
    frequencies : jnp.ndarray
        Angular frequency vectors, shape ``(n_freq, num_dims)``.
    tau : jnp.ndarray
        Per-frequency amplitude ``sqrt(spectral_density)``, shape ``(n_freq,)``.
    num_features : int
        Length ``M`` of the feature vector returned by :meth:`features`
        (equal to ``2 * n_freq``).
    """

    def __init__(
        self,
        max_freq: int,
        num_dims: int,
        spectral_density: Callable[[jnp.ndarray], jnp.ndarray],
        tol: float = 1e-5,
    ) -> None:
        if max_freq < 1:
            raise ValueError("max_freq must be >= 1")
        if num_dims < 1:
            raise ValueError("num_dims must be >= 1")

        # Enumerate integer frequency vectors on the grid, dropping the
        # all-zero (constant) mode which is handled by an explicit bias term
        # in the model.
        grid = np.stack(
            np.unravel_index(
                np.arange(max_freq**num_dims),
                shape=tuple(max_freq for _ in range(num_dims)),
            ),
            axis=1,
        )[1:]
        omega = 2.0 * np.pi * grid  # angular frequencies

        sq_norm = jnp.asarray(np.sum(omega**2, axis=1))
        tau = jnp.sqrt(jnp.asarray(spectral_density(sq_norm)))

        # Prune low-amplitude features.
        thresh = jnp.max(tau) * tol
        keep = np.asarray(tau > thresh)

        self.frequencies = jnp.asarray(omega[keep])
        self.tau = tau[keep]
        self.num_features = 2 * int(self.frequencies.shape[0])

        # Bookkeeping / config.
        self.max_freq = max_freq
        self.num_dims = num_dims
        self.spectral_density = spectral_density
        self.tol = tol

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the (fixed) feature vector at a single input ``x``.

        Parameters
        ----------
        x : jnp.ndarray
            A single input of shape ``(num_dims,)`` (or a scalar when
            ``num_dims == 1``).

        Returns
        -------
        jnp.ndarray
            Feature vector of shape ``(num_features,)``.
        """
        x = jnp.atleast_1d(x)
        u = self.frequencies @ x  # (n_freq,)
        return jnp.concatenate([self.tau * jnp.sin(u), self.tau * jnp.cos(u)])

    def evaluate(self, weights: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        r"""Evaluate ``sum_m weights_m * features_m(x)`` for a single ``x``.

        ``weights`` has shape ``(num_features,)``.  Under the standard-normal
        prior on ``weights``, ``evaluate`` is a draw from the approximate GP.
        """
        return weights @ self.features(x)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TruncatedFourierBasis(max_freq={self.max_freq}, "
            f"num_dims={self.num_dims}, num_features={self.num_features})"
        )
