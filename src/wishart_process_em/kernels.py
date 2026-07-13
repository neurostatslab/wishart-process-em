r"""Spectral densities for the finite-basis Gaussian-process priors.

A stationary kernel :math:`k(x, x')` has a spectral density :math:`S(\omega)`
given by the Fourier transform of :math:`k` (Bochner's theorem).  The truncated
Fourier basis in :mod:`wishart_process_em.basis` weights each frequency feature
by :math:`\sqrt{S(\omega)}`, so choosing a spectral density is equivalent to
choosing a kernel / smoothness prior.

These constructors return callables mapping squared angular frequency
``|omega|**2`` to the density value, which is exactly the interface expected by
:class:`~wishart_process_em.basis.TruncatedFourierBasis`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

__all__ = ["squared_exponential", "matern", "SpectralDensity"]

SpectralDensity = Callable[[jnp.ndarray], jnp.ndarray]


def squared_exponential(lengthscale: float, variance: float = 1.0) -> SpectralDensity:
    r"""Spectral density of a squared-exponential (RBF) kernel.

    For :math:`k(r) = \text{variance} \cdot \exp(-r^2 / (2\ell^2))` the spectral
    density is proportional to :math:`\exp(-\ell^2 |\omega|^2 / 2)`.  Smaller
    ``lengthscale`` admits higher frequencies and yields rougher sample paths;
    this is the analogue of the bandwidth hyperparameter :math:`\lambda` in the
    Wishart-process paper.

    Parameters
    ----------
    lengthscale : float
        Length scale :math:`\ell > 0` controlling smoothness.
    variance : float, optional
        Marginal variance (overall amplitude) of the process.
    """
    if lengthscale <= 0:
        raise ValueError("lengthscale must be positive")
    ell2 = lengthscale**2

    def density(sq_omega: jnp.ndarray) -> jnp.ndarray:
        return variance * jnp.exp(-0.5 * ell2 * sq_omega)

    return density


def matern(lengthscale: float, nu: float = 1.5, variance: float = 1.0) -> SpectralDensity:
    r"""Spectral density of a Matern kernel with smoothness ``nu``.

    Uses the (unnormalised) isotropic Matern spectral density
    :math:`S(\omega) \propto (2\nu/\ell^2 + |\omega|^2)^{-(\nu + d/2)}`.  The
    dimensional term ``d/2`` is absorbed into an effective exponent; here we use
    the common 1-D form and let ``variance`` set the amplitude, which is
    adequate for a smoothness prior in weight space.

    Parameters
    ----------
    lengthscale : float
        Length scale :math:`\ell > 0`.
    nu : float, optional
        Smoothness. ``0.5`` is exponential (Ornstein-Uhlenbeck); larger is
        smoother; ``nu -> inf`` recovers the squared exponential.
    variance : float, optional
        Overall amplitude.
    """
    if lengthscale <= 0:
        raise ValueError("lengthscale must be positive")
    if nu <= 0:
        raise ValueError("nu must be positive")
    scale = 2.0 * nu / lengthscale**2
    exponent = nu + 0.5

    def density(sq_omega: jnp.ndarray) -> jnp.ndarray:
        return variance * (scale + sq_omega) ** (-exponent)

    return density
