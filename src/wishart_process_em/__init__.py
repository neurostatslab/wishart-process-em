"""wishart_process_em: finite-basis Wishart processes for neural noise covariance.

Estimate how the trial-to-trial covariance of a neural population changes
smoothly across continuously-parameterised experimental conditions, using a
finite (Fourier-feature) Wishart process fit by marginal-likelihood
maximisation with quasi-Monte-Carlo + Laplace latent integration -- an
EM-style alternative to the variational inference of Nejatbakhsh, Garon &
Williams (2023).

Quickstart
----------
>>> import jax.numpy as jnp, jax.random as jxr
>>> from wishart_process_em import (
...     fourier_basis, WishartProcessModel, fit, squared_exponential)
>>> basis = fourier_basis(max_freq=8, num_dims=1)  # a nemos FourierEval
>>> model = WishartProcessModel(basis, num_neurons=10, rank=2,
...                             spectral_density=squared_exponential(0.2))
>>> params = model.init_params(jxr.PRNGKey(0))
>>> X = jnp.linspace(0, 1, 200)
>>> Y, _ = model.sample(jxr.PRNGKey(1), params, X)
>>> result = fit(model, Y, X, num_steps=200, progress=False)  # doctest: +SKIP
"""

from __future__ import annotations

import os as _os

import jax as _jax


def enable_x64() -> None:
    """Enable 64-bit floating point in JAX (required for numerical stability).

    The Wishart process routinely forms near-low-rank covariance matrices and
    takes their Cholesky factors and log-determinants; in 32-bit precision these
    are unstable and fitting can diverge.  This is called automatically on
    import unless the environment variable ``WISHART_PROCESS_EM_NO_X64`` is set.
    """
    _jax.config.update("jax_enable_x64", True)


if _os.environ.get("WISHART_PROCESS_EM_NO_X64", "").lower() not in ("1", "true", "yes"):
    enable_x64()

from .baselines import (
    ConditionCovarianceEstimator,
    gaussian_loglike,
    grand_empirical_covariance,
    ledoit_wolf_covariance,
)
from .basis import fourier_basis, fourier_feature_scale
from .covariance import WPParams
from .data import NeuralDataset, group_by_condition, scale_conditions
from .diagnostics import (
    covariance_operator_norm_error,
    fisher_information,
    fisher_information_curve,
    heldout_loglike,
    qda_accuracy,
    qda_predict,
)
from .fit import FitResult, fit, grand_covariance, init_bias_from_data
from .inference import (
    dataset_marginal_loglike,
    marginal_loglike_point_conjugate,
    marginal_loglike_point_laplace,
    marginal_loglike_point_qmc,
)
from .kernels import matern, squared_exponential
from .likelihoods import Gaussian, Likelihood, NegativeBinomial, Poisson, get_likelihood
from .model import WishartProcessModel
from .optim import NewtonResult, newton_minimize
from .qmc import QMCLattice

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "enable_x64",
    # basis / kernels
    "fourier_basis",
    "fourier_feature_scale",
    "squared_exponential",
    "matern",
    # model
    "WishartProcessModel",
    "WPParams",
    # likelihoods
    "Likelihood",
    "Gaussian",
    "Poisson",
    "NegativeBinomial",
    "get_likelihood",
    # inference / fitting
    "fit",
    "FitResult",
    "dataset_marginal_loglike",
    "marginal_loglike_point_conjugate",
    "marginal_loglike_point_qmc",
    "marginal_loglike_point_laplace",
    "init_bias_from_data",
    "grand_covariance",
    # numerics
    "QMCLattice",
    "newton_minimize",
    "NewtonResult",
    # data
    "NeuralDataset",
    "group_by_condition",
    "scale_conditions",
    # baselines
    "ConditionCovarianceEstimator",
    "grand_empirical_covariance",
    "ledoit_wolf_covariance",
    "gaussian_loglike",
    # diagnostics
    "heldout_loglike",
    "fisher_information",
    "fisher_information_curve",
    "qda_predict",
    "qda_accuracy",
    "covariance_operator_norm_error",
]
