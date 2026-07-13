r"""High-level fitting driver.

Maximises the (penalised) marginal log-likelihood

.. math::

    \mathcal L(\theta) \;=\; \sum_{c,k} \log p(y_{ck} \mid x_c, \theta)
                             \;+\; \log p(\theta),

with respect to all model parameters using ``optax``.  For the Gaussian model
the objective is deterministic (the latent integrates out analytically); for
count models each step uses a freshly randomised quasi-Monte-Carlo estimate of
the marginal likelihood, giving a stochastic-EM style optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as jxr
import numpy as np
import optax
from tqdm.auto import trange

from .covariance import WPParams, softplus_inverse
from .data import group_by_condition
from .inference import dataset_marginal_loglike
from .model import WishartProcessModel
from .qmc import QMCLattice

__all__ = ["FitResult", "fit", "init_bias_from_data", "grand_covariance"]


@dataclass
class FitResult:
    """Container for the outcome of :func:`fit`.

    Attributes
    ----------
    params : WPParams
        The optimised parameters.
    loss_history : np.ndarray
        Objective (negative log-joint) at each optimisation step.
    model : WishartProcessModel
        The model that was fit (for convenient prediction).
    lattice : QMCLattice or None
        The QMC lattice used (``None`` for Gaussian models).
    init_params : WPParams
        The parameters before optimisation.
    """

    params: WPParams
    loss_history: np.ndarray
    model: WishartProcessModel
    lattice: object = None
    init_params: WPParams = field(default=None, repr=False)


def init_bias_from_data(model: WishartProcessModel, Y: jnp.ndarray) -> jnp.ndarray:
    """Initialise the per-neuron mean bias from the data's marginal mean.

    For Gaussian models this is the empirical mean; for count models it is the
    inverse-softplus of the empirical mean (so ``softplus(bias) ~ mean count``).
    """
    y_mean = jnp.mean(jnp.asarray(Y), axis=0)
    if model.likelihood.conjugate_gaussian:
        return y_mean
    return softplus_inverse(jnp.maximum(y_mean, 1e-3))


def grand_covariance(Y: jnp.ndarray, X: jnp.ndarray | None = None) -> jnp.ndarray:
    """Grand empirical covariance (pooled across conditions).

    If per-trial conditions ``X`` are given, trials are centred within each
    unique condition before pooling (eq. 2 of the paper); otherwise the global
    covariance is returned.
    """
    Y = np.asarray(Y)
    if X is None:
        return jnp.asarray(np.cov(Y, rowvar=False))
    # Pool within-condition-centred trials (eq. 2 of the paper).
    _, groups = group_by_condition(Y, X)
    centered = np.concatenate([g - g.mean(axis=0, keepdims=True) for g in groups])
    return jnp.asarray(np.cov(centered, rowvar=False, bias=True))


def fit(
    model: WishartProcessModel,
    Y: jnp.ndarray,
    X: jnp.ndarray,
    num_steps: int = 1000,
    learning_rate: float = 1e-1,
    seed: int = 0,
    params0: WPParams | None = None,
    lattice: QMCLattice | None = None,
    num_qmc_points: int = 151,
    method: str = "auto",
    optimizer: optax.GradientTransformation | None = None,
    init_bias: bool = True,
    init_scale_from_grand_cov: bool = True,
    progress: bool = True,
) -> FitResult:
    r"""Fit a :class:`WishartProcessModel` by marginal-likelihood maximisation.

    Parameters
    ----------
    model : WishartProcessModel
    Y : array, shape (T, N)
        Trial-by-neuron observations.
    X : array, shape (T, D) or (T,)
        Per-trial condition values, scaled into ``[0, 1]``.
    num_steps : int
        Number of optimisation steps.
    learning_rate : float
        Adam step size (ignored if ``optimizer`` is given).
    seed : int
        Base PRNG seed (used to randomise the QMC estimate each step).
    params0 : WPParams, optional
        Initial parameters; if ``None`` they are initialised from ``model`` and
        the data.
    lattice : QMCLattice, optional
        Lattice for count models; built automatically if needed.
    num_qmc_points : int
        Number of lattice points when building a lattice automatically.
    method : {"auto", "conjugate", "qmc", "laplace"}
        Marginal-likelihood estimator (see
        :mod:`wishart_process_em.inference`).
    optimizer : optax.GradientTransformation, optional
        Custom optimiser; defaults to ``optax.adam(learning_rate)``.
    init_bias : bool
        Initialise the mean bias from the data (see :func:`init_bias_from_data`).
    init_scale_from_grand_cov : bool
        When ``use_scale`` is enabled, initialise ``L`` from the Cholesky of the
        grand empirical covariance.
    progress : bool
        Show a progress bar.

    Returns
    -------
    FitResult
    """
    Y = jnp.asarray(Y)
    X = jnp.asarray(X)
    X2 = X[:, None] if X.ndim == 1 else X

    conjugate = model.likelihood.conjugate_gaussian
    if lattice is None and not conjugate:
        lattice = QMCLattice(num_qmc_points, model.latent_dim, seed=seed)

    # ----- Initialise parameters -----------------------------------------
    if params0 is None:
        key = jxr.PRNGKey(seed)
        mean_bias = init_bias_from_data(model, Y) if init_bias else None
        scale_tril = None
        if model.use_scale and init_scale_from_grand_cov:
            gc = grand_covariance(Y, X2)
            n = model.num_neurons
            scale_tril = jnp.linalg.cholesky(gc + model.jitter * jnp.eye(n))
        params0 = model.init_params(key, mean_bias=mean_bias, scale_tril=scale_tril)

    # ----- Objective ------------------------------------------------------
    def objective(params, key):
        ll = dataset_marginal_loglike(
            model, params, Y, X2, key=key, lattice=lattice, method=method
        )
        return -(ll + model.log_prior(params))

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    optimizer = optax.adam(learning_rate) if optimizer is None else optimizer
    opt_state = optimizer.init(params0)

    params = params0
    losses = np.empty(num_steps)
    iterator = trange(num_steps, disable=not progress, desc="fit")
    for step in iterator:
        # For the conjugate objective the key is unused; for QMC it randomises
        # the lattice each step.
        loss, grads = value_and_grad(params, jxr.PRNGKey(seed + step + 1))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses[step] = float(loss)
        if progress and step % max(1, num_steps // 100) == 0:
            iterator.set_postfix(loss=f"{losses[step]:.1f}")

    return FitResult(
        params=params,
        loss_history=losses,
        model=model,
        lattice=lattice,
        init_params=params0,
    )
