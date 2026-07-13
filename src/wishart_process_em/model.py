r"""The finite-basis Wishart process model.

This module ties together the pieces of the package into a single
:class:`WishartProcessModel`:

* a nemos basis (e.g. :class:`nemos.basis.FourierEval`) with an optional
  Gaussian-process spectral scaling, giving weight-space GP priors on the mean
  function and covariance factors;
* the covariance assembly of :mod:`wishart_process_em.covariance`
  (:math:`\Sigma(x) = L(U U^\top + \Lambda)L^\top`);
* an observation model from :mod:`wishart_process_em.likelihoods`.

The generative model for a trial at condition ``x`` is

.. math::

    v \sim N(0, I_Q), \quad
    \eta = \mu(x) + G(x) v, \quad
    y \sim p(\,\cdot \mid \eta),

with :math:`G(x) G(x)^\top = \Sigma(x)`.  For the Gaussian likelihood the latent
``v`` integrates out and ``y ~ N(mu, Sigma)`` exactly; for count likelihoods the
integral is estimated with quasi-Monte-Carlo (see
:mod:`wishart_process_em.inference`).

All heavy methods operate on a single trial and are vectorised over trials with
``jax.vmap``; the public ``predict_*`` / ``batched_*`` helpers do this for you.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jxr

from .basis import fourier_feature_scale
from .covariance import (
    WPParams,
    scale_tril_from_raw,
    softplus_inverse,
)
from .kernels import SpectralDensity
from .likelihoods import Likelihood, get_likelihood

__all__ = ["WishartProcessModel"]


def _as_conditions(X: jnp.ndarray) -> jnp.ndarray:
    """Coerce conditions to shape ``(T, D)``; a 1-D array is treated as ``D=1``."""
    X = jnp.asarray(X)
    return X[:, None] if X.ndim == 1 else X


class WishartProcessModel:
    r"""Finite-basis Wishart process over smoothly-parameterised conditions.

    Parameters
    ----------
    basis : nemos basis
        A nemos evaluation basis (e.g. :class:`nemos.basis.FourierEval`, most
        easily built with :func:`wishart_process_em.fourier_basis`) shared by the
        mean and covariance functions.  Its features span the condition space.
    num_neurons : int
        Number of neurons ``N``.
    rank : int, optional
        Rank ``P`` of the low-rank covariance factor ``U``.  May be ``0`` if
        ``use_diagonal=True`` (a purely diagonal / heteroscedastic model).
    likelihood : str or Likelihood, optional
        Observation model: ``"gaussian"``, ``"poisson"`` or
        ``"negative_binomial"`` (or a :class:`Likelihood` instance).
    use_diagonal : bool, optional
        Include the non-negative diagonal term :math:`\Lambda(x)` (the ``WPlrd``
        extension).  This raises the latent dimension to ``P + N`` for count
        models, so keep it off (default) when relying on low-dimensional QMC.
    use_scale : bool, optional
        Include the learnable lower-triangular scale matrix ``L`` (initialise it
        from the grand empirical covariance via :meth:`init_params`).
    jitter : float, optional
        Small value added to covariance diagonals for numerical stability.
    init_weight_scale : float, optional
        Standard deviation for random initialisation of the GP weights.
    prior_weight_scale : float, optional
        Standard deviation of the (Gaussian) prior on the GP weights; ``1.0``
        corresponds to the standard weight-space GP prior.
    spectral_density : Callable, optional
        Spectral density of the desired GP kernel (see
        :mod:`wishart_process_em.kernels`).  When given, the Fourier features are
        scaled by :math:`\sqrt{S(2\pi k)}` so that the weight-space prior is a GP
        with that kernel -- this is what controls smoothness across conditions.
        Requires a Fourier ``basis``.  If ``None``, features are used unscaled
        (a ridge prior on the raw features).
    """

    def __init__(
        self,
        basis,
        num_neurons: int,
        rank: int = 2,
        likelihood: str | Likelihood = "gaussian",
        use_diagonal: bool = False,
        use_scale: bool = False,
        jitter: float = 1e-5,
        init_weight_scale: float = 1.0,
        prior_weight_scale: float = 1.0,
        spectral_density: SpectralDensity | None = None,
    ) -> None:
        if rank < 0:
            raise ValueError("rank P must be >= 0")
        if rank == 0 and not use_diagonal:
            raise ValueError("rank=0 requires use_diagonal=True (else Sigma is zero)")

        self.basis = basis
        self.num_dims = int(basis.ndim)
        self.num_features = int(basis.n_basis_funcs)
        self.spectral_density = spectral_density
        self._feature_scale = (
            fourier_feature_scale(basis, spectral_density)
            if spectral_density is not None
            else jnp.ones(self.num_features)
        )

        self.num_neurons = int(num_neurons)
        self.rank = int(rank)
        self.likelihood: Likelihood = get_likelihood(likelihood)
        self.use_diagonal = bool(use_diagonal)
        self.use_scale = bool(use_scale)
        self.jitter = float(jitter)
        self.init_weight_scale = float(init_weight_scale)
        self.prior_weight_scale = float(prior_weight_scale)

    def _features(self, x: jnp.ndarray) -> jnp.ndarray:
        """GP-scaled basis features at a single input ``x``, shape ``(M,)``.

        Delegates evaluation to the nemos basis and applies the spectral scaling.
        Traceable, so it composes with :func:`jax.vmap` / :func:`jax.grad`.
        """
        x = jnp.atleast_1d(x)
        cols = tuple(x[d : d + 1] for d in range(self.num_dims))
        return self._feature_scale * self.basis.evaluate(*cols)[0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def latent_dim(self) -> int:
        """Dimension ``Q`` of the per-trial latent ``v`` to be integrated."""
        return self.rank + (self.num_neurons if self.use_diagonal else 0)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_params(
        self,
        key: jax.Array,
        mean_bias: jnp.ndarray | None = None,
        scale_tril: jnp.ndarray | None = None,
    ) -> WPParams:
        """Randomly initialise model parameters.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        mean_bias : jnp.ndarray, optional
            Per-neuron mean bias ``(N,)`` (e.g. inverse-link of the empirical
            mean response).  Defaults to zeros.
        scale_tril : jnp.ndarray, optional
            Lower-triangular initialiser for the scale matrix ``L`` (e.g. the
            Cholesky factor of the grand empirical covariance).  Only used when
            ``use_scale=True``; defaults to the identity.
        """
        n, p, m = self.num_neurons, self.rank, self.num_features
        k_mean, k_fac, k_diag, k_lik = jxr.split(key, 4)
        s = self.init_weight_scale

        mean_w = s * jxr.normal(k_mean, (n, m))
        mean_b = jnp.zeros(n) if mean_bias is None else jnp.asarray(mean_bias)
        factor_w = s * jxr.normal(k_fac, (n, p, m))
        log_latent_scale = jnp.full((p,), softplus_inverse(jnp.asarray(1.0)))

        diag_w = s * jxr.normal(k_diag, (n, m)) if self.use_diagonal else None

        scale_raw = None
        if self.use_scale:
            from .covariance import raw_from_scale_tril

            init_tril = jnp.eye(n) if scale_tril is None else jnp.asarray(scale_tril)
            scale_raw = raw_from_scale_tril(init_tril)

        lik = self.likelihood.init_params(n, k_lik)
        return WPParams(mean_w, mean_b, factor_w, log_latent_scale, diag_w, scale_raw, lik)

    # ------------------------------------------------------------------
    # Single-condition building blocks (vectorise with vmap)
    # ------------------------------------------------------------------
    def mean_at(self, params: WPParams, x: jnp.ndarray) -> jnp.ndarray:
        """Latent mean predictor ``mu(x)``, shape ``(N,)``."""
        phi = self._features(x)
        return params.mean_w @ phi + params.mean_b

    def factor_at(self, params: WPParams, x: jnp.ndarray) -> jnp.ndarray:
        """Low-rank covariance factor ``U(x) = A(x) diag(sZ)``, shape ``(N, P)``."""
        phi = self._features(x)
        a = jnp.einsum("npm,m->np", params.factor_w, phi)  # (N, P)
        sz = jax.nn.softplus(params.log_latent_scale)  # (P,)
        return a * sz[None, :]

    def diag_at(self, params: WPParams, x: jnp.ndarray) -> jnp.ndarray:
        """Non-negative diagonal ``lambda(x)``, shape ``(N,)`` (zeros if disabled)."""
        if not self.use_diagonal:
            return jnp.zeros(self.num_neurons)
        phi = self._features(x)
        return jax.nn.softplus(params.diag_w @ phi)

    def scale_tril(self, params: WPParams) -> jnp.ndarray:
        """Scale matrix ``L`` (identity if ``use_scale=False``), shape ``(N, N)``."""
        if not self.use_scale:
            return jnp.eye(self.num_neurons)
        return scale_tril_from_raw(params.scale_raw)

    def sqrt_factor_at(self, params: WPParams, x: jnp.ndarray) -> jnp.ndarray:
        r"""Covariance square root ``G(x)`` with ``G G^T = Sigma(x)``.

        Shape ``(N, Q)`` where ``Q = P`` (or ``P + N`` with the diagonal term).
        The latent predictor is ``eta = mu(x) + G(x) v``, ``v ~ N(0, I_Q)``.
        """
        u = self.factor_at(params, x)  # (N, P)
        L = self.scale_tril(params)
        if self.use_diagonal:
            d = jnp.diag(jnp.sqrt(self.diag_at(params, x)))  # (N, N)
            b = jnp.concatenate([u, d], axis=1)  # (N, P + N)
        else:
            b = u
        return L @ b

    def cov_at(self, params: WPParams, x: jnp.ndarray) -> jnp.ndarray:
        """Latent covariance ``Sigma(x) = G G^T + jitter I``, shape ``(N, N)``."""
        g = self.sqrt_factor_at(params, x)
        return g @ g.T + self.jitter * jnp.eye(self.num_neurons)

    # ------------------------------------------------------------------
    # Batched predictions
    # ------------------------------------------------------------------
    def predict_mean_latent(self, params: WPParams, X: jnp.ndarray) -> jnp.ndarray:
        """Latent mean ``mu(x)`` for each condition, shape ``(T, N)``."""
        return jax.vmap(lambda x: self.mean_at(params, x))(_as_conditions(X))

    def predict_mean(self, params: WPParams, X: jnp.ndarray) -> jnp.ndarray:
        """Expected observation ``E[y|x] = link(mu(x))``, shape ``(T, N)``."""
        mu = self.predict_mean_latent(params, X)
        return jax.vmap(lambda e: self.likelihood.mean(params.lik, e))(mu)

    def predict_cov(self, params: WPParams, X: jnp.ndarray) -> jnp.ndarray:
        """Latent covariance ``Sigma(x)`` for each condition, shape ``(T, N, N)``.

        For the Gaussian likelihood this is exactly the covariance of ``y``.  For
        count likelihoods it is the covariance of the *linear predictor*; use
        :meth:`predict_observed_cov` for the covariance of the counts themselves.
        """
        return jax.vmap(lambda x: self.cov_at(params, x))(_as_conditions(X))

    def predict_observed_cov(
        self, key: jax.Array, params: WPParams, X: jnp.ndarray, num_samples: int = 500
    ) -> jnp.ndarray:
        """Monte-Carlo covariance of the *observations* ``y`` at each condition.

        Necessary for non-Gaussian likelihoods, where the observation covariance
        differs from ``Sigma`` because of the link and the count noise.
        """
        X = _as_conditions(X)
        keys = jxr.split(key, num_samples)
        # (num_samples, T, N)
        samples = jax.vmap(lambda k: self.sample(k, params, X)[0])(keys)
        # covariance across samples, per condition.
        return jax.vmap(lambda s: jnp.cov(s, rowvar=False), in_axes=1)(samples)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(
        self, key: jax.Array, params: WPParams, X: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample observations for a batch of conditions.

        Returns ``(y, eta)`` with shapes ``(T, N)`` each, where ``eta`` is the
        latent linear predictor.
        """
        X = _as_conditions(X)
        t = X.shape[0]
        k_lat, k_obs = jxr.split(key)
        v = jxr.normal(k_lat, (t, self.latent_dim))

        def one(x, vi, ko):
            eta = self.mean_at(params, x) + self.sqrt_factor_at(params, x) @ vi
            y = self.likelihood.observe(ko, params.lik, eta)
            return y, eta

        obs_keys = jxr.split(k_obs, t)
        return jax.vmap(one)(X, v, obs_keys)

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------
    def log_prior(self, params: WPParams) -> jnp.ndarray:
        r"""Log-prior over parameters (the weight-space GP prior + likelihood).

        A standard-normal prior on the GP weights corresponds to the GP prior on
        the mean and covariance-factor functions.
        """
        scale = self.prior_weight_scale

        def normal_lp(w):
            return jnp.sum(jax.scipy.stats.norm.logpdf(w, 0.0, scale))

        lp = normal_lp(params.mean_w) + normal_lp(params.factor_w)
        if self.use_diagonal:
            lp = lp + normal_lp(params.diag_w)
        lp = lp + self.likelihood.log_prior(params.lik)
        return lp

    # ------------------------------------------------------------------
    # Marginal likelihood (delegated to inference module)
    # ------------------------------------------------------------------
    def marginal_loglike(
        self,
        params: WPParams,
        Y: jnp.ndarray,
        X: jnp.ndarray,
        key: jax.Array | None = None,
        lattice=None,
        method: str = "auto",
    ) -> jnp.ndarray:
        """Sum of per-trial marginal log-likelihoods over a dataset.

        Thin wrapper around :func:`wishart_process_em.inference.dataset_marginal_loglike`.
        """
        from .inference import dataset_marginal_loglike

        return dataset_marginal_loglike(
            self, params, Y, X, key=key, lattice=lattice, method=method
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"WishartProcessModel(N={self.num_neurons}, P={self.rank}, "
            f"likelihood={type(self.likelihood).__name__}, "
            f"use_diagonal={self.use_diagonal}, use_scale={self.use_scale}, "
            f"latent_dim={self.latent_dim})"
        )
