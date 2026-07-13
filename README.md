# wishart-process-em

**Finite-basis Wishart process models of neural noise covariance, fit by marginal-likelihood (QMC + Laplace) EM.**

Estimate how the *trial-to-trial* covariance of a neural population changes
smoothly across continuously-parameterised experimental conditions (grating
orientation, reach angle, tone frequency, …), pooling statistical power from
neighbouring conditions so that accurate covariance estimates are possible even
with **few trials per condition**.

This package is a self-contained [JAX](https://github.com/google/jax)
implementation of the Wishart process model of
[Nejatbakhsh, Garon & Williams (2023; NeurIPS)](https://arxiv.org/abs/2308.11824), with
two deliberate differences from the reference implementation:

1. **Finite (Fourier-feature) basis.** Each latent Gaussian process is
   represented in *weight space* by a truncated Fourier expansion, turning the
   nonparametric GP into a finite, differentiable model.
2. **Marginal-likelihood / EM-style inference instead of variational inference.**
   Per-trial latents are integrated out with a randomised **quasi-Monte-Carlo**
   lattice and an optional **Laplace-approximation** importance-sampling
   correction; the marginal likelihood is then maximised with `optax`.

It supports **Gaussian, Poisson, and negative-binomial** observations and the
full covariance family from the paper, `Σ(x) = L (U(x)U(x)ᵀ + Λ(x)) Lᵀ`.

---

## Installation

```bash
pip install wishart-process-em            # from PyPI
# or, from source:
git clone https://github.com/neurostatslab/wishart-process-em
cd wishart-process-em
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. For plotting and baselines, install the extras:
`pip install "wishart-process-em[viz]"`.

## Quickstart

```python
import jax.numpy as jnp, jax.random as jxr
from wishart_process_em import (
    TruncatedFourierBasis, WishartProcessModel, fit, squared_exponential,
)

# 1. A smooth GP basis over a 1-D periodic condition (e.g. orientation).
basis = TruncatedFourierBasis(max_freq=8, num_dims=1,
                              spectral_density=squared_exponential(lengthscale=0.2))

# 2. A Gaussian Wishart process over N=25 neurons, rank P=3.
model = WishartProcessModel(basis, num_neurons=25, rank=3, likelihood="gaussian")

# 3. Simulate some data from the generative model.
true_params = model.init_params(jxr.PRNGKey(0))
X = jnp.linspace(0, 1, 400)                       # 400 trials, condition in [0,1]
Y, _ = model.sample(jxr.PRNGKey(1), true_params, X)

# 4. Fit and predict smooth condition-dependent covariance.
result = fit(model, Y, X, num_steps=500, learning_rate=1e-1)
Sigma = model.predict_cov(result.params, jnp.linspace(0, 1, 50))   # (50, N, N)
```

For **spike counts**, pass `likelihood="poisson"` (or `"negative_binomial"`):
the latent integral is then estimated by QMC automatically.

```python
model = WishartProcessModel(basis, num_neurons=25, rank=3, likelihood="poisson")
result = fit(model, counts, X, num_steps=1000)          # uses QMC each step
```

## Command line

```bash
# Fit a model to data stored in an .npz with arrays `Y` (T,N) and `X` (T,D):
wishart-em fit data.npz --likelihood poisson --rank 3 --steps 1000 -o fit.npz

# Evaluate held-out log-likelihood of a saved fit:
wishart-em evaluate fit.npz heldout.npz

# Sample synthetic data from a fitted model:
wishart-em sample fit.npz --trials 500 -o synthetic.npz
```

See `wishart-em --help` and the [documentation](https://neurostatslab.github.io/wishart-process-em).

## How it works

| Component | Module | Notes |
|---|---|---|
| Weight-space GP basis | `basis`, `kernels` | Truncated Fourier features scaled by a kernel spectral density. |
| QMC integration | `qmc` | Randomised Korobov lattice → Gaussian latents. |
| Laplace approximation | `optim` | JAX-traceable damped Newton returning the Hessian Cholesky. |
| Observation models | `likelihoods` | Gaussian (conjugate), Poisson, negative-binomial. |
| Covariance `Σ(x)=L(UUᵀ+Λ)Lᵀ` | `covariance`, `model` | Rank-`P` factor, optional diagonal, optional scale matrix. |
| Marginal likelihood | `inference` | Conjugate / QMC / Laplace estimators. |
| Fitting | `fit` | `optax`-based, stochastic-EM for count models. |
| Baselines & metrics | `baselines`, `diagnostics` | Ledoit-Wolf, grand-empirical; held-out LL, Fisher information, QDA. |

See [`docs/model.md`](docs/model.md) for the full mathematical description.

## Citation

If you use this software, please cite the paper:

```bibtex
@inproceedings{nejatbakhsh2023wishart,
  title     = {Estimating Noise Correlations Across Continuous Conditions With Wishart Processes},
  author    = {Nejatbakhsh, Amin and Garon, Isabel and Williams, Alex H.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
}
```

## License

MIT — see [LICENSE](LICENSE).
