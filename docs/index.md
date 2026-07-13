# wishart-process-em

**Finite-basis Wishart process models of neural noise covariance, fit by
marginal-likelihood (QMC + Laplace) EM.**

`wishart-process-em` estimates how the *trial-to-trial* covariance of a neural
population changes smoothly across continuously-parameterised experimental
conditions (grating orientation, reach angle, tone frequency, …). By pooling
statistical power from neighbouring conditions, it produces accurate covariance
estimates even in the challenging **few-trials-per-condition** regime.

It is a self-contained [JAX](https://github.com/google/jax) reimplementation of
the Wishart process model of
[Nejatbakhsh, Garon & Williams (2023)](https://arxiv.org/abs/2308.11824), with
two deliberate differences from the reference implementation:

1. **Finite (Fourier-feature) basis.** Each latent Gaussian process is
   represented in *weight space* by a truncated Fourier expansion, turning the
   nonparametric GP into a finite, differentiable model.
2. **Marginal-likelihood / EM-style inference instead of variational
   inference.** Per-trial latents are integrated out with a randomised
   **quasi-Monte-Carlo** (Korobov) lattice and an optional
   **Laplace-approximation** importance-sampling correction; the marginal
   likelihood is then maximised with [`optax`](https://github.com/google-deepmind/optax).

It supports **Gaussian, Poisson, and negative-binomial** observations and the
full covariance family from the paper,
$\Sigma(x) = L\,(U(x)U(x)^\top + \Lambda(x))\,L^\top$.

## Installation

```bash
pip install wishart-process-em            # from PyPI
# or, from source:
git clone https://github.com/neurostatslab/wishart-process-em
cd wishart-process-em
pip install -e ".[dev]"
```

Requires Python ≥ 3.12 (following nemos). For plotting (and the comparison
baselines in `examples/`), install the extras:
`pip install "wishart-process-em[viz]"`.

## Quickstart

```python
import jax.numpy as jnp, jax.random as jxr
from wishart_process_em import (
    fourier_basis, WishartProcessModel, fit, squared_exponential,
)

# A Fourier basis (a nemos FourierEval) over a 1-D periodic condition.
basis = fourier_basis(max_freq=8, num_dims=1)

# A Gaussian Wishart process over N=25 neurons, rank P=3. The spectral density
# sets the GP smoothness across conditions.
model = WishartProcessModel(basis, num_neurons=25, rank=3, likelihood="gaussian",
                            spectral_density=squared_exponential(lengthscale=0.2))

# Simulate some data from the generative model.
true_params = model.init_params(jxr.PRNGKey(0))
X = jnp.linspace(0, 1, 400)                       # 400 trials, condition in [0,1]
Y, _ = model.sample(jxr.PRNGKey(1), true_params, X)

# Fit and predict smooth condition-dependent covariance.
result = fit(model, Y, X, num_steps=500, learning_rate=1e-1)
Sigma = model.predict_cov(result.params, jnp.linspace(0, 1, 50))   # (50, N, N)
```

For **spike counts**, pass `likelihood="poisson"` (or `"negative_binomial"`);
the latent integral is then estimated by QMC automatically.

## Documentation

- [Model](model.md) — the full mathematical description: generative model,
  weight-space GP basis, covariance family, likelihoods, and the three
  marginal-likelihood estimators.
- [Quickstart](quickstart.md) — a longer walkthrough: simulate, fit, predict
  covariance, plot covariance ellipses in PCA space, and the Poisson variant.
- [Command line](cli.md) — the `wishart-em` CLI for fitting, evaluating, and
  sampling from the shell.
- [API reference](api.md) — every public symbol, documented from the source.

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

MIT — see the [LICENSE](https://github.com/neurostatslab/wishart-process-em/blob/main/LICENSE) file.
