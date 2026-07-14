# Quickstart walkthrough

This page walks through a complete workflow: simulate Gaussian data from a known
Wishart process, fit a model, predict smooth condition-dependent covariance, and
visualise it as covariance ellipses in PCA space. It then shows the Poisson
(spike-count) variant and its identifiability caveat.

For the mathematical background behind each step, see [the model page](model.md).

## 1. A smooth GP basis

Every latent function in the model is a weight-space GP built on a truncated
Fourier basis. Here we model a single **periodic** condition $x \in [0,1]$ (think
grating orientation mapped so that $0$ and $1$ coincide):

```python
import jax.numpy as jnp, jax.random as jxr
from wishart_process_em import (
    fourier_basis, WishartProcessModel, fit, squared_exponential,
)

# `fourier_basis` returns a nemos `FourierEval`; the GP smoothness is set later
# by the `spectral_density` passed to the model.
basis = fourier_basis(max_freq=8, num_dims=1)
```

A smaller `lengthscale` admits higher frequencies and yields rougher covariance
functions; `max_freq` caps the number of Fourier modes per input dimension. For
a rougher prior you can swap in `matern(lengthscale=0.2, nu=1.5)`.

## 2. Simulate Gaussian data

We instantiate a Gaussian Wishart process over $N = 25$ neurons with a rank
$P = 3$ covariance factor, then sample trials from its generative model. Sampling
draws a latent $v \sim \mathcal N(0, I_Q)$ per trial, forms $\eta = \mu(x) +
G(x)\,v$, and (for the Gaussian model) returns $y = \eta$:

```python
model = WishartProcessModel(basis, num_neurons=25, rank=3, likelihood="gaussian",
                            spectral_density=squared_exponential(lengthscale=0.2))

true_params = model.init_params(jxr.PRNGKey(0))
X = jnp.linspace(0, 1, 400)                       # 400 trials, condition in [0,1]
Y, _ = model.sample(jxr.PRNGKey(1), true_params, X)   # Y has shape (400, 25)
```

In a real analysis, `Y` is your trial-by-neuron response matrix (shape `(T, N)`)
and `X` your per-trial condition values, scaled into `[0, 1]`.

## 3. Fit the model

`fit` maximises the penalised marginal log-likelihood with Adam. For the Gaussian
likelihood the objective is exact (the latent integrates out), so no Monte-Carlo
is involved and the fit is deterministic:

```python
result = fit(model, Y, X, num_steps=500, learning_rate=1e-1)
```

`result` is a `FitResult` bundling the optimised `params`, the `loss_history`
(negative log-joint per step, handy for a convergence plot), the `model`, and —
for count models — the QMC `lattice`. Plot `result.loss_history` to confirm the
objective has plateaued.

## 4. Predict smooth covariance

`predict_cov` evaluates $\Sigma(x)$ on a grid of conditions. For the Gaussian
likelihood this is exactly the covariance of the responses:

```python
grid = jnp.linspace(0, 1, 50)
Sigma = model.predict_cov(result.params, grid)     # (50, 25, 25)
```

You can also predict the mean response with `model.predict_mean(result.params,
grid)`.

## 5. Plot covariance ellipses in PCA space

A covariance matrix over 25 neurons is hard to view directly. A standard
visualisation projects each condition's covariance into a shared 2-D PCA space
and draws it as an ellipse — the ellipse's axes are the eigenvectors of the
projected covariance, scaled by the square roots of its eigenvalues. Compute a
PCA basis once (e.g. from the neurons' grand covariance), project every
`Sigma[c]` into it, and draw one ellipse per condition, coloured by the
condition value, to see the covariance rotate and stretch smoothly across the
periodic axis:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# A shared 2-D projection: top-2 principal axes of the grand covariance.
Sigma = np.asarray(Sigma)
grand = Sigma.mean(axis=0)                          # (25, 25)
evals, evecs = np.linalg.eigh(grand)
W = evecs[:, -2:]                                   # (25, 2) projection

fig, ax = plt.subplots()
cmap = plt.get_cmap("twilight")                     # periodic colormap
for c, xc in enumerate(np.asarray(grid)):
    cov2 = W.T @ Sigma[c] @ W                        # (2, 2) projected covariance
    vals, vecs = np.linalg.eigh(cov2)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    width, height = 2.0 * np.sqrt(np.maximum(vals, 0.0))
    ax.add_patch(Ellipse((0, 0), width, height, angle=angle,
                         fill=False, edgecolor=cmap(xc), lw=1.2))
ax.set_aspect("equal")
ax.autoscale()
ax.set(xlabel="PC 1", ylabel="PC 2",
       title="Condition-dependent covariance (PCA space)")
plt.show()
```

The `[viz]` extra (`pip install "wishart-process-em[viz]"`) pulls in `matplotlib`
and `scikit-learn` for exactly this kind of plotting.

## 6. Spike counts: the Poisson variant

For raw spike counts, switch the likelihood to `"poisson"` (or
`"negative_binomial"`). The latent no longer integrates out analytically, so
`fit` builds a QMC lattice and estimates the marginal likelihood by
quasi-Monte-Carlo at every step — a stochastic-EM-style optimisation:

```python
counts_model = WishartProcessModel(basis, num_neurons=25, rank=3,
                                   likelihood="poisson",
                                   spectral_density=squared_exponential(lengthscale=0.2))
counts, _ = counts_model.sample(jxr.PRNGKey(2), counts_model.init_params(jxr.PRNGKey(3)), X)

result = fit(counts_model, counts, X, num_steps=1000)     # uses QMC each step
Sigma_latent = counts_model.predict_cov(result.params, grid)   # covariance of eta
```

For count models, `predict_cov` returns the covariance of the **linear
predictor** $\eta$, not of the counts themselves. For the covariance of the
observed counts (which also reflects the softplus link and the count noise), use
a Monte-Carlo estimate:

```python
obs_cov = counts_model.predict_observed_cov(jxr.PRNGKey(4), result.params, grid,
                                            num_samples=500)   # (50, 25, 25)
```

!!! warning "Identifiability of Poisson noise covariance"
    With a plain Poisson likelihood the count variance is tied to the mean
    ($\operatorname{Var} = \mathbb E[y]$), so there is little room for the latent
    Wishart covariance to express itself: the shared, condition-dependent
    covariance is only **well-identified when the data are meaningfully
    over-dispersed** (roughly $\operatorname{Var}/\operatorname{mean} \gtrsim
    1.5$). If your counts are close to Poisson, prefer the
    `"negative_binomial"` likelihood — its per-neuron dispersion absorbs the
    single-neuron over-dispersion and lets the Wishart process capture the
    genuine *shared* covariance structure. Always sanity-check the empirical
    Fano factor before interpreting a Poisson covariance fit.

## Where to go next

- [Model](model.md) — the math behind the basis, covariance family, and
  estimators.
- [API reference](api.md) — every public function and class.
