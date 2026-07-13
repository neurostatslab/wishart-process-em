# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Ledoit-Wolf (and the regularised baselines) are no longer spuriously
  singular.** For very small per-condition samples, sklearn's Ledoit-Wolf
  shrinkage can collapse to ~0 and return the singular empirical covariance,
  giving `-inf` held-out log-likelihood. `ledoit_wolf_covariance` now guarantees
  a positive-definite result by clipping the smallest eigenvalue to a tiny
  fraction of the largest (a no-op when already well-conditioned).

### Changed

- `NeuralDataset.train_test_split` now **stratifies by condition** by default
  (`stratify=True`): each condition contributes `round(test_frac * K_c)` test
  trials, so no condition is starved of training trials. Pass `stratify=False`
  for the previous global random split.

### Removed

- **`TruncatedFourierBasis` class removed.** Models now consume a
  [nemos](https://github.com/flatironinstitute/nemos) basis object directly
  (e.g. `nemos.basis.FourierEval`).

### Added

- `fourier_basis(max_freq, num_dims=1, bounds=(0, 1))` — a thin factory returning
  a nemos `FourierEval` with this package's defaults (unit-box bounds, no
  intercept term).
- `fourier_feature_scale(basis, spectral_density)` — the Gaussian-process
  spectral scaling, exposed as a standalone helper.
- `WishartProcessModel` gains a `spectral_density=` argument; the GP smoothness
  scaling is applied inside the model, so any nemos evaluation basis can be used
  (Fourier bases get the GP scaling; others use a ridge prior on raw features).

### Changed

- Fourier basis functions are now provided entirely by nemos (frequency
  enumeration, N-D Cartesian product, sine/cosine evaluation). Adds a
  `nemos>=0.2.9` dependency.
- **Python requirement raised to ≥ 3.12** (to match nemos); dropped 3.10 / 3.11.
- Unified the two grand-empirical-covariance implementations
  (`fit.grand_covariance` now reuses `baselines.grand_empirical_covariance`).

### Note

- The old `tol` frequency-pruning option is gone; control the number of Fourier
  modes with `max_freq` instead.

## [0.1.0] - 2026-07-13

Initial release.

### Added

- **Finite-basis Wishart process model** (`WishartProcessModel`): smooth,
  condition-dependent neural noise covariance built on weight-space Gaussian
  processes with a truncated Fourier feature basis (`TruncatedFourierBasis`) and
  squared-exponential / Matérn spectral densities.
- **Observation models**: Gaussian (conjugate, exact marginal), Poisson
  (softplus link), and negative-binomial (per-neuron dispersion) likelihoods.
- **Full covariance family** $\Sigma(x) = L\,(U U^\top + \Lambda)\,L^\top$:
  low-rank factor of rank $P$ with per-latent ARD scaling, optional non-negative
  diagonal $\Lambda$, and an optional learnable lower-triangular scale matrix
  $L$.
- **Marginal-likelihood inference**: conjugate, quasi-Monte-Carlo (randomised
  Korobov lattice), and Laplace-approximation multiple-importance-sampling
  estimators, with an `optax`-based, stochastic-EM fitting driver (`fit`).
- **Baselines**: Ledoit-Wolf and grand-empirical covariance estimators for
  comparison.
- **Diagnostics**: held-out marginal log-likelihood, Fisher information, and
  quadratic discriminant analysis (QDA) metrics.
- **Command-line interface** (`wishart-em`) with `fit`, `evaluate`, and `sample`
  subcommands operating on `.npz` datasets.
- **Test suite** covering the model, basis, kernels, likelihoods, inference,
  QMC, and optimisation utilities.
- **Documentation** (MkDocs Material): landing page, full mathematical model
  description, quickstart walkthrough, CLI guide, and an auto-generated API
  reference.

[0.1.0]: https://github.com/neurostatslab/wishart-process-em/releases/tag/v0.1.0
