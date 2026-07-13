# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Fourier basis now backed by [nemos](https://github.com/flatironinstitute/nemos).**
  `TruncatedFourierBasis` delegates frequency enumeration and sine/cosine
  evaluation to `nemos.basis.FourierEval`, keeping only the Wishart-process
  Gaussian-process spectral scaling. The public constructor and behaviour are
  unchanged. Adds a `nemos>=0.2.9` dependency.
- **Python requirement raised to ≥ 3.12** (to match nemos); dropped 3.10 / 3.11.
- Unified the two grand-empirical-covariance implementations
  (`fit.grand_covariance` now reuses `baselines.grand_empirical_covariance`).

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
