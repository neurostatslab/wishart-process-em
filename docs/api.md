# API reference

This page documents every public symbol exported by `wishart_process_em`
(everything in `wishart_process_em.__all__`), grouped by area. The docstrings are
rendered directly from the source with
[mkdocstrings](https://mkdocstrings.github.io/).

## Model

::: wishart_process_em.WishartProcessModel

::: wishart_process_em.WPParams

## Basis and kernels

::: wishart_process_em.TruncatedFourierBasis

::: wishart_process_em.squared_exponential

::: wishart_process_em.matern

## Likelihoods

::: wishart_process_em.Likelihood

::: wishart_process_em.Gaussian

::: wishart_process_em.Poisson

::: wishart_process_em.NegativeBinomial

::: wishart_process_em.get_likelihood

## Inference and fitting

::: wishart_process_em.fit

::: wishart_process_em.FitResult

::: wishart_process_em.dataset_marginal_loglike

::: wishart_process_em.marginal_loglike_point_conjugate

::: wishart_process_em.marginal_loglike_point_qmc

::: wishart_process_em.marginal_loglike_point_laplace

::: wishart_process_em.init_bias_from_data

::: wishart_process_em.grand_covariance

## Numerics

::: wishart_process_em.QMCLattice

::: wishart_process_em.newton_minimize

::: wishart_process_em.NewtonResult
