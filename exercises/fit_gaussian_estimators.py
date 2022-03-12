import numpy as np
import pandas as pd
import plotly.io as pio
from matplotlib import pyplot as plt

from IMLearn.learners import UnivariateGaussian

pio.templates.default = "simple_white"


def fit_gaussian(samples: np.ndarray) -> UnivariateGaussian:
    return UnivariateGaussian(biased_var=False).fit(samples)


def test_univariate_gaussian():
    mean, var = 10, 1
    n_samples = 1000

    # Question 1 - Draw samples and print fitted model
    rand_samples = np.random.normal(loc=mean, scale=var ** 0.5, size=n_samples)

    gaussian = fit_gaussian(rand_samples)

    print(f'({gaussian.mu_}, {gaussian.var_})')

    # Question 2 - Empirically showing sample mean is consistent
    delta_samples = 10
    model_size = np.arange(delta_samples, n_samples + 1, delta_samples)
    samples_mean_err = [abs(fit_gaussian(rand_samples[:model_n_samples]).mu_ - mean)
                        for model_n_samples in model_size]

    pd.DataFrame({'err': samples_mean_err, 'sample_size': model_size}) \
        .plot(x='sample_size', y='err', grid=True, marker='o', linestyle='None', legend=False, fontsize=11, xlim=0)

    plt.xlabel('Sample size', size=14)
    plt.ylabel('Error, $|\mu-\hat{\mu}|$', size=14)
    plt.semilogy()
    plt.title('Gaussian mean estimation error as a function of sample size', fontsize=16)

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()

    plt.show()
