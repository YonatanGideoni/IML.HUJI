import numpy as np
import pandas as pd
import plotly.io as pio
from matplotlib import pyplot as plt

from IMLearn.learners import UnivariateGaussian

pio.templates.default = "simple_white"


def set_graph_attr(title='', xlabel='', ylabel='', label_size=14, title_size=16, ticks_size=11, ylim=0, grid=True):
    if grid:
        plt.grid(zorder=-np.inf)

    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)

    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)

    plt.ylim(ylim)


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
    plt.figure('Q2')
    delta_samples = 10
    model_size = np.arange(delta_samples, n_samples + 1, delta_samples)
    samples_mean_err = [abs(fit_gaussian(rand_samples[:model_n_samples]).mu_ - mean)
                        for model_n_samples in model_size]

    pd.DataFrame({'err': samples_mean_err, 'sample_size': model_size}) \
        .plot(x='sample_size', y='err', marker='o', linestyle='None', legend=False, xlim=0, ax=plt.gca())

    set_graph_attr(title='Gaussian mean estimation error as a function of sample size',
                   ylabel='Error, $|\mu-\hat{\mu}|$', xlabel='Sample size')

    # Question 3 - Plotting Empirical PDF of fitted model
    plt.figure('Q3')
    sigma_dist = 3
    n_points = 1000
    stand_dev = var ** 0.5
    pdf_x = np.linspace(mean - sigma_dist * stand_dev, mean + sigma_dist * stand_dev, num=n_points)
    pdf_vals = gaussian.pdf(pdf_x)

    plt.plot(pdf_x, pdf_vals, label='Fitted model', c='b', linestyle='--')

    real_gaussian_vals = np.exp(-(pdf_x - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    plt.plot(pdf_x, real_gaussian_vals, label='Empirical model', c='r', zorder=-1)

    set_graph_attr(xlabel='x', ylabel='PDF(x)', title='Empirical and fitted gaussian comparison')

    # TODO - finish this once it's better understood. Still need to plot the real model
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
