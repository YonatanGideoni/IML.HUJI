import numpy as np
import pandas as pd
import plotly.io as pio
from matplotlib import pyplot as plt

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

pio.templates.default = "simple_white"


def set_graph_attr(title='', xlabel='', ylabel='', label_size=14, title_size=16, ticks_size=11, ylim: float = 0,
                   grid: bool = True):
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

    plt.scatter(rand_samples, [0] * len(rand_samples), label='Data points', c='r', zorder=np.inf, alpha=0.2)

    set_graph_attr(xlabel='x', ylabel='PDF(x)', title='Fitted gaussian and sample density comparison', ylim=-0.005)


def test_multivariate_gaussian():
    mean = np.array([0, 0, 4, 0])
    cov_mat = np.array([[1, 0.2, 0, 0.5],
                        [0.2, 2, 0, 0],
                        [0, 0, 1, 0],
                        [0.5, 0, 0, 1]])

    # Question 4 - Draw samples and print fitted model
    n_samples = 1000
    rand_samples = np.random.multivariate_normal(mean, cov_mat, n_samples)
    gaussian = MultivariateGaussian().fit(rand_samples)

    print(gaussian.mu_)
    print(gaussian.cov_)

    # Question 5 - Likelihood evaluation
    n_test_vals = 200
    delta_vals = 10
    f1_vals = np.linspace(-delta_vals, delta_vals, n_test_vals)
    f3_vals = f1_vals.copy()

    gaussian.pdf(rand_samples)
    log_likelihood = [[gaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov_mat, rand_samples)
                       for f1 in f1_vals] for f3 in f3_vals]

    plt.figure('Q5')
    plt.imshow(np.transpose(log_likelihood), extent=[-delta_vals, delta_vals, -delta_vals, delta_vals])
    cbar = plt.colorbar()
    cbar.set_label('Log-likelihood', fontsize=13)
    set_graph_attr(grid=False, title='Log-likelihood heatmap for different values of $f_1,f_3$', xlabel='$f_3$',
                   ylabel='$f_1$', ylim=None)

    # Question 6 - Maximum likelihood
    max_likelihood_ind = np.argmax(log_likelihood)
    f1_ind = max_likelihood_ind % n_test_vals
    f3_ind = max_likelihood_ind // n_test_vals

    likely_f1 = f1_vals[f1_ind]
    likely_f3 = f3_vals[f3_ind]

    print(f'{likely_f1:.3f}, {likely_f3:.3f}')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

    plt.show()
