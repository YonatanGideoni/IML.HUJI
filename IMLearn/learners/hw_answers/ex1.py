import numpy as np

from IMLearn.learners.gaussian_estimators import UnivariateGaussian, MultivariateGaussian


def fit_univ_gaussian(mean: float = 10, var: float = 1, n_samples: int = 1000) -> UnivariateGaussian:
    rand_samples = np.random.normal(loc=mean, scale=var ** 0.5, size=n_samples)

    return UnivariateGaussian(biased_var=False).fit(rand_samples)


def question_one():
    gaussian = fit_univ_gaussian()

    print(f'({gaussian.mu_}, {gaussian.var_})')
