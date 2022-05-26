from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    MIN_X = -1.2
    MAX_X = 2
    x = np.linspace(MIN_X, MAX_X, n_samples)
    true_vals = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    model_noise = np.random.normal(0, noise, n_samples)
    noisy_vals = true_vals + model_noise

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame({'x': x}), pd.Series(noisy_vals),
                                                        train_proportion=2 / 3)

    plt.figure('Q1')
    plt.scatter(x, true_vals, marker='o', label='True model')
    plt.scatter(train_X, train_y, marker='d', c='m', label='Train')
    plt.scatter(test_X, test_y, marker='+', c='darkgreen', label='Test')

    plt.legend()
    plt.grid()
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('f(x) for the true model and train/test sets', fontsize=14)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    MAX_POLY_DEG = 10
    poly_degs = np.arange(MAX_POLY_DEG + 1)
    train_score = np.zeros(MAX_POLY_DEG + 1)
    val_score = np.zeros(MAX_POLY_DEG + 1)
    for deg in poly_degs:
        model = PolynomialFitting(deg)
        train_score[deg], val_score[deg] = cross_validate(model, train_X.x, train_y, mean_square_error)

    plt.figure('Q2')
    plt.scatter(poly_degs, train_score, label='Train', marker='d', c='m')
    plt.scatter(poly_degs, val_score, label='Validation', marker='p', c='c')

    plt.legend()
    plt.grid()
    plt.xlabel('k', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Average train/validation score for polynomials of different degrees', fontsize=14)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    BEST_DEG = 5
    poly_fit = PolynomialFitting(BEST_DEG).fit(train_X.x, train_y)
    test_err = mean_square_error(poly_fit.predict(test_X.x), test_y)
    print(f'Test error: {test_err:.2f}')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree(noise=10, n_samples=1500)

    select_regularization_parameter()

    plt.show()
