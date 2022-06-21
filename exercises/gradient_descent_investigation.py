import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import sklearn.metrics
from sklearn.metrics import RocCurveDisplay

from IMLearn import BaseModule, BaseLR
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    recorded_weights = []

    def callback(val, weights, **kwargs):
        values.append(val)
        recorded_weights.append(weights)

    return callback, values, recorded_weights


def plot_module_results(eta: float, learning_rate: BaseLR, module: type(BaseModule), mod_name: str,
                        init_weights: np.ndarray, plot: bool = True):
    callback, vals, weights = get_gd_state_recorder_callback()

    mod = module(init_weights)

    GradientDescent(learning_rate, callback=callback).fit(mod, X=None, y=None)

    if plot:
        plot_descent_path(module, np.array(weights), title=f'for {mod_name} for LR={eta:.3f}').show()

        plt.figure()
        plt.plot(vals)

        plt.xlim(0)
        plt.title(f'Convergence rate for {mod_name} for $\eta$={eta:.3f}', fontsize=16)
        plt.xlabel('Iteration', fontsize=13)
        plt.ylabel('Loss value', fontsize=13)

        plt.semilogy()
        plt.grid()

        plt.savefig(f'{mod_name}_{eta:.3f}.png')

    print(f'Min loss for {mod_name}={min(vals):.2e}, eta={eta}')

    return vals


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        plot_module_results(eta, FixedLR(eta), L2, 'L2', init)
        plot_module_results(eta, FixedLR(eta), L1, 'L1', init)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    l1_scores_per_iter = [plot_module_results(eta, ExponentialLR(eta, gamma), L1, 'L1', init, plot=False) for gamma in
                          gammas]

    # Plot algorithm's convergence for the different values of gamma
    linestyles = ['solid', 'dotted', 'dashed', '-.']
    for score, gamma, linestyle in zip(l1_scores_per_iter, gammas, linestyles):
        plt.plot(score, linestyle=linestyle, label=f'$\gamma$={gamma}')

    plt.legend()
    plt.grid()
    plt.xlim(0)
    plt.semilogy()
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title('Loss per iteration for different decay rates', fontsize=16)

    # Plot descent path for gamma=0.95
    plot_module_results(eta, ExponentialLR(eta, gammas[1]), L1, 'L1', init)
    plot_module_results(eta, ExponentialLR(eta, gammas[1]), L2, 'L2', init)


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    solver = GradientDescent(FixedLR(1e-4), max_iter=20000)
    log_reg = LogisticRegression(solver=solver).fit(X_train, y_train)

    train_pred = log_reg.predict_proba(X_train)
    fpr, tpr, alpha_vals = sklearn.metrics.roc_curve(y_train, train_pred)

    RocCurveDisplay.from_predictions(y_train, train_pred)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('ROC curve for logistic regression', fontsize=15)

    opt_alpha = alpha_vals[np.argmax(tpr - fpr)]
    print(f'Alpha:{opt_alpha:.3f}')

    log_reg.alpha_ = opt_alpha
    print(f'Test error: {log_reg.loss(X_test, y_test):.2f}')

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambda_vals = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)

    best_lam = 0
    best_score = np.inf
    for lam_val in lambda_vals:
        regularized_logreg = LogisticRegression(solver=solver, lam=lam_val, penalty='l1')

        _, val_score = cross_validate(regularized_logreg, X_train, y_train, misclassification_error)
        if val_score < best_score:
            best_score = val_score
            best_lam = lam_val

    print(f'Opt lambda: {best_lam}')


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()

    plt.show()
