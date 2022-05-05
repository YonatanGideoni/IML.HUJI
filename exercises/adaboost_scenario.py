from typing import Tuple, Union

from matplotlib import pyplot as plt

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
from utils import *


def set_graph_attr(title='', xlabel='', ylabel='', label_size=14, title_size=16, ticks_size=11,
                   ylim: Union[None, float] = 0, grid: bool = True, legend: bool = True, legend_size: int = 13):
    """
    Sets several of a graphs attributes.

    Parameters
    ----------
    title: str
        The graph's title.
    xlabel: str
        The graph's x axis label.
    ylabel: str
        The graph's y axis label.
    label_size: int
        The x/y axes label sizes.
    title_size: int
        The title's size.
    ticks_size: int
        The x/y ticks sizes.
    ylim: float
        Bottom limit of the y axis.
    grid: bool
        Whether or not to have a grid.
    legend: bool
    legend_size: int
    """
    if grid:
        plt.grid(zorder=-np.inf)

    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)

    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)

    plt.ylim(ylim)
    if legend:
        plt.legend(fontsize=legend_size)


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def plot_decision_surface(learner: AdaBoost, num_learners: int, lims: np.ndarray, density=125):
    x_lims, y_lims = lims
    xrange, yrange = np.linspace(*x_lims, density), np.linspace(*y_lims, density)
    xx, yy = np.meshgrid(xrange, yrange)

    pred = learner.partial_predict(np.c_[xx.ravel(), yy.ravel()], num_learners).reshape(xx.shape)

    plt.contourf(xx, yy, pred, cmap='Paired')


def plot_ensemble_decision_surface(learner: AdaBoost, num_learners: int, X: np.ndarray, y: np.ndarray,
                                   lims: np.ndarray, weights: np.ndarray = None):
    colors = np.array(['m'] * len(y))
    colors[y == 1] = 'c'

    plt.figure()
    plot_decision_surface(learner, num_learners, lims)
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=weights)

    test_err = learner.partial_loss(X, y, num_learners)

    set_graph_attr(title=f'Decision surface for Adaboosted model with {num_learners} stumps\nError={test_err}',
                   grid=False, legend=False)
    plt.xlim(*lims[0])
    plt.ylim(*lims[1])


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    boosted_learner = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    diff_n_learners_train_err = [boosted_learner.partial_loss(train_X, train_y, i) for i in range(1, n_learners)]
    diff_n_learners_test_err = [boosted_learner.partial_loss(test_X, test_y, i) for i in range(1, n_learners)]

    plt.figure()
    plt.plot(diff_n_learners_train_err, label='Train', c='r', linestyle='dashed')
    plt.plot(diff_n_learners_test_err, label='Test', c='g')

    set_graph_attr(title='Train and test error for different number of learners', xlabel='# of learners',
                   ylabel='Misclassification error')
    plt.xlim(0)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    for num_learners in T:
        plot_ensemble_decision_surface(boosted_learner, num_learners, test_X, test_y, lims)

    # Question 3: Decision surface of best performing ensemble
    best_ensemble_size = np.argmin(diff_n_learners_test_err)
    plot_ensemble_decision_surface(boosted_learner, best_ensemble_size, test_X, test_y, lims)

    # Question 4: Decision surface with weighted samples
    weights = boosted_learner.D_
    weights *= 200 / weights.max()
    plot_ensemble_decision_surface(boosted_learner, n_learners, train_X, train_y, lims, weights)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)

    plt.show()
