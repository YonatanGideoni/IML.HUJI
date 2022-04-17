import os.path
from math import atan2, pi
from typing import Tuple

import matplotlib.pyplot as plt
# from IMLearn.learners.classifiers import Perceptron, GaussianNaiveBayes
from matplotlib.lines import Line2D
from sklearn.naive_bayes import GaussianNB

from IMLearn import BaseEstimator
from IMLearn.learners.classifiers import LDA
from utils import *


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for dataset_name, filename in [("Linearly Separable", "linearly_separable.npy"),
                                   ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data, response = load_dataset(os.path.join('../datasets', filename))

        # Fit Perceptron and record loss in each fit iteration
        loss_per_iteration = []

        def record_loss(perceptron: Perceptron, sample, sample_response):
            loss_per_iteration.append(perceptron.loss(data, response))

        Perceptron(callback=record_loss).fit(data, response)

        # Plot figure of loss as function of fitting iteration
        plt.figure(dataset_name)
        plt.plot(loss_per_iteration)

        plt.title(f'Loss per Perceptron iteration for {dataset_name} dataset', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Misclassification loss', fontsize=14)
        plt.xlim(0)
        plt.ylim(-0.01)
        plt.grid()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def plot_classifier_res(data: np.ndarray, response: np.ndarray, classifier: BaseEstimator, axis: plt.axis,
                        dataset_name: str, classifier_name: str, colours: list, markers: list):
    predicted_data_labels = classifier.predict(data)

    for pred_class, c in enumerate(colours):
        for true_class, m in enumerate(markers):
            rel_data_inds = (response == true_class) & (predicted_data_labels == pred_class)
            axis.scatter(data[rel_data_inds, 0], data[rel_data_inds, 1], c=c, marker=m)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for filename in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data, response = load_dataset(os.path.join('../datasets', filename))

        # Fit models and predict over training set
        gaussian_n_b = GaussianNB().fit(data, response)
        linear_disc_analysis = LDA().fit(data, response)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
        fig.subplots_adjust(wspace=0)

        COLOURS = np.array(['m', 'r', 'darkgreen'])
        MARKERS = np.array(['o', 'D', 's'])

        plot_classifier_res(data, response, gaussian_n_b, axs[0], filename.split('.')[0], 'Gaussian Naive Bayes',
                            COLOURS, MARKERS)
        plot_classifier_res(data, response, linear_disc_analysis, axs[1], filename.split('.')[0], 'LDA', COLOURS,
                            MARKERS)

        legend_content = [Line2D([], [], color=c, marker=m, linestyle='None',
                                 label=f'Class={true_class}, Prediction={pred_class}')
                          for true_class, m in enumerate(MARKERS)
                          for pred_class, c in enumerate(COLOURS)]
        plt.legend(handles=legend_content, bbox_to_anchor=(-1, -.02, 1, 0.2), loc="lower left", ncol=3)

        # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # TODO - uncomment stuff
    # run_perceptron()
    compare_gaussian_classifiers()

    plt.show()
