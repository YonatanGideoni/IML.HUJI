from __future__ import annotations

from typing import Tuple, NoReturn

import numpy as np
import pandas as pd

from ...base import BaseEstimator
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_loss = np.inf
        for feature_ind in range(X.shape[1]):
            for sign in [-1, 1]:
                rel_feature = X[:, feature_ind]
                optim_thresh, thresh_err = self._find_threshold(rel_feature, y, sign)

                if thresh_err < min_loss:
                    min_loss = thresh_err
                    self.sign_ = self.sign_
                    self.j_ = feature_ind
                    self.threshold_ = optim_thresh

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        rel_features = X[:, self.j_]

        return (2 * (rel_features >= self.threshold_) - 1) * self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        labelled_features = pd.DataFrame({'feature': values, 'label': np.sign(labels)}).sort_values(by='feature')
        cumul_result = labelled_features.label.cumsum()
        optim_threshold_ind = cumul_result.argmax() if sign == -1 else cumul_result.argmin()
        optim_threshold = labelled_features.feature.iloc[optim_threshold_ind:optim_threshold_ind + 1].mean()

        pred_sign = sign * (2 * (labelled_features.values >= optim_threshold) - 1)
        thresh_err = ((labelled_features.label != pred_sign) * abs(labels)).sum()

        return optim_threshold, thresh_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(np.sign(y), self.predict(X))
