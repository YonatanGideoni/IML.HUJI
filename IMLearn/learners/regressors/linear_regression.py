from __future__ import annotations

from typing import NoReturn, Union

import numpy as np
import pandas as pd
from numpy.linalg import pinv

from ...base import BaseEstimator
from ...metrics import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : pd.Series of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        design_matrix = self.__get_design_mat(X)

        pseudo_inv = np.linalg.pinv(design_matrix)

        self.coefs_ = pseudo_inv @ y.values

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.__get_design_mat(X) @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(self.predict(X), y)

    def __get_design_mat(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = X.copy()
        if self.include_intercept_:
            X['dummy_intercept_col'] = 1

        return X.values if isinstance(X, pd.DataFrame) else X
