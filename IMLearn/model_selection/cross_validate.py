from __future__ import annotations

from typing import Tuple, Callable

import numpy as np

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds = np.arange(len(y)) // (len(y) / cv)
    np.random.shuffle(folds)
    train_scores = np.zeros(cv)
    validation_scores = np.zeros(cv)

    for validation_fold in range(cv):
        fold_train_X = X[folds != validation_fold]
        fold_train_y = y[folds != validation_fold]

        fold_estimator = estimator.fit(fold_train_X, fold_train_y)

        train_scores[validation_fold] = scoring(fold_estimator.predict(fold_train_X), fold_train_y)

        fold_val_X = X[folds == validation_fold]
        fold_val_y = y[folds == validation_fold]

        validation_scores[validation_fold] = scoring(fold_estimator.predict(fold_val_X), fold_val_y)

    return train_scores.mean(), validation_scores.mean()
