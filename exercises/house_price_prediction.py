import os
from typing import NoReturn

import numpy as np
import pandas as pd
import plotly.io as pio
from matplotlib import pyplot as plt

from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def __drop_invalid_values(houses_data: pd.DataFrame) -> pd.DataFrame:
    NONPOS_COLS = {'lat', 'long', 'yr_renovated', 'view', 'waterfront', 'bathrooms', 'bedrooms', 'sqft_basement'}
    for col in set(houses_data.columns) - NONPOS_COLS:
        if not pd.api.types.is_numeric_dtype(houses_data[col].dtype):
            continue
        houses_data = houses_data[houses_data[col] > 0]

    return houses_data.dropna()


def load_data(filename: str) -> pd.DataFrame:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)

    data = __drop_invalid_values(data)

    # one-hot encoding
    CAT_VARS = ['zipcode', 'grade', 'waterfront', 'view', 'condition']
    data = pd.get_dummies(data, columns=CAT_VARS)

    # get relevant month from date
    dates = data.date.astype('datetime64')
    min_date = dates.min()
    data['days_from_beginning'] = (dates - min_date).apply(lambda timerange: timerange.days)

    # add relative lot/living space sizes columns
    data['rel_lot_size'] = data.sqft_lot / data.sqft_lot15
    data['rel_living_size'] = data.sqft_living / data.sqft_living15

    return data.drop(['id', 'date'], axis='columns')


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = np.std(y)
    ZORDER = 10
    for col in X.columns:
        if col == y.name or 'day' in col:
            continue

        corr_coeff = np.cov(X[col], y)[0, 1] / (y_std * np.std(X[col]))
        X.plot.scatter(x=col, y=y.name, grid=True,
                       title=f'{col} vs house prices, correlation coefficient={corr_coeff:.2f}', zorder=ZORDER)

        plt.savefig(os.path.join(output_path, col.replace('.', '_')))


def fit_linear_regression(data: pd.DataFrame, response: pd.Series) -> np.ndarray:
    design_matrix = data.values

    pseudo_inv = np.linalg.pinv(design_matrix)

    return pseudo_inv @ response.values


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, data.price)

    # Question 3 - Split samples into training- and testing sets.
    train_data, _, test_data, _ = split_train_test(data, data.price)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()

    plt.show()
