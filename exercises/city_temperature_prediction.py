from itertools import cycle

import numpy as np
import pandas as pd
import plotly.io as pio
from matplotlib import pyplot as plt

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date'])

    # remove absurdly low temperatures
    MIN_TEMP = -40
    data = data[data.Temp > MIN_TEMP]

    data['DayOfYear'] = data.Date.dt.dayofyear

    return data


def get_avg_temp_per_month(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby('Month').Temp.agg(['mean', 'std']).reset_index()


def plot_temp_per_year_day(data: pd.DataFrame, country='Israel'):
    country_data = data[data.Country == country]

    cycol = cycle(list('bgrcmk') + ['orange', 'pink', 'goldenrod', 'darkblue', 'lime', 'dimgray', 'brown'])

    country_data.groupby('Year') \
        .apply(lambda df: df.plot.scatter(x='DayOfYear', y='Temp', label=df.name, ax=plt.gca(), grid=True,
                                          c=next(cycol)))

    plt.legend(fontsize=12, ncol=5)
    plt.xlabel('Day of year', fontsize=14)
    plt.ylabel('Temperature[C]', fontsize=14)
    plt.title(f'{country} temperature vs. day of year for different years', fontsize=16)
    MAX_DAY_OF_YEAR = 366
    plt.xlim(0, MAX_DAY_OF_YEAR)

    get_avg_temp_per_month(country_data).plot.bar(x='Month', y='mean', yerr='std')

    plt.ylabel('Temperature', fontsize=14)
    plt.xlabel('Month', fontsize=14)
    plt.title(f'Average {country} temperature at each month', fontsize=16)
    plt.gca().get_legend().remove()


def plot_countries_avg_temp(data: pd.DataFrame):
    colours = list('rgbm')
    plt.figure()
    data.groupby('Country') \
        .apply(lambda country_df: get_avg_temp_per_month(country_df)
               .plot(x='Month', y='mean', yerr='std', c=colours.pop(), label=country_df.name, ax=plt.gca(),
                     marker='o', grid=True))

    plt.title('Monthly temperature over time for different countries', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Temperature[C]', fontsize=14)
    plt.legend(fontsize=12)


def fit_polynomials_to_model(data: pd.DataFrame, country='Israel', max_deg=10):
    country_data = data[data.Country == 'Israel']
    train_data, train_resp, test_data, test_resp = split_train_test(country_data.DayOfYear, country_data.Temp)
    res_loss = []
    for k in range(1, max_deg + 1):
        poly_fit = PolynomialFitting(k).fit(train_data, train_resp)

        loss = poly_fit.loss(test_data, test_resp)

        res_loss.append({'degree': k, 'loss': loss})

    res_loss = pd.DataFrame.from_records(res_loss)
    print(res_loss)

    res_loss.plot.bar(x='degree', y='loss', grid=True, fontsize=12, zorder=np.inf)
    plt.gca().get_legend().remove()

    plt.title(f'MSE vs degree of {country} temperature prediction polynomial', fontsize=16)
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('MSE', fontsize=14)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    plot_temp_per_year_day(data)

    # Question 3 - Exploring differences between countries
    plot_countries_avg_temp(data)

    # Question 4 - Fitting model for different values of `k`
    fit_polynomials_to_model(data)

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()

    plt.show()
