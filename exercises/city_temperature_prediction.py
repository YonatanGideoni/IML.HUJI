from itertools import cycle

import numpy as np
import pandas as pd
import plotly.io as pio
from matplotlib import pyplot as plt

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


def plot_temp_per_year_day(data: pd.DataFrame, country='Israel'):
    country_data = data[data.Country == country]

    cycol = cycle(list('bgrcmk') + ['orange', 'pink', 'goldenrod', 'darkblue', 'lime', 'dimgray', 'brown'])

    country_data.groupby('Year') \
        .apply(lambda df: df.plot.scatter(x='DayOfYear', y='Temp', label=df.name, ax=plt.gca(), grid=True,
                                          c=next(cycol)))

    plt.legend(fontsize=12, ncol=5)
    plt.xlabel('Day of year', fontsize=14)
    plt.ylabel('Temperature[C]', fontsize=14)
    plt.title('Temperature vs. day of year for different years', fontsize=16)
    MAX_DAY_OF_YEAR = 366
    plt.xlim(0, MAX_DAY_OF_YEAR)

    country_data.groupby('Month').Temp.agg(['mean', 'std']).reset_index().plot.bar(x='Month', y='mean', yerr='std')

    plt.ylabel('Temperature', fontsize=14)
    plt.xlabel('Month', fontsize=14)
    plt.title('Average temperature at each month', fontsize=16)
    plt.gca().get_legend().remove()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    plot_temp_per_year_day(data)

    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()

    plt.show()
