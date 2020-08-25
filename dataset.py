# Batteries
from datetime import timedelta

# Externals
import numpy as np


def get_data(df, country, dt_range):
    """
    country = 'Germany'
    start_dt = '2020-07-01'
    end_dt = '2020-09-01'
    """
    ind = 'confirmed cases'
    cou = country
    sdt = dt_range[0]
    edt = dt_range[1]
    wd = 4  # friday

    query = 'indicator == @ind & country == @cou & date >= @sdt & date <= @edt & date.dt.weekday == @wd'
    new_df = df.query(query)
    return new_df.loc[:, 'date'], new_df.loc[:, 'rate_14_day']


def create_x(days_fr, len_predict):
    X = np.arange(1, len(days_fr) + 1)
    X = X.reshape((len(days_fr), 1))

    X_pred = np.arange(len(days_fr) + 1, len(days_fr) + len_predict + 1)
    X_pred = X_pred.reshape((len_predict, 1))

    dt0 = days_fr.iloc[0]
    x_days = [dt0 + timedelta(days=7) * x for x in range(len(X) + len(X_pred))]
    return X, X_pred, x_days
