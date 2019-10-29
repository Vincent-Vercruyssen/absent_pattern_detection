""" Functions for validation """

# Authors: Vincent Vercruyssen

import warnings
import numpy as np
import pandas as pd
from datetime import datetime


def check_time_series(timestamps, time_series, y):
    """ Check the input values.
    :param timestamps : np.array(), shape (n_samples)
        The timestamps of the time series, datetime.datetime objects.
    :param time_series : np.array(), shape (n_samples, n_variables)
        The measured time series data.
    :param y : np.array(), shape (n_samples)
        Indicates the user-labeled y of a pattern (0, 1).
    :returns times_converted : np.array()
        Converted and validated timestamps.
    :returns ts_converted : np.array()
        Converted and validated time_series.
    :returns occs_converted : np.array()
        Converted and validated y.
    """

    # convert: timestamps
    if not(isinstance(timestamps, np.ndarray)):
        raise ValueError('Input timestamps is not a numpy array.')
    if isinstance(timestamps[0], np.datetime64):
        timestamps = np.array([datetime.fromtimestamp((ts - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')) for ts in timestamps])

    # convert: time series
    if not(isinstance(time_series, np.ndarray)):
        raise ValueError('Input time_series is not a numpy array.')

    if not(len(time_series) == len(timestamps)):
        raise ValueError('Input time_series and timestamps do not have same length.')

    # convert: y
    if not(isinstance(y, np.ndarray)):
        if y is not None:
            raise ValueError('Input y is not a numpy array or none.')
    if not(isinstance(y, np.ndarray)) and y is None:
        y = np.zeros(len(time_series), dtype=float)

    return timestamps, time_series, y
