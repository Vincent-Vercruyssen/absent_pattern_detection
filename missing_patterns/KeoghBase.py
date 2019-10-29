""" KeoghBase : Keoghs anomaly detection baseline for finding missing patterns.
"""

# Authors: Vincent Vercruyssen, 2018.

import sys, os, time
import numpy as np

from .BaseDetector import BaseDetector
from .PatternDetector import PatternDetector
from .utils.validation import check_time_series

# TODO: nicer solution than this
script_loc, _ = os.path.split(os.path.realpath(__file__))
sys.path.insert(0, os.path.split(script_loc)[0])

from anomaly_detection_time_series.WCAD import WCAD
from anomaly_detection_time_series.MatrixProfileAD import MatrixProfileAD


# -------------
# CLASSES
# -------------

class KeoghBase(BaseDetector, PatternDetector):
    """ Finding missing patterns given some known locations of the pattern.

    Parameters
    ----------
    ...

    Comments
    --------
    - Extends to OUT-OF-SAMPLE setting.
    - Fit-predict is somewhat different.
    - Requires at least one pattern occurence to be labeled.
    """

    def __init__(self,
                 ws = 0,
                 missing_classifier='MatrixProfile',  # 'MatrixProfile', 'WCAD'
                 missing_alpha=0.95,
                 tol=1e-8, verbose=False):
        super(KeoghBase, self).__init__()

        self.ws = int(ws)
        self.missing_classifier = str(missing_classifier)
        self.missing_alpha = float(missing_alpha)

        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict(self, timestamps, time_series, y):
        """ Fit the model to the training data and return predictions of missing
            patterns.

        :param timestamps : np.array(), shape (n_samples)
            The timestamps of the time series, datetime.datetime objects.
        :param time_series : np.array(), shape (n_samples, n_variables)
            The measured time series data.
        :param y : np.array(), shape (n_samples)
            Indicates the user-labeled y of a pattern (0, 1).

        :returns y_score : np.array(), shape (n_samples)
            Missing pattern probability in each time point.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers (i.e., missing patterns).
        """

        return self.fit(timestamps, time_series, y).predict_suspicion(timestamps, time_series)

    def fit(self, timestamps, time_series, y):
        """ Fit the model using the time series data.

        :param timestamps : np.array(), shape (n_samples)
            The timestamps of the time series, datetime.datetime objects.
        :param time_series : np.array(), shape (n_samples, n_variables)
            The measured time series data.
        :param y : np.array(), shape (n_samples)
            Indicates the user-labeled y of a pattern (0, 1).

        :returns self : object
        """

        times, ts, y = check_time_series(timestamps, time_series, y)
        n_samples = len(ts)

        # construct the feature vectors
        # TODO: fails if not a single pattern is annotated
        ranges = self._find_pattern_ranges(y)
        if self.ws > 0:
            self.w_size = self.ws
        else:
            self.w_size = self._find_window_size(ranges)

        # select the appropriate classifier
        self.clf = self._get_anomaly_detector()
        self.clf.fit(ts)

        return self

    def predict_suspicion(self, timestamps, time_series, occurrences=None):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns y_score : np.array(), shape (n_samples)
            Missing pattern probability in each time point.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers (i.e., missing patterns).
        """

        times, ts, _ = check_time_series(timestamps, time_series, None)
        n_samples = len(ts)

        # compute the anomaly score
        y_score, y_pred = self.clf.predict(ts)

        # fix: pad the end with zeros to match original length
        # this should not do anything here because it is already taken care off
        y_score = np.pad(y_score, (0, n_samples-len(y_score)), 'constant', constant_values=(0.0, 0.0))
        y_pred = np.pad(y_pred, (0, n_samples-len(y_pred)), 'constant', constant_values=(0.0, 0.0))

        return y_score, y_pred

    def _get_anomaly_detector(self):
        """ Return the anomaly detector. """

        if self.missing_classifier == 'MatrixProfile':
            return MatrixProfileAD(m=self.w_size, contamination=1.0-self.missing_alpha, verbose=self.verbose)
        elif self.missing_classifier == 'WCAD':
            return WCAD(m=self.w_size, contamination=1.0-self.missing_alpha, verbose=self.verbose)
