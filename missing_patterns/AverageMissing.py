""" AverageMissing : simple baseline that predicts the average interarrival time.
"""

# Authors: Vincent Vercruyssen, 2018.

import sys, os, time
import numpy as np

from .BaseDetector import BaseDetector
from .PatternDetector import PatternDetector
from .FingerprintDetector import FingerprintDetector
from .utils.validation import check_time_series


# -------------
# CLASSES
# -------------

class AverageMissing(BaseDetector, PatternDetector, FingerprintDetector):
    """ Finding missing patterns given some known y of the pattern.

    Comments
    --------
    - Extends to OUT-OF-SAMPLE setting.
    - Fit-predict is somewhat different.
    - Keywords are passed along to the parent classes.
    """

    def __init__(self,
                missing_alpha=0.95,
                occurrence_detection='pattern',  # 'pattern', 'fingerprint'
                # PatternDetector
                features='all',
                n_clusters=10,
                warping_width=0.1,
                alpha=0.5,
                bagging_classifier='SVM',
                bagging_kernel='linear',
                bagging_K=200,
                bagging_T=100,
                bagging_cweights=False,
                clf_scaling='standard',
                smoothing='low-pass',  # 'convolve', 'low-pass'
                # FingerprintDetector
                detector_type='dtw',
                # remaining
                verbose=False,
                tol=1e-8):

        self.missing_alpha = float(missing_alpha)
        self.occurrence_detection = str(occurrence_detection)

        # PatternDetector parameters
        self.features = str(features)
        self.n_clusters = int(n_clusters)
        self.warping_width = float(warping_width)
        self.alpha = float(alpha)
        self.bagging_classifier = str(bagging_classifier)
        self.bagging_kernel = str(bagging_kernel)
        self.bagging_K = int(bagging_K)
        self.bagging_T = int(bagging_T)
        self.bagging_cweights = bool(bagging_cweights)
        self.clf_scaling = str(clf_scaling)
        self.smoothing = str(smoothing)

        # FingerprintDetector parameters
        self.detector_type = str(detector_type)

        self.tol = float(tol)
        self.verbose = bool(verbose)

        # instantiate the parent classes
        super().__init__()

    def fit_predict(self, timestamps, time_series, y):
        """ Fit the model to the training data and return predictions of missing
            patterns.

        :param timestamps : np.array(), shape (n_samples)
            The timestamps of the time series, datetime.datetime objects.
        :param time_series : np.array(), shape (n_samples, n_variables)
            The measured time series data.
        :param y : np.array(), shape (n_samples)
            Indicates the user-labeled y of a pattern (0, 1).

        :returns y_prob : np.array(), shape (n_samples)
            Missing pattern probability in each time point.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers (i.e., missing patterns).
        :returns y_locations : np.array(), shape (n_samples)
            Predicted locations of the pattern of interest.
        """

        times, ts, y = check_time_series(timestamps, time_series, y)

        # predict locations and compute suspicion
        if self.occurrence_detection == 'pattern':
            y_locations = self.fit_predict_occurrences(times, ts, y)
        elif self.occurrence_detection == 'fingerprint':
            y_locations = self.fit_predict_fingerprints(times, ts, y)
        else:
            pass

        # 2. construct classifier to find missing patterns
        times = self._construct_interarrival_times(ts, y_locations)
        self.avg_time = np.mean(times)
        y_prob, y_pred = self._compute_suspicion(ts, y_locations, self.avg_time)

        return y_prob, y_pred, y_locations

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

        # 1. construct classifier to detect patterns
        if self.occurrence_detection == 'pattern':
            y_locations = self.fit_predict_occurrences(times, ts, y)
        elif self.occurrence_detection == 'fingerprint':
            y_locations = self.fit_predict_fingerprints(times, ts, y)
        else:
            pass

        # 2. construct classifier to find missing patterns
        # TODO: FIX: not enough occurrences to fit suspicion
        times = self._construct_interarrival_times(ts, y_locations)
        self.avg_time = np.mean(times)

        return self

    def predict(self, timestamps, time_series):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns y_prob : np.array(), shape (n_samples)
            Missing pattern probability in each time point.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers (i.e., missing patterns).
        :returns y_locations : np.array(), shape (n_samples)
            Predicted locations of the pattern of interest.
        """

        times, ts, _ = check_time_series(timestamps, time_series, None)

        # 1. detect the pattern occurrences
        if self.occurrence_detection == 'pattern':
            y_locations = self.predict_occurrences(times, ts)
        elif self.occurrence_detection == 'fingerprint':
            y_locations = self.predict_fingerprints(times, ts)
        else:
            pass

        # 2. find the missing patterns
        y_prob, y_pred = self._compute_suspicion(ts, y_locations, self.avg_time)

        return y_prob, y_pred, y_locations

    def predict_suspicion(self, timestamps, time_series, occurrences):
        """ Compute suspicion given the occurrences.

        :returns y_prob : np.array(), shape (n_samples)
            Missing pattern probability in each time point.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers (i.e., missing patterns).
        """

        times, ts, y = check_time_series(timestamps, time_series, occurrences)

        # find the missing patterns
        y_prob, y_pred = self._compute_suspicion(ts, y, self.avg_time)

        return y_prob, y_pred

    def _compute_suspicion(self, ts, exact_locations, avg_time):
        """ Compute the survival probability under the Weibull model.

        :returns y_prob : np.array(), shape (n_samples)
            Missing pattern probability in each time point.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers (i.e., missing patterns).
        """

        y_prob = np.zeros(len(ts), dtype=float)

        value_last_event = 0
        for i in range(len(y_prob)):
            # event or not
            event = True if exact_locations[i] == 1.0 else False
            if event:
                value_last_event = i

            # value of the variable to compute weibull prob
            ft = i - value_last_event

            # compute survival
            if ft <= avg_time:
                y_prob[i] = 0.0
            else:
                y_prob[i] = 1.0

        # hard predictions with threshold
        y_pred = y_prob.copy()

        return y_prob, y_pred

    def _construct_interarrival_times(self, time_series, exact_locations):
        """ Construct the event features.

        :returns descriptors : np.array(), shape (n_samples, n_features)
            Construct the descriptors for the missingness model.
        """

        ts = np.where(exact_locations == 1.0)[0]  # TODO: better error handling
        return np.diff(ts)
