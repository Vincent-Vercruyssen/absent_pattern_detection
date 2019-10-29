""" ADBase : anomaly detection baseline for finding missing patterns.
"""

# Authors: Vincent Vercruyssen, 2018.

import sys, os, time
import numpy as np
from collections import Counter

from sklearn.preprocessing import StandardScaler

from .BaseDetector import BaseDetector
from .PatternDetector import PatternDetector
from .utils.validation import check_time_series

# TODO: nicer solution than this
script_loc, _ = os.path.split(os.path.realpath(__file__))
sys.path.insert(0, os.path.split(script_loc)[0])

from anomaly_detection.kNNo import kNNo
from anomaly_detection.IF import IF
from anomaly_detection.LOF import LOF
from anomaly_detection.SSAD import SSAD
from anomaly_detection.SSDO import SSDO


# -------------
# CLASSES
# -------------

class ADBase(BaseDetector, PatternDetector):
    """ Finding missing patterns given some known locations of the pattern.

    Parameters
    ----------
    ...

    Comments
    --------
    - Extends to OUT-OF-SAMPLE setting.
    - Fit-predict is somewhat different.
    """

    def __init__(self,
                 features='all',
                 k=10,
                 ws = 0,
                 n_clusters = 10,
                 warping_width=0.1,
                 alpha=0.5,
                 missing_classifier='kNNo',  # 'kNNo', 'IF', 'LOF', 'MatrixProfile', 'WCAD'
                 missing_alpha=0.95,
                 tol=1e-8, verbose=False):
        super(ADBase, self).__init__()

        self.features = str(features)
        try:
            self.k = int(k)
        except:
            self.k = float(k)
        self.ws = int(ws)
        self.n_clusters = int(n_clusters)
        self.warping_width = float(warping_width)
        self.alpha = float(alpha)
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
        t = time.time()
        ranges = self._find_pattern_ranges(y)
        if self.ws > 0:
            self.w_size = self.ws
        else:
            self.w_size = self._find_window_size(ranges)

        print('1) Found window range in:', time.time() - t, 's')

        t = time.time()
        patterns = np.array([time_series[ixs] for ixs in ranges])
        self.shape_templates = self._find_shape_templates(patterns)

        print('2) Found shape templates in:', time.time() - t, 's')

        t = time.time()
        features, labels = self._construct_features_and_labels(times, ts, self.shape_templates, ranges=ranges, labels=True)
        n_segments = features.shape[0]

        print('3) Constructed features in:', time.time() - t, 's')

        # fit the anomaly detection classifier
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        self.clf = self._get_anomaly_detector()
        # semi-supervised anomaly detectors vs the rest
        if self.missing_classifier in ['SSDO', 'SSAD']:
            # labels: 1.0 = normal occurrence of the pattern, -1.0 = non-occurrence
            # fix to the right format for the classifiers
            labels[labels == -1.0] = 0.0
            labels[labels == 1.0] = -1.0
            # train on a subset of the data (for speed purposes for SSAD)
            ixl = np.where(labels != 0.0)[0]
            # randomly sample 1000 instances from the remaining data 
            ixu = np.random.choice(np.setdiff1d(np.arange(0, len(labels), 1), ixl), 1000, replace=False)
            ixc = np.concatenate((ixl, ixu))
            subset_features = features[ixc, :]
            subset_labels = labels[ixc]
            self.clf.fit(subset_features, subset_labels)
        else:
            self.clf.fit(features)

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

        # construct features
        features = self._construct_features_and_labels(times, ts, self.shape_templates, labels=False)

        # predict anomalies
        features = self.scaler.transform(features)
        y_score, y_pred = self.clf.predict(features)

        # fix: pad the end with zeros to match original length
        y_score = np.pad(y_score, (0, n_samples-len(y_score)), 'constant', constant_values=(0.0, 0.0))
        y_pred = np.pad(y_pred, (0, n_samples-len(y_pred)), 'constant', constant_values=(0.0, 0.0))

        return y_score, y_pred

    def _get_anomaly_detector(self):
        """ Return the anomaly detector. """

        if self.missing_classifier == 'IF':
            return IF(contamination=1.0-self.missing_alpha, verbose=self.verbose)
        elif self.missing_classifier == 'kNNo':
            return kNNo(contamination=1.0-self.missing_alpha, k=self.k, verbose=self.verbose)
        elif self.missing_classifier == 'LOF':
            return LOF(contamination=1.0-self.missing_alpha, k=self.k, verbose=self.verbose)
        elif self.missing_classifier == 'SSDO':
            return SSDO(contamination=1.0-self.missing_alpha, k=self.k, base_classifier='IF')
        elif self.missing_classifier == 'SSAD':
            return SSAD(contamination=1.0-self.missing_alpha, Cu=self.k, Cl=self.k)
