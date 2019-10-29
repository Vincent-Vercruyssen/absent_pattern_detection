""" FZapPa : finding missing patterns in time series data.
"""

# Authors: Vincent Vercruyssen, 2018.

import sys, os, time
import numpy as np
import scipy.stats as sps

from scipy.optimize import minimize

from .BaseDetector import BaseDetector
from .PatternDetector import PatternDetector
from .FingerprintDetector import FingerprintDetector
from .utils.validation import check_time_series


# -------------
# CLASSES
# -------------

class FZapPa(BaseDetector, PatternDetector, FingerprintDetector):
    """ Finding missing patterns given some known y of the pattern.

    Comments
    --------
    - Extends to OUT-OF-SAMPLE setting.
    - Fit-predict is somewhat different.
    - Keywords are passed along to the parent classes.
    """

    def __init__(self,
                missing_features='all',  # 'all', 'time', 'usage'
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

        self.missing_features = str(missing_features)
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
        descriptors = self._construct_descriptors(ts, y_locations)
        self.model, self.best_descriptor = self._fit_weibull_model(descriptors)
        y_prob, y_pred = self._compute_suspicion(ts, y_locations, self.model, self.best_descriptor)
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
        descriptors = self._construct_descriptors(ts, y_locations)
        self.model, self.best_descriptor = self._fit_weibull_model(descriptors)

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
        y_prob, y_pred = self._compute_suspicion(ts, y_locations, self.model, self.best_descriptor)

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
        y_prob, y_pred = self._compute_suspicion(ts, y, self.model, self.best_descriptor)

        return y_prob, y_pred

    def _compute_suspicion(self, ts, exact_locations, model, feature_name):
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
            if feature_name == 'time':
                ft = i - value_last_event
            elif feature_name == 'usage':
                ft = np.sum(ts[value_last_event:i])

            # compute survival
            susp = model.survival(ft)
            y_prob[i] = 1.0 - susp

        # hard predictions with threshold
        flagged_missing = np.zeros(len(ts), dtype=float)
        flagged_missing[y_prob <= self.missing_alpha] = 0.0
        flagged_missing[y_prob > self.missing_alpha] = 1.0

        y_pred = np.zeros(len(ts), dtype=float)
        miss_ranges = self._find_pattern_ranges(flagged_missing)
        if len(miss_ranges) > 0:
            ixs = np.array([mr[0] for mr in miss_ranges])
            y_pred[ixs] = 1.0

        return y_prob, y_pred

    def _fit_weibull_model(self, descriptors):
        """ Find the best fitting weibull model

        :return model : object
            Best Weibull model.
        :return best_descriptor : str
            Name of best fitting feature.
        """

        # find best descriptor
        best_gof = np.inf
        for fname, feat in descriptors.items():
            # weibull = standardWeibull()
            weibull = scipyWeibull()
            weibull.fit(feat)
            new_gof = weibull.gof
            print(fname, ':', weibull.gof)
            if new_gof < best_gof:
                best_gof = new_gof
                best_descriptor = fname

        print('Selected:', best_descriptor)
        model = scipyWeibull()
        model.fit(descriptors[best_descriptor])
        return model, best_descriptor

    def _construct_descriptors(self, time_series, exact_locations):
        """ Construct the event features.

        :returns descriptors : np.array(), shape (n_samples, n_features)
            Construct the descriptors for the missingness model.
        """

        if self.missing_features == 'all':
            use_features = ['usage', 'time']
        else:
            use_features = [self.missing_features]

        event_features = {}
        ts = np.where(exact_locations == 1.0)[0]  # TODO: better error handling

        """ ARBITRARY FIX if no locations found or 1 location found """
        # should be avoided here

        # 1. time between events
        if 'time' in use_features:
            event_features['time'] = np.diff(ts)

        # 2. usage between events
        if 'usage' in use_features:
            usage_meter = []
            for i in range(len(ts) - 1):
                ct = ts[i]
                nt = ts[i+1]
                usage_meter.append(np.sum(time_series[ct:nt]))
            event_features['usage'] = np.array(usage_meter)

        return event_features



# -------------
# SCIPY WEIBULL
# -------------

class scipyWeibull:

    def __init__(self):

        self.l = 1.0
        self.k = 1.0
        self.loc = 0.0
        self.scale = 1.0

        self.gof = 0.0  # goodness of fit

    def fit(self, data):
        """ Maximum likelihood (MLE estimation) of the pdf parameters. """

        # fit the weibull distribution to the data
        self.l, self.k, self.loc, self.scale = sps.exponweib.fit(data, loc=0.0, scale=1.0)

        # goodness of fit
        nll = - np.sum(np.log(sps.exponweib.pdf(data, self.l, self.k, loc=self.loc, scale=self.scale)))
        self.gof = nll

    def pdf(self, x):
        """ Probability density function, l is the rate. """

        weibull = sps.exponweib.pdf(x, self.l, self.k, loc=self.loc, scale=self.scale)
        return weibull

    def cdf(self, x):
        """ Cumulative density function. """

        weibull = sps.exponweib.cdf(x, self.l, self.k, loc=self.loc, scale=self.scale)
        return weibull

    def compute_threshold(self, alpha=0.9):
        """ Compute the threshold-value containing \alpha percent of examples. """

        return (self.l * np.power(-np.log((1 - alpha)), 1 / self.k)) * self.scale_factor

    def survival(self, x):
        """ Probability of survival: P(X > x). """

        weibull = sps.exponweib.sf(x, self.l, self.k, loc=self.loc, scale=self.scale)
        return weibull



# -------------
# WEIBULL CLASS
# -------------

class standardWeibull:

    def __init__(self):

        self.l = 1.0
        self.k = 1.0
        self.scale_factor = 0.0

        self.gof = 0.0  # goodness of fit

    def fit(self, data):
        """ Maximum likelihood (MLE estimation) of the pdf parameters. """

        # function to fit: negative log likelihood of the pmf/pdf
        def target(params, x):
            l, k = params[0], params[1]
            weibull = (k / l) * ((x / l) ** (k - 1)) * np.exp(- (x / l) ** k)
            for i, e in enumerate(weibull):
                # if e < 0: weibull[i] == 0.0
                if e < 0: weibull[i] == 0.0
            nll = - np.sum(np.log(weibull))
            return nll

        # scale data to range [1, +inf[ (to avoid overflows)
        n_data = data / min(data)

        # fit function and store parameters
        result = minimize(target, x0=np.ones(2), args=(n_data,), method='Powell')  # Powell, does not require derivatives
        self.l = result.x[0]
        self.k = result.x[1]
        self.scale_factor = min(data)

        # goodness of fit (negative log likelihood)
        self.gof = target([self.l, self.k], n_data)

    def pdf(self, x):
        """ Probability density function, l is the rate. """

        # scale x
        x = x / self.scale_factor

        weibull = (self.k / self.l) * ((x / self.l) ** (self.k - 1)) * np.exp(- (x / self.l) ** self.k)
        if type(weibull) in [list, np.ndarray]:
            for i, e in enumerate(weibull):
                if e < 0.0: weibull[i] = 0.0
        else:
            weibull = max(0.0, weibull)
        return weibull

    def cdf(self, x):
        """ Cumulative density function. """

        # scale x
        x = x / self.scale_factor

        weibull = 1 - np.exp(- (x / self.l) ** self.k)
        if type(weibull) in [list, np.ndarray]:
            for i, e in enumerate(weibull):
                if e < 0.0: weibull[i] = 0.0
        else:
            weibull = max(0.0, weibull)
        return weibull

    def compute_threshold(self, alpha=0.9):
        """ Compute the threshold-value containing \alpha percent of examples. """

        return (self.l * np.power(-np.log((1 - alpha)), 1 / self.k)) * self.scale_factor

    def survival(self, x):
        """ Probability of survival: P(X > x). """

        return 1.0 - self.cdf(x)
