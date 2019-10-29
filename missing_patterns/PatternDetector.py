""" PatternDetector: train classifier to find pattern occurrences in time series.
"""

# Authors: Vincent Vercruyssen, 2018.

import sys, os, time
import numpy as np
import pandas as pd
import scipy.stats as sps
from datetime import datetime
from collections import Counter
from tqdm import tqdm

from dtaidistance import dtw
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .utils.validation import check_time_series

# TODO: nicer solution than this
script_loc, _ = os.path.split(os.path.realpath(__file__))
sys.path.insert(0, os.path.split(script_loc)[0])

from classifiers.BaggingSVM import BaggingSVM
from classifiers.BaggingRF import BaggingRF
#from classifiers.BaggingXGBoost import BaggingXGBoost


# -------------
# CLASSES
# -------------

class PatternDetector():
    """ Find occurrences of a pattern in the time series """

    def __init__(self,
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
                 smoothing='convolve',  # 'convolve', 'low-pass'
                 tol=1e-8, verbose=False):

        # class parameters
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

        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict_occurrences(self, timestamps, time_series, y):
        """ Find all pattern occurrences in the time series

        :param timestamps : np.array(), shape (n_samples)
            The timestamps of the time series, datetime.datetime objects.
        :param time_series : np.array(), shape (n_samples, n_variables)
            The measured time series data.
        :param y : np.array(), shape (n_samples)
            Indicates the user-labeled y of a pattern (0, 1).

        :returns exact_locations : np.array(), shape (n_samples)
            Exact locations of each pattern: 1 = pattern, 0 = no pattern.
        """

        # TODO: make better, this is just to save time on the feature construction
        self.fit_occurrences(timestamps, time_series, y)
        return self.exact_locations

    def fit_occurrences(self, timestamps, time_series, y):
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

        # 1. determine the exact locations of the known patterns + window size
        t = time.time()
        ranges = self._find_pattern_ranges(y)
        self.w_size = self._find_window_size(ranges)
        if self.verbose: print('Finished: window size')

        # 2. find the shape templates
        t = time.time()
        patterns = np.array([time_series[ixs] for ixs in ranges])
        self.shape_templates = self._find_shape_templates(patterns)
        if self.verbose: print('Finished: shape templates')

        # 3. construct the feature vectors and labels for the classifier
        t = time.time()
        features, labels = self._construct_features_and_labels(times, ts, self.shape_templates, ranges, labels=True)
        n_segments = features.shape[0]
        if self.verbose: print('Finished: feature construction')

        # 4. fit and predict the classifier
        t = time.time()
        if self.clf_scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.clf_scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.clf_scaling == 'robust':
            self.scaler = RobustScaler()
        else:
            pass
        features = self.scaler.fit_transform(features)
        if self.bagging_classifier == 'SVM':
            self.classifier = BaggingSVM(kernel_type=self.bagging_kernel, K=self.bagging_K, T=self.bagging_T, class_weights=self.bagging_cweights, feature_names=self.feature_names)
        elif self.bagging_classifier == 'RF':
            self.classifier = BaggingRF(class_weights=self.bagging_cweights, feature_names=self.feature_names)
        elif self.bagging_classifier == 'XGBoost':
            pass
            #self.classifier = BaggingXGBoost(class_weights=self.bagging_cweights, feature_names=self.feature_names)
        else:
            pass
        y_prob, y_pred = self.classifier.fit_predict(features, labels)  # TODO: this is to save time...
        if self.verbose: print('Finished: classifier construction')

        # 5. NEW: find the exact locations of the patterns and the detection threshold
        self.exact_locations, roll_prob = self._find_exact_pattern_locations(y_pred, n_samples, self.w_size, ranges)
        if self.verbose: print('Finished: finding exact locations')

        # # OLD: 5. prepare final predictions
        # y_pred = np.pad(y_pred, (int(round(self.w_size / 2)), 0), 'constant', constant_values=(0.0, 0.0))[:n_segments]
        # y_prob = np.pad(y_prob, (int(round(self.w_size / 2)), 0), 'constant', constant_values=(0.0, 0.0))[:n_segments]
        # self.exact_locations = self._find_exact_pattern_locations(y_pred, y_prob, n_samples, ranges)

        return self

    def predict_occurrences(self, timestamps, time_series):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns exact_locations : np.array(), shape (n_samples)
            Exact locations of each pattern: 1 = pattern, 0 = no pattern.
        """

        times, ts, _ = check_time_series(timestamps, time_series, None)
        n_samples = len(ts)

        # 1. construct the feature vectors
        features = self._construct_features_and_labels(times, ts, self.shape_templates, labels=False)
        n_segments = features.shape[0]

        # 2. predict the occurrences
        features = self.scaler.transform(features)
        y_prob, y_pred = self.classifier.predict(features)

        # 3. NEW: find the exact locations of the patterns and the detection threshold
        exact_locations, roll_prob = self._find_exact_pattern_locations(y_pred, n_samples, self.w_size)

        return exact_locations #, roll_prob

    def _find_exact_pattern_locations(self, y_prob, n, w, ranges=[]):
        """ Find the exact pattern locations: bit hacky.
            Pad with zeros to achieve the desired length.

        :returns exact_locations : np.array(), shape (n_samples)
            Exact locations of each pattern: 1 = pattern, 0 = no pattern.
        """

        # parameters
        ny = len(y_prob)
        w = int(w)
        w1 = int(w / 2) - 1
        w2 = w - w1 - 1

        # 1. append 0s to the end of the prob
        y_prob = np.pad(y_prob, (0, w - 1), 'constant', constant_values=(0.0, 0.0))
        assert len(y_prob) == n, 'y_prob padded :: wrong dimensions'

        # 2. compute the rolling sum + pad with zeros on both sides
        if self.smoothing == 'convolve':
            roll_prob = self._rolling_sum_array(y_prob, int(w))
            roll_prob = np.pad(roll_prob, (w1, w2), 'constant', constant_values=(0.0, 0.0))
            assert len(roll_prob) == n, 'roll_prob :: wrong dimensions :: {} - {}'.format(len(roll_prob), n)

        elif self.smoothing == 'low-pass':
            # try: choosing filter that minimizes standarddeviation of threshold
            if len(ranges) > 0:
                best_std = np.inf
                for order in np.arange(1, 20, 1):
                    for wn in np.arange(0.01, 0.71, 0.01):
                        b, a = butter(order, wn)
                        roll_prob_temp = filtfilt(b, a, y_prob)
                        new_t, new_std = self._determine_threshold(roll_prob_temp, ranges, True)
                        if new_t < self.tol:
                            continue
                        if new_std <= best_std:
                            best_std = new_std
                            self.best_b = b
                            self.best_a = a
            roll_prob = filtfilt(self.best_b, self.best_a, y_prob)

        else:
            pass

        # 2.5 shift roll_prob such that it aligns with the given patterns!
        roll_prob = np.pad(roll_prob, (int(w / 2), 0), 'constant', constant_values=(0.0, 0.0))[:n]
        assert len(roll_prob) == n, 'roll_prob 2 :: not the right length'

        # 3. compute the threshold (only training)
        if len(ranges) > 0:
            self.threshold = self._determine_threshold(roll_prob, ranges)

        # 4. determine the possible locations
        possible_locations = np.zeros(n, dtype=float)
        possible_locations[roll_prob < self.threshold] = 0.0
        possible_locations[roll_prob >= self.threshold] = 1.0

        # 5. narrow the exact locations
        loc_ranges = self._find_pattern_ranges(possible_locations)
        ixs = np.array([int((lr[0] + lr[-1]) / 2) for lr in loc_ranges])
        exact_locations = np.zeros(n, dtype=float)
        if len(ixs) > 0:
            exact_locations[ixs] = 1.0

        # 6. fix the ranges that have not been detected (only training)
        if len(ranges) > 0:
            if len(ixs) < len(ranges):
                print('The PatternDetector does not pick up on all given occurrences, only:', len(ixs), '/', len(ranges))
                for ranger in ranges:
                    rl = int((ranger[-1] + ranger[0]) / 2)
                    exact_locations[rl] = 1.0

        # 6. make sure the pattern does not overlap with itself
        ixs = np.where(exact_locations == 1.0)[0]
        if len(ixs) > 0:
            remove = []
            for i, ix in enumerate(ixs):
                if i > 1:
                    if ix - prev_ix < w:
                        remove.append(ix)
                        continue
                prev_ix = ix
            remove = np.array(remove)
            if len(remove) > 0:
                exact_locations[remove] = 0.0

        return exact_locations, roll_prob

    def _determine_threshold(self, roll_prob, ranges, std=False):
        """ Determine the detection threshold given the known occurrences. """

        possible_thresholds = []
        for ranger in ranges:
            possible_thresholds.append(np.amax(roll_prob[ranger]))
        possible_thresholds = np.array(possible_thresholds)
        # TODO: multiplier is arbitrary
        t = 1.0 * np.amin(possible_thresholds[np.nonzero(possible_thresholds)])
        if std:
            std = np.std(possible_thresholds[np.nonzero(possible_thresholds)])
            return t, std
        return t

    def _construct_features_and_labels(self, times, ts, shape_templates, ranges=None, labels=True):
        """ Construct the feature vectors and labels.

        :returns features : array, shape (n_segments, n_features)
            Feature vectors constructed from the the time series.
        :returns labels : array, shape (n_segments)
            Labels for the constructed segments.
        """

        # construct the segments
        n = len(ts)
        segments = np.zeros((n-self.w_size+1, self.w_size), dtype=float)
        for i in range(n-self.w_size+1):
            segment = ts[i:i+self.w_size]
            segments[i, :] = segment

        # construct feature vectors and labels
        features = self._construct_feature_matrix(segments, times, n-self.w_size+1, shape_templates)
        if labels:
            labels = self._construct_labeling(ranges, n-self.w_size+1)
            return features, labels
        else:
            return features

    def _construct_labeling(self, ranges, n):
        """ Construct labels for the segments. Rules:
            1. if contained in labeled segment: give it that label
            2. if not contained but overlapping: ignore later on
            3. if not contained and not overlapping: unlabeled

        :returns labels : array, shape (n_segments)
            Labels for the constructed segments.
        """

        # we ignore all segments that contain only a part of the pattern unless they are fully overlapped by the pattern
        labeling = np.zeros(n, dtype=float)
        pattern_locations = [[ixs[0], ixs[-1]+1] for ixs in ranges]
        for _, v in enumerate(pattern_locations):
            """FIX: does not work when the final pattern is too close to the end of the series """
            b = v[1] - self.w_size
            if b < v[0]:
                # pattern is shorter than w_size: every segment fully containing the pattern is pos
                pos = np.arange(b, v[0]+1, 1)
                ign = np.concatenate((np.arange(v[0]-self.w_size+1, b, 1),
                                     np.arange(v[0]+1, v[1], 1)))
            else:
                # pattern is longer than w_size: every segment fully contained in the pattern is pos
                pos = np.arange(v[0], v[1]-self.w_size+1, 1)
                ign = np.concatenate((np.arange(v[0]-self.w_size+1, v[0], 1),
                                     np.arange(v[1]-self.w_size+1, v[1], 1)))
            # cut the indices that would label segments falling outside the allowed number: n
            pos = pos[pos < n]
            ign = ign[ign < n]
            # annotate
            labeling[pos.astype(int)] = 1.0
            labeling[ign.astype(int)] = -1.0

        return labeling

    def _construct_feature_matrix(self, segments, times, n, shape_templates):
        """ Construct the feature vectors.

        :returns features : array, shape (n_segments, n_features)
            Feature vectors constructed from the the time series.
        """

        if self.features == 'all':
            use_features = ['stat', 'time', 'shape']
        elif self.features == 'stat_time':
            use_features = ['stat', 'time']
        elif self.features == 'stat_shape':
            use_features = ['stat', 'shape']
        elif self.features == 'time_shape':
            use_features = ['time', 'shape']
        else:
            use_features = [self.features]  # stat, time, shape

        # summary statistics
        if 'stat' in use_features:
            if self.verbose: print('\tconstructing statistics...')
            avg = pd.Series(np.mean(segments, axis=1))
            std = pd.Series(np.std(segments, axis=1))
            vari = pd.Series(np.var(segments, axis=1))
            maxi = pd.Series(np.amax(segments, axis=1))
            mini = pd.Series(np.amin(segments, axis=1))
            med = pd.Series(np.median(segments, axis=1))
            tsum = pd.Series(np.sum(segments, axis=1))
            skew = pd.Series(sps.describe(segments, axis=1, bias=False).skewness)
            kurt = pd.Series(sps.describe(segments, axis=1, bias=False).kurtosis)

        # time features
        if 'time' in use_features:
            if self.verbose: print('\tconstructing time features...')
            time_stamps = np.array([ts.hour for ts in times[:n]])
            xhr = pd.Series(np.sin(2 * np.pi * time_stamps / 24))
            yhr = pd.Series(np.cos(2 * np.pi * time_stamps / 24))

        # shape features
        if 'shape' in use_features:
            # find nearest euclid function
            def _find_nearest_euclid(a, b):
                if len(a) >= len(b):
                    short = b
                    long = a
                else:
                    short = a
                    long = b
                n1, n2 = len(long), len(short)
                d = np.zeros(n1-n2+1, dtype=float)
                for i in range(n1-n2+1):
                    d[i] = np.linalg.norm(long[i:i+n2] - short)
                return np.amin(d)

            # z-score normalization of the segments: per segment!
            if self.verbose: print('\tnormalizing...')
            scaler = StandardScaler()
            n_segments = scaler.fit_transform(segments.T).T

            if self.verbose: print('\tconstructing shape features...')
            c = 0
            shape_features = pd.DataFrame()
            for _, shape in tqdm(enumerate(shape_templates), disable=not(self.verbose)):  # shape templates are normalized!
                ########
                # DTW
                ########
                dists = np.zeros(n, dtype=float)
                for i, v in enumerate(n_segments):
                    dists[i] = dtw.distance(shape, v, use_c=True, window=int(self.warping_width * self.w_size))
                shape_features[c] = dists
                c += 1

            # for _, shape in tqdm(enumerate(shape_templates), disable=not(self.verbose)):  # shape templates are normalized!
            #     ########
            #     # EUCLID
            #     ########
            #     dists = np.zeros(n, dtype=float)
            #     for i, v in enumerate(n_segments):
            #         dists[i] = _find_nearest_euclid(shape, v)
            #     shape_features[c] = dists
            #     c += 1

        # combine features
        self.feature_names = []
        features = pd.DataFrame()
        if 'stat' in use_features:
            new_features = pd.concat([avg, std, vari, maxi, mini, med, tsum, skew, kurt], axis=1)
            #new_features = pd.concat([avg, std, vari, mini, med, skew, kurt], axis=1)
            features = pd.concat([features, new_features], axis=1)
            self.feature_names.append(['avg', 'std', 'vari', 'maxi', 'mini', 'med', 'tsum', 'skew', 'kurt'])
            #self.feature_names.append(['avg', 'std', 'vari', 'mini', 'med', 'skew', 'kurt'])
        if 'time' in use_features:
            new_features = pd.concat([xhr, yhr], axis=1)
            features = pd.concat([features, new_features], axis=1)
            self.feature_names.append(['time_x', 'time_y'])
        if 'shape' in use_features:
            features = pd.concat([features, shape_features], axis=1)
            self.feature_names.append(['shape'+str(i+1) for i in range(len(shape_templates))])

        self.feature_names = np.array([item for sublist in self.feature_names for item in sublist])

        return features.values

    def _find_shape_templates(self, patterns):
        """ Find the shape templates for the given patterns.

        :returns shape_templates : array of arrays
            Found shape templates.
        """

        # normalize the patterns
        norm_patterns = []
        for i, p in enumerate(patterns):
            # normalize v
            m, s = np.mean(p), np.std(p)
            if s == 0.0:
                norm_patterns.append(p)
            else:
                norm_patterns.append((p - m) / s)
        norm_patterns = np.array(norm_patterns)

        # calculate shape templates (depending on number of clusters)
        if len(patterns) > self.n_clusters:
            # DTW distance matrix
            dists = dtw.distance_matrix(norm_patterns, use_nogil=True, window=int(self.warping_width * self.w_size))
            dists[dists == np.inf] = 0
            dists = dists + dists.T - np.diag(np.diag(dists))
            affinities = np.exp(-dists * self.alpha)

            # spectral clustering
            spec = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed')
            spec.fit(affinities)
            split_labels = spec.labels_.astype(np.int)

            # find mediods
            centers = []
            for l in np.unique(split_labels):
                ix = np.where(split_labels == l)[0]
                if len(ix) == 1:
                    # there is only one pattern in the cluster
                    centers.append(norm_patterns[ix[0]])
                elif len(ix) == 2:
                    # there are 2 patterns in the cluster: select randomly
                    centers.append(norm_patterns[ix[0]])
                else:
                    # more than 2 patterns in the cluster
                    c = ix[np.argmin(np.sum(dists[ix, :], axis=1))] # select mediod
                    centers.append(norm_patterns[c])
            shape_templates = np.array(centers)

        else:
            shape_templates = norm_patterns

        return shape_templates

    def _find_exact_pattern_locations_old(self, y_pred, y_prob, n_samples, ranges=[]):
        """ Find the exact pattern locations: bit hacky.
            Pad with zeros to achieve the desired length.

        :returns exact_locations : np.array(), shape (n_samples)
            Exact locations of each pattern: 1 = pattern, 0 = no pattern.
        """

        # cumulative prediction
        """ the y_pred makes sure that it goes to 0? """
        pattern_support = np.zeros(len(y_pred), dtype=float)
        c = 0.0
        for i, e in enumerate(y_pred):
            c += e
            if e == 0.0:
                c = 0
            pattern_support[i] = c

        # fit to the known pattern occurrences
        if len(ranges) > 0:
            # peak values and detection threshold
            peak_vals = []
            for ranger in ranges:
                peak_vals.append(max(pattern_support[ranger]))  # TODO: somewhat ad-hoc (because patterns can have different lengths)
            #m, s = np.mean(peak_vals), np.std(peak_vals)
            #threshold = m - self.detection_threshold * s
            peak_vals = np.array(peak_vals)
            self.threshold = np.min(peak_vals[np.nonzero(peak_vals)])

        # exact pattern locations
        possible_locations = np.zeros(len(y_pred), dtype=float)
        possible_locations[pattern_support < self.threshold] = 0.0
        possible_locations[pattern_support >= self.threshold] = 1.0

        # TODO: NOW: exact locations in the beginning of the pattern...
        loc_ranges = self._find_pattern_ranges(possible_locations)
        ixs = np.array([lr[0] for lr in loc_ranges])
        # TODO: better way to deal with no patterns found
        exact_locations = np.zeros(len(y_pred), dtype=float)
        if len(ixs) > 0:
            exact_locations[ixs] = 1.0

        """ Preliminary fix that ensures that:
        - at least the indicated occurrences are found
        - at most the number of non-overlapping segments in the data
        """
        if len(ranges) > 0:
            ixs = np.where(exact_locations == 1.0)[0]
            if len(ixs) < len(ranges):
                print('The PatternDetector does not pick up on all given occurrences, only:', len(ixs), '/', len(ranges))
                for ranger in ranges:
                    rl = int((ranger[-1] + ranger[0]) / 2)
                    exact_locations[rl] = 1.0

        """ IMPORTANT IN THE PAPER """
        # a pattern cannot overlap itself!
        ixs = np.where(exact_locations == 1.0)[0]
        if len(ixs) > 0:
            remove = []
            prev_ix = ixs[0]
            for i, ix in enumerate(ixs):
                if i > 1:
                    if ix - prev_ix < self.w_size:
                        remove.append(ix)
                        continue
                prev_ix = ix
            remove = np.array(remove)
            if len(remove) > 0:
                exact_locations[remove] = 0.0

        # FIX: pad with zeros
        exact_locations = np.pad(exact_locations, (0, n_samples-len(exact_locations)), 'constant', constant_values=(0.0, 0.0))

        return exact_locations

    def _find_pattern_ranges(self, y):
        """ Find the indices of each free-range pattern.

        :returns ranges : array of arrays
            Each array corresponds to the indices of the pattern.
        """

        ranges = []
        ix = np.where(y == 1.0)[0]
        if len(ix) > 0:
            bp, ep = ix[0], ix[0]
            for i, e in enumerate(ix):
                if i == len(ix) - 1:
                    if e - ep > 1:
                        ranges.append(np.arange(bp, ep+1, 1))
                    else:
                        ranges.append(np.arange(bp, e+1, 1))
                elif e - ep > 1:
                    ranges.append(np.arange(bp, ep+1, 1))
                    bp = e
                ep = e
        return np.array(ranges)

    def _find_window_size(self, ranges):
        """ Gaussian heuristic to finding the window size. """

        lengths = []
        for _, r in enumerate(ranges):
            lengths.append(len(r))

        # fit gaussian (round to nearest int)
        # return integer
        return int(round(np.mean(lengths)))

    def _rolling_sum_array(self, arr, w):
        """ Rolling sum of the array using the window """
        n = len(arr)
        cs = np.cumsum(arr)
        return cs[int(w)-1:] - np.concatenate((np.array([0.0]), cs[:n-int(w)]))
