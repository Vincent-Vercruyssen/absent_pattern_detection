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
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .utils.validation import check_time_series


# -------------
# CLASSES
# -------------

class FingerprintDetector():
    """ Find occurrences of a pattern in the time series """

    def __init__(self,
                 features='all',
                 n_clusters=10,
                 warping_width=0.1,
                 alpha=0.5,
                 detector_type='dtw',  # 'dtw', 'feature'
                 tol=1e-8, verbose=False):

        # class parameters
        self.features = str(features)
        self.n_clusters = int(n_clusters)
        self.warping_width = float(warping_width)
        self.alpha = float(alpha)
        self.detector_type = str(detector_type)

        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict_fingerprints(self, timestamps, time_series, y):
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
        self.fit_fingerprints(timestamps, time_series, y)
        return self.exact_locations

    def fit_fingerprints(self, timestamps, time_series, y):
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
        user_labeled_ranges = ranges.copy()
        self._average_length = np.mean([len(r) for r in user_labeled_ranges])
        self._average_deviation = np.std([len(r) for r in user_labeled_ranges])
        patterns = np.array([time_series[ixs] for ixs in ranges])
        self.w_size = self._find_window_size(ranges)

        # 2. find the shape templates: the raw patterns or feature vectors
        if self.detector_type == 'dtw':
            # also normalize the patterns
            self.shape_templates = []
            for i, p in enumerate(patterns):
                m, s = np.mean(p), np.std(p)
                if s == 0.0:
                    self.shape_templates.append(p)
                else:
                    self.shape_templates.append((p - m) / s)
            self.shape_templates = np.array(self.shape_templates)

        elif self.detector_type == 'feature':
            # the shape templates are feature vectors
            self.pattern_templates = self._find_shape_templates(patterns)
            features = self._construct_features_and_labels(times, ts, self.pattern_templates, ranges, labels=False)
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
            """ TODO: this is a bit ad-hoc! """
            labels = np.array([ix[0] for ix in ranges])
            self.shape_templates = features[labels, :].copy()

        else:
            pass

        # 3. apply the shape templates to find determine the detection threshold
        self.max_threshold = self._determine_fingerprint_threshold(self.shape_templates)

        # 4. detect the remaining occurrences using the threshold
        if self.detector_type == 'dtw':
            segments = np.zeros((int(n_samples-self.w_size)+1, int(self.w_size)), dtype=float)
            for i in range(int(n_samples-self.w_size)+1):
                segment = ts[i:int(i+self.w_size)]
                segments[i, :] = segment
            scaler = StandardScaler()
            segments = scaler.fit_transform(segments.T).T
        elif self.detector_type == 'feature':
            segments = features
        else:
            pass
        self.exact_locations = self._find_exact_fingerprint_locations(segments, n_samples, self.w_size, self.shape_templates)

        """ FIX: correction if it does not find all labeled ranges """
        ix_found = np.where(self.exact_locations > 0.0)[0]
        if len(ix_found) < len(user_labeled_ranges):
            print('The PatternDetector does not pick up on all given occurrences, only:', len(ix_found), '/', len(user_labeled_ranges))
            for r in user_labeled_ranges:
                ixr = int((r[-1] + r[0]) / 2)
                self.exact_locations[ixr] = 1.0

        return self

    def predict_fingerprints(self, timestamps, time_series):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns exact_locations : np.array(), shape (n_samples)
            Exact locations of each pattern: 1 = pattern, 0 = no pattern.
        """

        times, ts, _ = check_time_series(timestamps, time_series, None)
        n_samples = len(ts)

        # 1. construct the feature vectors (if needed)
        if self.detector_type == 'dtw':
            segments = np.zeros((int(n_samples-self.w_size)+1, int(self.w_size)), dtype=float)
            for i in range(int(n_samples-self.w_size)+1):
                segment = ts[i:int(i+self.w_size)]
                segments[i, :] = segment
            scaler = StandardScaler()
            segments = scaler.fit_transform(segments.T).T
        elif self.detector_type == 'feature':
            features = self._construct_features_and_labels(times, ts, self.pattern_templates, labels=False)
            segments = self.scaler.transform(features)
        else:
            pass

        # 2. predict the occurrences
        exact_locations = self._find_exact_fingerprint_locations(segments, n_samples, self.w_size, self.shape_templates)

        return exact_locations

    def _determine_fingerprint_threshold(self, shapes):
        """ Determine the max threshold """

        thresholds = []
        for i, s1 in enumerate(shapes):
            for j, s2 in enumerate(shapes):
                if i == j:
                    continue
                else:
                    if self.detector_type == 'dtw':
                        d = dtw.distance(s1, s2, use_c=True, window=int(self.warping_width * self.w_size))
                    elif self.detector_type == 'feature':
                        d = np.linalg.norm(s1 - s2)
                    else:
                        pass
                    thresholds.append(d)
        """ TODO: ad-hoc trimming of outliers """
        thresholds = np.sort(np.array(thresholds))
        return np.amax(thresholds[:int(0.9 * len(thresholds))])

    def _find_exact_fingerprint_locations(self, segments, n, w, shapes):
        """ Find the exact locations """

        ns = len(segments)
        w = int(w)
        w1 = int(w / 2) - 1
        w2 = w - w1 - 1

        # 1. fingerprint detection
        pattern_locations = np.zeros(n, dtype=np.float)
        for _, shape in tqdm(enumerate(shapes), disable=not(self.verbose)):
            # distance between each segment and a shape (sliding window with w_increment = 1)
            dists = np.zeros(n, dtype=np.float)
            for i in range(ns):
                v = segments[i, :]
                if self.detector_type == 'dtw':
                    dists[i] = dtw.distance(shape, v, use_c=True, window=int(self.warping_width * w))
                elif self.detector_type == 'feature':
                    dists[i] = np.linalg.norm(shape - v)
                else:
                    pass
            new_locations = np.zeros(n, dtype=np.float)
            new_locations[dists < self.max_threshold] = 1.0
            pattern_locations += new_locations
        pattern_locations = np.minimum(pattern_locations, 1.0)

        """ FIX: it is possible that the threshold is too high!!! and everything is considered to be the pattern """
        """ THIS IS NOT FOOL-PROOF: it could be that the thresholds on the IF-tests are not strong enough """
        correction = False
        pattern_ranges = self._find_pattern_ranges(pattern_locations)
        if len(pattern_ranges) == 1 and len(pattern_ranges[0]) > self._average_length * 10:  # HIGHLY suspicious
            correction = True
        if abs(len(np.where(pattern_locations > 0.0)[0]) - ns) < self._average_length:
            correction = True
        if correction:
            # no patterns found because it is NOT discriminative enough
            exact_locations = np.zeros(n, dtype=np.float)
            return exact_locations

        # 2. only keep the center occurrence if multiple detected
        range_ixs = np.array([int((ix[0] + ix[-1]) / 2) for ix in pattern_ranges])
        remove = []
        for i, ix in enumerate(range_ixs):
            if i > 1:
                if ix - prev_ix < w:
                    remove.append(ix)
                    continue
            prev_ix = ix
        remove = np.array(remove)
        locs = np.setdiff1d(range_ixs, remove)

        # 3. exact locations + shift with window size / 2
        exact_locations = np.zeros(n, dtype=float)
        exact_locations[locs] = 1.0
        exact_locations = np.pad(exact_locations[:-w+1], (w1, w2), 'constant', constant_values=(0.0, 0.0))

        return exact_locations

    def _construct_features_and_labels(self, times, ts, shape_templates, ranges=None, labels=True):
        """ Construct the feature vectors and labels.

        :returns features : array, shape (n_segments, n_features)
            Feature vectors constructed from the the time series.
        :returns labels : array, shape (n_segments)
            Labels for the constructed segments.
        """

        # construct the segments
        n = len(ts)
        segments = np.zeros((int(n-self.w_size)+1, int(self.w_size)), dtype=float)
        for i in range(int(n-self.w_size)+1):
            segment = ts[i:int(i+self.w_size)]
            segments[i, :] = segment

        # construct feature vectors and labels
        features = self._construct_feature_matrix(segments, times, int(n-self.w_size)+1, shape_templates)
        if labels:
            labels = self._construct_labeling(ranges, int(n-self.w_size)+1)
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
        return round(np.mean(lengths))

    def _rolling_sum_array(self, arr, w):
        """ Rolling sum of the array using the window """
        n = len(arr)
        cs = np.cumsum(arr)
        return cs[int(w)-1:] - np.concatenate((np.array([0.0]), cs[:n-int(w)]))
