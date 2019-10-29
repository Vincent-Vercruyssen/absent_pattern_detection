""" Matrix profile anomaly detection.

Reference:
    Yeh, C. C. M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H. A., Keogh, E. (2016, December).
    Matrix profile I: all pairs similarity joins for time series: a unifying view that includes motifs, discords and shapelets.
    In Data Mining (ICDM), 2016 IEEE 16th International Conference on (pp. 1317-1322). IEEE.

"""

# Authors: Vincent Vercruyssen, 2018.

import math
import numpy as np
import pandas as pd
import scipy.signal as sps
from tqdm import tqdm

from .BaseDetector import BaseDetector


# -------------
# CLASSES
# -------------

class MatrixProfileAD(BaseDetector):
    """ Anomaly detection in time series using the matrix profile

    Parameters
    ----------
    m : int (default=10)
        Window size.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    Comments
    --------
    - This only works on time series data.
    """

    def __init__(self, m=10, contamination=0.1,
                 tol=1e-8, verbose=False):
        super(MatrixProfileAD, self).__init__()

        self.m = int(m)
        self.contamination = float(contamination)

        self.tol = float(tol)
        self.verbose = bool(verbose)

    def ab_join(self, T, split):
        """ Compute the ABjoin and BAjoin side-by-side,
            where `split` determines the splitting point.
        
        """

        # algorithm options
        excZoneLen = int(np.round(self.m * 0.5))
        radius = 1.1
        dataLen = len(T)
        proLen = dataLen - self.m + 1

        # change Nan and Inf to zero
        T = np.nan_to_num(T)

        # precompute the mean, standard deviation
        s = pd.Series(T)
        dataMu = s.rolling(self.m).mean().values[self.m-1:dataLen]
        dataSig = s.rolling(self.m).std().values[self.m-1:dataLen]

        matrixProfile = np.ones(proLen) * np.inf
        idxOrder = excZoneLen + np.arange(0, proLen, 1)
        idxOrder = idxOrder[np.random.permutation(len(idxOrder))]

        # construct the matrixprofile
        for i, idx in enumerate(idxOrder):
            # query
            query = T[idx:idx+self.m-1]
            
            # distance profile
            distProfile = self._diagonal_dist(T, idx, dataLen, self.m, proLen, dataMu, dataSig)
            distProfile = abs(distProfile)
            distProfile = np.sqrt(distProfile)

            # position magic
            pos1 = np.arange(idx, proLen, 1)
            pos2 = np.arange(0, proLen-idx+1, 1)

            # split magic
            distProfile = distProfile[np.where((pos2 <= split) & (pos1 > split))[0]]
            pos1Split = pos1[np.where((pos2 <= split) & (pos1 > split))[0]]
            pos2Split = pos2[np.where((pos2 <= split) & (pos1 > split))[0]]
            pos1 = pos1Split
            pos2 = pos2Split

            # update magic
            updatePos = np.where(matrixProfile[pos1] > distProfile)[0]
            matrixProfile[pos1[updatePos]] = distProfile[updatePos]
            updatePos = np.where(matrixProfile[pos2] > distProfile)[0]
            matrixProfile[pos2[updatePos]] = distProfile[updatePos]

        return matrixProfile


    def fit_predict(self, T):
        """ Fit the model to the time series T.

        :param T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the samples in T.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        return self.fit(np.array([])).predict(T)

    def fit(self, T=np.array([])):
        """ Fit the model to the time series T.

        :param T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.

        :returns self : object
        """

        self.T_train = T

        return self

    def predict(self, T=np.array([])):
        """ Compute the anomaly score + predict the label of each sample in T.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the samples in T.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        # fuse T_train and T
        nt = len(T)
        nT = np.concatenate((self.T_train, T))
        n = len(nT)

        # compute the matrix profile
        matrix_profile = self._compute_matrix_profile_stomp(nT, self.m)

        # transform to an anomaly score (1NN distance)
        # the largest distance = the largest anomaly
        # rescale between 0 and 1, this yields the anomaly score
        y_score = (matrix_profile - min(matrix_profile)) / (max(matrix_profile) - min(matrix_profile))
        y_score = np.append(y_score, np.zeros(n-len(matrix_profile), dtype=float))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(n * (1.0 - self.contamination))]
        y_pred = np.ones(n, dtype=float)
        y_pred[y_score < self.threshold] = -1

        # cut y_pred and y_score to match length of T
        return y_score[-nt:], y_pred[-nt:]

    def _compute_matrix_profile_stomp(self, T, m):
        """ Compute the matrix profile and profile index for time series T using correct STOMP.

        :param T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.
        :param m : int
            Length of the query.

        :returns matrix_profile : np.array(), shape (n_samples)
            The matrix profile (distance) for the time series T.

        comments
        --------
        - Includes a fix for straight line time series segments.
        """

        n = len(T)

        # precompute the mean, standard deviation
        s = pd.Series(T)
        data_m = s.rolling(m).mean().values[m-1:n]
        data_s = s.rolling(m).std().values[m-1:n]

        # where the data is zero
        idxz = np.where(data_s < 1e-8)[0]
        data_s[idxz] = 0.0
        idxn = np.where(data_s > 0.0)[0]

        zero_s = False
        if len(idxz) > 0:
            zero_s = True

        # precompute distance to straight line segment of 0s
        slD = np.zeros(n-m+1, dtype=float)
        if zero_s:
            for i in range(n-m+1):
                Tsegm = T[i:i+m]
                Tm = data_m[i]
                Ts = data_s[i]
                if Ts == 0.0:  # data_s is effectively 0
                    slD[i] = 0.0
                else:
                    Tn = (Tsegm - Tm) / Ts
                    slD[i] = np.sqrt(np.sum(Tn ** 2))

        # compute the first dot product
        q = T[:m]
        QT = sps.convolve(T.copy(), q[::-1], 'valid', 'direct')
        QT_first = QT.copy()

        # compute the distance profile
        D = self._compute_fixed_distance_profile(T[:m], QT, n, m, data_m, data_s, data_m[0], data_s[0], slD.copy(), idxz, idxn, zero_s)

        # initialize matrix profile
        matrix_profile = D

        # in-order evaluation of the rest of the profile
        for i in tqdm(range(1, n-m+1, 1), disable=not(self.verbose)):
            # update the dot product
            QT[1:] = QT[:-1] - (T[:n-m] * T[i-1]) + (T[m:n] * T[i+m-1])
            QT[0] = QT_first[i]

            # compute the distance profile: without function calls!
            if data_s[i] == 0.0:  # query_s is effectively 0
                D = slD.copy()
            elif zero_s:
                D[idxn] = np.sqrt(2 * (m - (QT[idxn] - m * data_m[idxn] * data_m[i]) / (data_s[idxn] * data_s[i])))
                nq = (q - data_m[i]) / data_s[i]
                d = np.sqrt(np.sum(nq ** 2))
                D[idxz] = d
            else:
                D = np.sqrt(2 * (m - (QT - m * data_m * data_m[i]) / (data_s * data_s[i])))

            # update the matrix profile
            exclusion_range = (int(max(0, round(i-m/2))), int(min(round(i+m/2+1), n-m+1)))
            D[exclusion_range[0]:exclusion_range[1]] = np.inf

            ix = np.where(D < matrix_profile)[0]
            matrix_profile[ix] = D[ix]
            # matrix_profile = np.minimum(matrix_profile, D)

        return matrix_profile

    def _compute_fixed_distance_profile(self, q, QT, n, m, data_m, data_s, query_m, query_s, slD, idxz, idxn, zero_s):
        """ Compute the fixed distance profile """
        D = np.zeros(n-m+1, dtype=float)

        if query_s == 0.0:  # query_s is effectively 0
            return slD

        if zero_s:
            D[idxn] = np.sqrt(2 * (m - (QT[idxn] - m * data_m[idxn] * query_m) / (data_s[idxn] * query_s)))
            nq = (q - query_m) / query_s
            d = np.sqrt(np.sum(nq ** 2))
            D[idxz] = d
        else:
            D = np.sqrt(2 * (m - (QT - m * data_m * query_m) / (data_s * query_s)))

        return D

    def _compute_matrix_profile_stamp(self, T, m):
        """ Compute the matrix profile and profile index for time series T using STAMP.

        :param T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.
        :param m : int
            Length of the query.

        :returns matrix_profile : np.array(), shape (n_samples)
            The matrix profile (distance) for the time series T.
        :returns profile_index : np.array(), shape (n_samples)
            The matrix profile index accompanying the matrix profile.

        comments
        --------
        - Uses the STAMP algorithm to compute the matrix profile.
        - Includes a fix for straight line time series segments.
        """

        n = len(T)

        # initialize the empty profile and index
        matrix_profile = np.ones(n-m+1) * np.inf

        # precompute the mean, standard deviation
        s = pd.Series(T)
        data_m = s.rolling(m).mean().values[m-1:n]
        data_s = s.rolling(m).std().values[m-1:n]

        # where the data is zero
        idxz = np.where(data_s < 1e-8)[0]
        data_s[idxz] = 0.0
        idxn = np.where(data_s > 0.0)[0]

        zero_s = False
        if len(idxz) > 0:
            zero_s = True

        # precompute distance to straight line segment of 0s
        # brute force distance computation (because the dot_product is zero!)
        # --> this is a structural issue with the MASS algorithm for fast distance computation
        slD = np.zeros(n-m+1, dtype=float)
        if zero_s:
            for i in range(n-m+1):
                Tsegm = T[i:i+m]
                Tm = data_m[i]
                Ts = data_s[i]
                if Ts == 0.0:  # data_s is effectively 0
                    slD[i] = 0.0
                else:
                    Tn = (Tsegm - Tm) / Ts
                    slD[i] = np.sqrt(np.sum(Tn ** 2))

        # random search order for the outer loop
        indices = np.arange(0, n-m+1, 1)
        np.random.shuffle(indices)

        # compute the matrix profile
        if self.verbose: print('Iterations:', len(indices))
        for i, idx in tqdm(enumerate(indices), disable=not(self.verbose)):
            # query for which to compute the distance profile
            query = T[idx:idx+m]

            # normalized distance profile (using MASS)
            D = self._compute_MASS(query, T, n, m, data_m, data_s, data_m[idx], data_s[idx], slD.copy())

            # update the matrix profile (keeping minimum distances)
            # self-join is True! (we only look at constructing the matrix profile for a single time series)
            exclusion_range = (int(max(0, round(idx-m/2))), int(min(round(idx+m/2+1), n-m+1)))
            D[exclusion_range[0]:exclusion_range[1]] = np.inf

            ix = np.where(D < matrix_profile)[0]
            matrix_profile[ix] = D[ix]

        return matrix_profile

    def _compute_MASS(self, query, T, n, m, data_m, data_s, query_m, query_s, slD):
        """ Compute the distance profile using the MASS algorithm.

        :param query : np.array(), shape (self.m)
            Query segment for which to compute the distance profile.
        :param T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.
        :param n : int
            Length of time series T.
        :param m : int
            Length of the query.
        :param data_f : np.array, shape (n + m)
            FFT transform of T.
        :param data_m : np.array, shape (n - m + 1)
            Mean of every segment of length m of T.
        :param data_s : np.array, shape (n - m + 1)
            STD of every segment of length m of T.
        :param query_m : float
            Mean of the query segment.
        :param query_s : float
            Standard deviation of the query segment.

        :returns dist_profile : np.array(), shape (n_samples)
            Distance profile of the query to time series T.
        """

        # CASE 1: query is a straight line segment of 0s
        if query_s < 1e-8:
            return slD

        # CASE 2: query is every other possible subsequence
        # compute the sliding dot product
        reverse_query = query[::-1]
        dot_product = sps.fftconvolve(T, reverse_query, 'valid')

        # compute the distance profile without correcting for standard deviation of the main signal being 0
        # since this is numpy, it will result in np.inf if the data_sig is 0
        dist_profile = np.sqrt(2 * (m - (dot_product - m * query_m * data_m) / (query_s * data_s)))

        # correct for data_s being 0
        zero_idxs = np.where(data_s < 1e-8)[0]
        if len(zero_idxs) > 0:
            n_query = (query - query_m) / query_s
            d = np.linalg.norm(n_query - np.zeros(m, dtype=float))
            dist_profile[zero_idxs] = d

        return dist_profile

    def _compute_brute_force_distance_profile(self, query, T, n, m, data_f, data_m, data_s, query_m, query_s):
        """ Compute the brute force distance profile. """

        dist_profile = np.zeros(n-m+1, dtype=float)

        # normalize query
        if query_m < 1e-8:
            n_query = np.zeros(m, dtype=float)
        else:
            n_query = (query - query_m) / query_s

        # compute the distance profile
        for i in range(n-m+1):
            T_segm = T[i:i+m]
            Tm = data_m[i]
            Ts = data_s[i]
            # normalize time series segment
            if Ts < 1e-8:
                T_norm = np.zeros(m, dtype=float)
            else:
                T_norm = (T_segm - Tm) / Ts
            # compute distance
            dist_profile[i] = np.linalg.norm(T_norm - n_query)

        return dist_profile

    def _diagonal_dist(self, data, idx, dataLen, subLen, proLen, dataMu, dataSig):
        """ Compute the diagonal distance (as in the original matrix profile code) """
        
        xTerm = np.dot(np.ones(proLen-idx+1), np.dot(data[idx-1:idx+subLen-1], data[:subLen]))
        mTerm = data[idx-1:proLen-1] * data[:proLen-idx]
        aTerm = data[idx+subLen-1:] * data[subLen:dataLen-idx+1]
        if proLen != idx:
            xTerm[1:] = xTerm[1:] - np.cumsum(mTerm) + np.cumsum(aTerm)

        distProfile = np.divide(xTerm - subLen * dataMu[idx-1:] * dataMu[:proLen-idx+1],
                            subLen * dataSig[idx-1:] * dataSig[:proLen-idx+1])
        distProfile = 2 * subLen * (1 - distProfile)
