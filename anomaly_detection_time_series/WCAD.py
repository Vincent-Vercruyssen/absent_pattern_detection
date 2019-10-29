""" Window compression anomaly detection.

Reference:
    Keogh, E., Lonardi, S., & Ratanamahatana, C. A. (2004, August).
    Towards parameter-free data mining.
    In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 206-215). ACM.

"""

# Authors: Vincent Vercruyssen, 2018.

import os
import shutil
import zipfile
import numpy as np
from tqdm import tqdm

from .BaseDetector import BaseDetector


# -------------
# CLASSES
# -------------

class WCAD(BaseDetector):
    """ Anomaly detection in time series with window compression

    Parameters
    ----------
    m : int (default=10)
        Window size.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    Comments
    --------
    - This also works for attribute-value data, but it was introduced in the
    context of time series data.
    """

    def __init__(self, m=10, contamination=0.1,
                 tol=1e-8, verbose=False):
        super(WCAD, self).__init__()

        self.m = int(m)
        self.contamination = float(contamination)

        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict(self, T, y):
        """ Fit the model to the time series T.

        :param T : np.array(), shape (n_samples)
            The time series data for training (reference series).
        :param y : np.array(), shape (n_samples)
            The time series for which to compute anomaly score.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the samples in y.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        return self.fit(T).predict(y)

    def fit(self, T):
        """ Fit the model to the time series T.

        :param T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.

        :returns self : object
        """

        self.T = T

        return self

    def predict(self, y):
        """ Compute the anomaly score + predict the label of each sample in y.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the samples in T.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        # parameters
        n1 = len(self.T)
        n2 = len(y)

        # make temporary path
        temp_path = os.path.join(os.path.dirname(__file__), 'temp')
        if not(os.path.isdir(temp_path)):
            os.makedirs(temp_path)

        # compute the WCAD score (vis-a-vis the training data)
        # essentially this is again 1NN anomaly detection just using a compression metric
        distance_profile = np.zeros(n2-self.m+1, dtype=float)
        for i in tqdm(range(n2-self.m+1), disable=not(self.verbose)):
            query = y[i:i+self.m]
            best_d = np.inf
            for j in range(n1-self.m+1):
                cand = self.T[j:j+self.m]
                d = self._compression_based_distance(query, cand, temp_path)
                if d < best_d:
                    best_d = d
            distance_profile[i] = best_d

        # clean up temporary directory
        shutil.rmtree(temp_path)

        # transform to an anomaly score (1NN distance)
        # the largest distance = the largest anomaly
        # rescale between 0 and 1, this yields the anomaly score
        y_score = (distance_profile - min(distance_profile)) / (max(distance_profile) - min(distance_profile))
        y_score = np.append(y_score, np.zeros(n2-len(distance_profile), dtype=float))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(n2 * (1.0 - self.contamination))]
        y_pred = np.ones(n2, dtype=float)
        y_pred[y_score < self.threshold] = -1

        return y_score, y_pred

    def _compression_based_distance(self, t1, t2, temp_path):
        """ Compute the compression-based dissimilarity metric between t1 and t2.

        :param t1 : np.array()
            Time series segment.
        :param t2 : np.array()
            Time series segment.
        :param temp_path : str
            Name of the temporary path where to store the files.

        :returns cdm : float
            Compression-based dissimilarity metric
        """

        #TODO: make more efficient with https://newseasandbeyond.wordpress.com/2014/01/27/creating-in-memory-zip-file-with-python

        # 1. compress t1 and compute file size
        np.savetxt(os.path.join(temp_path, 't1.txt'), t1)
        zipf = zipfile.ZipFile(os.path.join(temp_path, 't1.zip'), 'w', zipfile.ZIP_DEFLATED)
        zipf.write(os.path.join(temp_path, 't1.txt'), 't1.zip')
        zipf.close()
        t1_bytes = os.path.getsize(os.path.join(temp_path, 't1.zip'))

        # 2. compress t2 and compute file size
        np.savetxt(os.path.join(temp_path, 't2.txt'), t2)
        zipf = zipfile.ZipFile(os.path.join(temp_path, 't2.zip'), 'w', zipfile.ZIP_DEFLATED)
        zipf.write(os.path.join(temp_path, 't2.txt'), 't2.zip')
        zipf.close()
        t2_bytes = os.path.getsize(os.path.join(temp_path, 't2.zip'))

        # 3. compress t1 and t2 and compute file size
        t12 = np.concatenate((t1, t2))
        np.savetxt(os.path.join(temp_path, 't12.txt'), t12)
        zipf = zipfile.ZipFile(os.path.join(temp_path, 't12.zip'), 'w', zipfile.ZIP_DEFLATED)
        zipf.write(os.path.join(temp_path, 't12.txt'), 't12.zip')
        zipf.close()
        t12_bytes = os.path.getsize(os.path.join(temp_path, 't12.zip'))

        # compute cdm
        cdm = t12_bytes / (t1_bytes + t2_bytes)
        return cdm
