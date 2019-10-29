""" Local outlier factor.

Reference:
    M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander. LOF: identifying density-based local outliers.
    In Proceedings of the 2000 ACM SIGMOD international conference on Management of data, vol. 29, no. 2. ACM, 2000, pp. 93–104.

"""

# Authors: Vincent Vercruyssen, 2018.

import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from .BaseDetector import BaseDetector
from .utils.validation import check_X_y


# -------------
# CLASSES
# -------------

class LOF(BaseDetector):
    """ Local outlier factor (LOF).

    Parameters
    ----------
    k : int (default=10)
        Number of nearest neighbors.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    metric : string (default=euclidean)
        Distance metric for the distance computation.

    Comments
    --------
    - This method DOES NOT EASILY extend to OUT-OF-SAMPLE setting!
    - The number of neighbors cannot be larger than the number of instances in
    the data: automatically correct if necessary.
    """

    def __init__(self, k=10, contamination=0.1, metric='euclidean',
                 tol=1e-8, verbose=False):
        super(LOF, self).__init__()

        self.k = int(k)
        self.contamination = float(contamination)
        self.metric = str(metric)

        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict(self, X, y=None):
        """ Fit the model to the training set X and returns the anomaly score
            of the instances in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        X, y = check_X_y(X, y)

        return self.fit(X, y).predict(X)

    def fit(self, X, y=None):
        """ Fit the model using data in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns self : object
        """

        X, y = check_X_y(X, y)
        n, _ = X.shape

        nn = self._check_valid_number_of_neighbors(n)
        self.clf = LocalOutlierFactor(n_neighbors=nn, contamination=self.contamination, metric=self.metric)
        self.clf.fit(X)

        return self

    def predict(self, X):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        X, y = check_X_y(X, None)
        n, _ = X.shape

        # predict the anomaly scores
        lof_score = self.clf._decision_function(X) * -1  # Shifted opposite of the Local Outlier Factor of X

        # scaled y_score
        y_score = (lof_score - min(lof_score)) / (max(lof_score) - min(lof_score))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(n * (1.0 - self.contamination))]
        y_pred = np.ones(n, dtype=float)
        y_pred[y_score < self.threshold] = -1

        return y_score, y_pred

    def _check_valid_number_of_neighbors(self, n_samples):
        """ Check if the number of nearest neighbors is valid and correct.

        :param n_samples : int
            Number of samples in the data.
        """

        return min(n_samples, self.k)
