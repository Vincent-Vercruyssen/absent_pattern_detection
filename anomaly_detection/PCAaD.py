""" PCA based anomaly detection. """

# Authors: Vincent Vercruyssen, 2018.

import numpy as np

from sklearn.decomposition import PCA
from .BaseDetector import BaseDetector
from .utils.validation import check_X_y


# -------------
# CLASSES
# -------------

class PCAaD(BaseDetector):
    """ PCA-based anomaly detection

    Parameters
    ----------
    n_components : int (default=10)
        Number of components for PCA.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    Comments:
    ---------
    - Uses the Euclidean distance to compute the reconstruction error.
    - The number of components cannot be larger than the dimension of the input
    data used for fitting: correct automatically if necessary.
    """

    def __init__(self, n_components=10, contamination=0.1,
                 tol=1e-8, verbose=False):
        super(PCAaD, self).__init__()

        self.n_components = int(n_components)
        self.contamination = float(contamination)

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

        nc = self._check_valid_number_of_components(X.shape[1])
        self.pca = PCA(n_components=nc)
        self.pca.fit(X)

        return self

    def predict(self, X):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        X, y = check_X_y(X, None)
        ni, _ = X.shape

        # encode
        X_encode = self.pca.transform(X)
        assert X_encode.shape[1] == self.n_components, 'Something went wrong in the encoding!'

        # decode
        X_decode = self.pca.inverse_transform(X_encode)

        # reconstruction error
        error = np.sqrt(np.sum((X - X_decode) ** 2, axis=1))
        y_score = (error - min(error)) / (max(error) - min(error))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(ni * (1.0 - self.contamination))]
        y_pred = np.ones(ni, dtype=float)
        y_pred[y_score < self.threshold] = -1

        return y_score, y_pred

    def _check_valid_number_of_components(self, nf):
        """ Check if the number of components is valid and correct.

        :param nf : int
            Number of features.
        """

        return min(nf, self.n_components)
