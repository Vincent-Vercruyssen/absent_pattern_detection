""" One-class SVM anomaly detection.

Reference:
    Chen, Y., Zhou, X. S., & Huang, T. S. (2001). One-class SVM for learning in image retrieval.
    In Image Processing, 2001. Proceedings. 2001 International Conference on (Vol. 1, pp. 34-37). IEEE.

"""

# Authors: Vincent Vercruyssen, 2018.

import numpy as np

from sklearn.svm import OneClassSVM
from .BaseDetector import BaseDetector
from .utils.validation import check_X_y


# -------------
# CLASSES
# -------------

class OSVM(BaseDetector):
    """ One-class SVM for anomaly detection.

    Parameters
    ----------
    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    ... (arguments as in Sklearn)

    Comments
    --------
    - Provides both score and absolute predictions in the predict method.
    - This is basically a wrapper around the sklearn OneClassSVM.
    """

    def __init__(self, contamination=0.1, kernel='rbf', degree=3, gamma='auto', coef0=0.0, nu=0.5, shrinking=False,
                 tol=1e-8):
        super(OSVM, self).__init__()

        self.contamination = float(contamination)
        self.kernel = str(kernel)
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.nu = float(nu)
        self.shrinking = bool(shrinking)
        self.tol = float(tol)

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

        self.clf = OneClassSVM(kernel=self.kernel,
                               degree=self.degree,
                               gamma=self.gamma,
                               coef0=self.coef0,
                               tol=self.tol,
                               nu=self.nu,
                               shrinking=self.shrinking)
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

        # outlier score (distance to the decision boundary)
        osvm_score = self.clf.decision_function(X) * -1
        osvm_score = osvm_score.flatten()
        y_score = (osvm_score - min(osvm_score)) / (max(osvm_score) - min(osvm_score))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(n * (1.0 - self.contamination))]
        y_pred = np.ones(n, dtype=float)
        y_pred[y_score < self.threshold] = -1

        return y_score, y_pred
