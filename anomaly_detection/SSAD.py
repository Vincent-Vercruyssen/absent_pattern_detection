"""
SSAD algorithm from:
    Görnitz, N., Kloft, M., Rieck, K., & Brefeld, U. (2013).
    Toward supervised anomaly detection.
    Journal of Artificial Intelligence Research, 46, 235-262.

Not tree-based though
"""

# + OVERHEAD TO RUN EXPRIMENTS
import sys, os, copy
import cvxopt as co
import random
import pickle
import argparse
import pandas as pd
from glob import glob
import numpy as np

from .ConvexSSAD import ConvexSSAD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid


# -------------
# FUNCTIONS
# -------------

def get_kernel(X, Y, type='linear', param=1.0):
    """Calculates a kernel given the data X and Y (dims x exms)"""
    _, Xn = X.shape
    _, Yn = Y.shape

    kernel = 1.0
    if type == 'linear':
        #print('Calculating linear kernel with size {0}x{1}.'.format(Xn, Yn))
        kernel = X.T.dot(Y)

    if type == 'rbf':
        #print('Calculating Gaussian kernel with size {0}x{1} and sigma2={2}.'.format(Xn, Yn, param))
        Dx = (np.ones((Yn, 1)) * np.diag(X.T.dot(X)).reshape(1, Xn)).T
        Dy = (np.ones((Xn, 1)) * np.diag(Y.T.dot(Y)).reshape(1, Yn))
        kernel = Dx - 2.* np.array(X.T.dot(Y)) + Dy
        kernel = np.exp(-kernel / param)

    return kernel


# -------------
# CLASSES
# -------------

class SSAD:
    """ SSAD baseline. """

    def __init__(self, contamination=0.1,
                 kappa=1.0, Cu=1.0, Cl=1.0,
                 kernel_type='rbf', kernel_param=1.0):

        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.c = contamination
        self.tol = 1e-8

        self.kappa = float(kappa)       # set to 1.0 in the paper
        self.Cu = float(Cu)             # regularization parameter [0.01, 0.1, 1, 10, 100]
        self.Cp = float(Cl)             # regularization parameter [0.01, 0.1, 1, 10, 100]
        self.Cn = float(Cl)             # regularization parameter [0.01, 0.1, 1, 10, 100]
        self.Cl = float(Cl)

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

        self.fit(X, y)
        return self.predict(X)

    def fit(self, X, y=None):
        """ Fit the model using data in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns self : object
        """

        n, _ = X.shape

        Xt = X.T
        if y is None:
            y = np.zeros(n, dtype=int)

        # build the training kernel
        kernel = get_kernel(Xt, Xt, type=self.kernel_type, param=self.kernel_param)

        # train SSAD
        """ REVERSE """
        y = y * -1
        """ END REVERSE """

        self.ssad = ConvexSSAD(kernel, y, self.kappa, self.Cp, self.Cu, self.Cn)
        self.ssad.fit()

        self.Xt = Xt

    def predict(self, X):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        n, _ = X.shape

        # build the test kernel
        kernel = get_kernel(X.T, self.Xt[:, self.ssad.svs], type=self.kernel_type, param=self.kernel_param)

        # predict
        # TODO: something's wrong here!
        y_score = np.array(self.ssad.apply(kernel)) * -1
        y_score = np.array(y_score).flatten()

        # compute y_pred
        offset = np.sort(y_score)[int(n * (1.0 - self.c))]
        y_pred = np.ones(n, dtype=int)
        y_pred[y_score < offset] = -1

        # simple score scaling
        y_score = (y_score - min(y_score)) / (max(y_score) - min(y_score))

        return y_score, y_pred
