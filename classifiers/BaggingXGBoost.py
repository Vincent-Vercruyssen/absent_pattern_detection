""" Bagging XGBoost classifier.

Inspired by the following:
    Mordelet, F., & Vert, J. P. (2014). A bagging SVM to learn from positive and unlabeled examples.
    Pattern Recognition Letters, 37, 201-209.

"""

# Authors: Vincent Vercruyssen, 2018.

import numpy as np

import xgboost as xgb


# -------------
# CLASSES
# -------------

class BaggingXGBoost():
    """ Bagging XGBoost classifier.

    Parameters
    ----------
    ...

    """

    def __init__(self,
                 K=200,
                 T=100,
                 class_weights=False,
                 feature_names=None,
                 tol=1e-8, verbose=False):

        self.K = int(K)
        self.T = int(T)
        self.class_weights = bool(class_weights)
        self.feature_names = feature_names

        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict(self, X, y=None):
        """ Fit the model to the training set X and return the prediction.

        :param X : np.array(), shape (n_samples, n_features)
            The examples to fit the classifier and predict the class
        :param y : np.array(), shape (n_samples), default = None
            Class labels for the examples in X.

        :returns y_prob : np.array(), shape (n_samples)
            Class probability for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Class predictions for the examples in X.
        """

        return self.fit(X, y).predict(X)

    def fit(self, X, y=None):
        """ Fit the model to the training set X and return the prediction.

        :param X : np.array(), shape (n_samples, n_features)
            The examples to fit the classifier and predict the class
        :param y : np.array(), shape (n_samples), default = None
            Class labels for the examples in X.

        :returns self : object
        """

        # unlabeled class check
        U_ix = np.where(y == 0.0)[0]
        if len(U_ix) < self.K:
            """ FIX: better handling here """
            sys.exit()

        # indices of the positive class
        P_ix = np.where(y == 1.0)[0]
        X_pos = X[P_ix, :]
        y_temp = np.ones(len(P_ix) + self.K)
        y_temp[len(P_ix):] = -1.0

        # class weight fraction
        fraction = round(self.K / len(P_ix))

        # loop over iterations
        feature_importances = np.zeros(X.shape[1], dtype=float)
        self.clfs = []
        for i in range(self.T):
            # construct the training set
            sample_ix = np.random.choice(U_ix, size=self.K, replace=False)
            X_temp = np.vstack((X_pos, X[sample_ix, :]))
            # train the classifier
            if self.class_weights:
                clf = xgb.XGBClassifier()
            else:
                clf = xgb.XGBClassifier()
            clf.fit(X_temp, y_temp)
            self.clfs.append(clf)
            # add the feature importances
            #feature_importances += clf.feature_importances_

        # collect the feature importances
        #feature_importances /= self.T
        #importances = {name: feature_importances[i] for i, name in enumerate(self.feature_names)}
        #print('\n\t', importances)

        return self

    def predict(self, X):
        """ Predict labels: 1 for positive, -1 for negative.

        :returns y_prob : np.array(), shape (n_samples)
            Class probability for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Class predictions for the examples in X.
        """

        labels = np.zeros(len(X))

        # bagging
        for clf in self.clfs:
            y_pred = clf.predict(X)
            labels = labels + y_pred
        labels = labels / self.T

        # linearly rescale to [0, 1]
        labels = (labels - min(labels)) / (max(labels) - min(labels))

        # absolute labels
        y_prob = labels.copy()
        y_pred = labels.copy()
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0

        return y_prob, y_pred
