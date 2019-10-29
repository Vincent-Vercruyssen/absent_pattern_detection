""" Bagging SVM classifier.

Reference:
    Mordelet, F., & Vert, J. P. (2014). A bagging SVM to learn from positive and unlabeled examples.
    Pattern Recognition Letters, 37, 201-209.

"""

# Authors: Vincent Vercruyssen, 2018.

import random
import numpy as np

from sklearn.svm import SVC, OneClassSVM


# -------------
# CLASSES
# -------------

class BaggingSVM():
    """ Bagging SVM classifier.

    Parameters
    ----------
    ...

    """

    def __init__(self,
                 K=200,
                 T=100,
                 kernel_type='linear',  # 'rbf', 'poly', 'sigmoid', 'mixed'
                 class_weights=False,
                 feature_names=None,
                 tol=1e-8, verbose=False):

        self.K = int(K)
        self.T = int(T)
        self.kernel_type = str(kernel_type)
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
        n_pos = len(P_ix)
        X_pos = X[P_ix, :]

        # number of features
        n_feat = X.shape[1]
        ns_feat = max(10, int(np.sqrt(n_feat)))

        # loop over iterations
        feature_importances = np.zeros(X.shape[1], dtype=float)
        self.clfs = []
        for i in range(self.T):
            # construct the training set
            unlab_sample_ix = np.random.choice(U_ix, size=self.K, replace=False)
            lab_sample_ix = np.random.choice(np.arange(0, n_pos, 1), size=min(20, n_pos), replace=False)
            X_temp = np.vstack((X_pos[lab_sample_ix, :], X[unlab_sample_ix, :]))
            y_temp = np.ones(len(lab_sample_ix) + self.K)
            y_temp[len(lab_sample_ix):] = -1

            # sample weight fraction
            fraction = round(self.K / len(lab_sample_ix))

            # subsample the features: 15 features each time
            #feat_ix = np.random.choice(np.arange(0, n_feat, 1), size=ns_feat, replace=False)
            feat_ix = np.arange(0, n_feat, 1)
            X_temp_sub = X_temp[:, feat_ix]

            # train the classifier
            if self.kernel_type == 'mixed':
                temp_kernel = random.choice(['linear', 'rbf', 'poly'])
            else:
                temp_kernel = self.kernel_type
            #temp_gamma = random.choice([0.001, 0.01, 0.1, 1.0, 10, 100, 1000])
            if self.class_weights:
                clf = SVC(kernel=temp_kernel, gamma='auto', class_weight={1.0: fraction, -1.0: 1.0}, probability=True)
                #clf = OneClassSVM(kernel=temp_kernel, gamma=temp_gamma)
            else:
                clf = SVC(kernel=temp_kernel, gamma='auto', class_weight='balanced', probability=True)
                #clf = OneClassSVM(kernel=temp_kernel, gamma=temp_gamma)
            clf.fit(X_temp_sub, y_temp)
            self.clfs.append((feat_ix, clf))

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
        for clf_info in self.clfs:
            feat_ix, clf = clf_info[0], clf_info[1]
            X_sub = X[:, feat_ix]
            y_pred = clf.predict(X_sub)
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
