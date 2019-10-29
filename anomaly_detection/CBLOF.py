""" Cluster-based local outlier factor.

Reference:
    He, Z., Xu, X., & Deng, S. (2003). Discovering cluster-based local outliers.
    Pattern Recognition Letters, 24(9-10), 1641-1650.

"""

# Authors: Vincent Vercruyssen, 2018.

import numpy as np

from collections import Counter
from sklearn.cluster import KMeans
from .BaseDetector import BaseDetector
from .utils.fastfuncs import fast_distance_matrix
from .utils.validation import check_X_y


# -------------
# CLASSES
# -------------

class CBLOF(BaseDetector):
    """ Cluster-based (k-means) anomaly detection.

    Parameters
    ----------
    k : int (default=10)
        Number of clusters.

    alpha : float (default=0.1)
        Boundary parameter for small/large clusters.

    beta : int (default=5)
        Boundary parameter for small/large clusters.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    elbow : bool (default=False)
        Use Elbow method to tune the number of clusters for k-means.

    ub_clusters : int (default=50)
        Upperbound on the number of clusters when tuning with the Elbow method.

    explained_var : float (default=0.8)
        Explained variance for the Elbow method.

    Comments
    --------
    - Implementation with support for out-of-sample setting.
    - The number of clusters cannot be larger than the number of instances in
    the data: automatically correct if necessary.
    """

    def __init__(self, k=10, alpha=0.1, beta=5, contamination=0.1, elbow=False, ub_clusters=50, explained_var=0.8,
                 tol=1e-8, verbose=False):
        super(CBLOF, self).__init__()

        self.k = int(k)
        self.alpha = min(1.0, max(0.0, float(alpha)))  # TODO: better error catching
        self.beta = int(beta)
        self.contamination = float(contamination)
        self.elbow = bool(elbow)
        self.ub_clusters = int(ub_clusters)
        self.explained_var = float(explained_var)

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

        # find the clusters
        cluster_labels, self.centroids = self._cluster_data(X)
        self.cluster_sizes = np.zeros(len(self.centroids))
        for k, v in Counter(cluster_labels).items():
            self.cluster_sizes[k] = v

        # find the large and small clusters
        self.large_idx, self.small_idx = self._find_small_large_clusters(n)

        return self

    def predict(self, X):
        """ Compute the anomaly score + predict the label of instances in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        X, y = check_X_y(X, None)
        n, _ = X.shape

        # compute the CBLOF score
        cblof_score = self._compute_cblof_score(X)
        y_score = (cblof_score - min(cblof_score)) / (max(cblof_score) - min(cblof_score))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(n * (1.0 - self.contamination))]
        y_pred = np.ones(n, dtype=float)
        y_pred[y_score < self.threshold] = -1

        return y_score, y_pred

    def _compute_cblof_score(self, X):
        """ Compute the CBLOF score for the points in X.

        :returns cblof_score : np.array(), shape (n_samples)
            CBLOF score for each point in X.
        """

        n, _ = X.shape

        # distance to each cluster centroid
        distance_matrix = fast_distance_matrix(X, self.centroids)
        cluster_labels = np.argmin(distance_matrix, axis=1)
        assert len(cluster_labels) == X.shape[0], 'Error - assigning instances to clusters!'

        # compute the CBLOF score
        cblof_score = np.zeros(n)
        for i, l in enumerate(cluster_labels):
            if l in self.large_idx:
                # large cluster
                cblof_score[i] = self.cluster_sizes[l] * distance_matrix[i, l]
            else:
                # small cluster
                min_distance = min(distance_matrix[i, self.large_idx])
                cblof_score[i] = self.cluster_sizes[l] * min_distance

        return cblof_score

    def _find_small_large_clusters(self, n):
        """ Find the small and large clusters.

        :param n : int
            Number of instances in the training data.

        :returns large_idx : np.array(), shape (n_clusters)
            Indices of the large clusters in self.centroids.
        :returns small_idx : np.array(), shape (n_clusters)
            Indices of the small clusters in self.centroids.
        """

        sc = np.argsort(self.cluster_sizes)[::-1]
        ss = np.sort(self.cluster_sizes)[::-1]

        # small and large clusters
        c_sum = 0
        b = 0
        stop = False
        only_large = False
        while not stop:
            if b < len(ss) - 1:
                current_size = ss[b]
                next_size = ss[b + 1]
                c_sum += current_size
                b += 1
                # condition one
                if c_sum >= int(n * (1.0 - self.alpha)):
                    stop = True
                # condition two
                if current_size / next_size >= self.beta:
                    stop = True
            else:
                stop = True
                only_large = True
        if only_large == True:
            large_idx = sc
            small_idx = np.array([])
        else:
            large_idx = sc[:b]
            small_idx = sc[b:]

        return large_idx, small_idx

    def _cluster_data(self, data):
        """ Cluster the given data with k-means.

        :param data: np.array()
            2D array containing the data.

        :returns cl_labels: np.array()
            Cluster labels for the individual datapoints.
        :returns centroids: np.array(), shape (n_centroids, n_features)
            2D array of the cluster centroids.
        """

        # determine the optimal number of clusters if needed
        if self.elbow:
            mean_dist = []
            for k in range(self.ub_clusters):
                kmeans = KMeans(n_clusters=k + 1)
                kmeans.fit(data)
                mean_dist.append(kmeans.inertia_ / data.shape[0])
            # Elbow: explain x % of the variance
            variance = np.diff(mean_dist) * -1
            distortion_percent = np.cumsum(variance) / (mean_dist[0] - mean_dist[-1])
            idx = np.where(distortion_percent > self.explained_var)[0]
            self.k = idx[0] + 1

        # cluster the data with KMeans
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(data)

        return kmeans.labels_, kmeans.cluster_centers_
