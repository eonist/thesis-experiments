import numpy as np
import scipy.linalg as la
from numpy.linalg import multi_dot as dot
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    name = "CSP"

    def __init__(self, avg_power=False):
        self.filters = None
        self.avg_power = avg_power

    def transform(self, X, *args):
        X = np.asarray([np.dot(self.filters, sample) for sample in X])

        new_shape = np.shape(X)

        X = np.reshape(X, [new_shape[0], new_shape[1] * new_shape[2], new_shape[3]])
        if self.avg_power:
            return (X ** 2).mean(axis=2)
        else:
            return X

    def fit(self, X, y):
        if len(y) < 2:
            print("Must have at least 2 tasks for filtering.")
            return self

        n_samples, n_signals, window_length = np.shape(X)

        classes = np.unique(y)
        n_classes = len(classes)

        filters = np.zeros([n_classes, n_signals, n_signals])

        for i, class_id in enumerate(classes):

            try:
                y = [int(n) for n in y]
                class_id = int(class_id)
                # x = X[y == class_id]

                x = [X[i] for i in range(len(y)) if y[i] == class_id]
            except Exception as e:
                print(class_id)
                print(y)
                raise e

            Rx = np.average(np.array([self.cov_matrix(sample) for sample in x]), axis=0)

            not_x = [X[i] for i in range(len(y)) if y[i] != class_id]
            not_Rx = np.average(np.array([self.cov_matrix(sample) for sample in not_x]), axis=0)

            SFx = self.spatial_filter(Rx, not_Rx)
            filters[i] = SFx

            # Special case: only two tasks, no need to compute any more mean variances
            if n_classes == 2:
                filters[1] = self.spatial_filter(not_Rx, Rx)
                break

        self.filters = filters
        return self

    # <--- HELPER METHODS --->

    @staticmethod
    def cov_matrix(m):
        """
        Calculate the covariance matrix
        https://en.wikipedia.org/wiki/Covariance_matrix
        """

        return np.cov(m)

    @staticmethod
    def spatial_filter(R1, R2):
        R = 0.5 * (R1 + R2)

        E, U = la.eig(R)

        order = np.flip(np.argsort(E), axis=0)
        E = E[order]
        U = U[:, order]

        D = np.diag(E)

        # Whitening transformation matrix
        P = dot([np.sqrt(la.inv(D)), U.T])

        S1 = dot([P, R1, P.T])
        S2 = dot([P, R2, P.T])

        E1, U1 = la.eig(S1, S2)
        order = np.flip(np.argsort(E1), axis=0)
        E1 = E1[order]
        U1 = U1[:, order]

        F = dot([U1.T, P]).astype(np.float32)

        return F
