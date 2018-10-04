import numpy as np
import scipy.linalg as la
from mne.decoding import CSP as MneCSP
from numpy.linalg import multi_dot as dot

from utils.exceptions import InvalidKernel
from utils.prints import Print


class CSP(MneCSP):
    name = "csp"

    def __init__(self, kernel="mne", n_components=4, **kwargs):
        super().__init__(n_components=n_components, transform_into="csp_space", **kwargs)
        self.kernel = kernel
        self.filters_ = None

        Print.data(self.n_components)

    def transform(self, X, *args):
        if self.kernel == "mne":
            X = super(CSP, self).transform(X)
        elif self.kernel == "custom":
            X = self.custom_transform(X, *args)
        else:
            raise InvalidKernel(self.kernel)

        print("")
        Print.point("CSP")
        Print.data(self.kernel)
        Print.data(self.n_components)
        Print.data(np.shape(X))

        return X

    def fit(self, X, y):
        if self.kernel == "mne":
            super(CSP, self).fit(X, y)
        elif self.kernel == "custom":
            self.custom_fit(X, y)
        else:
            raise InvalidKernel(self.kernel)

        Print.data(np.shape(self.filters_))

        return self

    # <--- CUSTOM CSP METHODS --->

    def custom_fit(self, X, y):
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

        self.filters_ = filters[0]

    def custom_transform(self, X, *args):
        filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(filters, sample) for sample in X])
        return X
        # new_shape = np.shape(X)
        # return np.reshape(X, [new_shape[0], new_shape[1] * new_shape[2], new_shape[3]])

    @staticmethod
    def cov_matrix(m):
        """ Calculate the covariance matrix """
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
