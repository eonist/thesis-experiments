import warnings

import numpy as np
import scipy.linalg as la
from mne.decoding import CSP as MneCSP
from numpy.linalg import multi_dot as dot

from utils.exceptions import InvalidKernel

warnings.filterwarnings('ignore')


class CSP(MneCSP):
    name = "csp"

    def __init__(self, kernel="mne", n_components=4, mode="1vall", **kwargs):
        super().__init__(n_components=n_components, transform_into="csp_space", **kwargs)
        self.kernel = kernel
        self.filters_ = None
        self.shape = None
        self.n_classes = None
        self.mode = mode

    def transform(self, X, *args):
        if self.kernel == "mne":
            X = super(CSP, self).transform(X)
        elif self.kernel == "custom":
            X = self.custom_transform(X, *args)
        else:
            raise InvalidKernel(self.kernel)

        self.shape = np.shape(X)
        return X

    def fit(self, X, y):
        if self.kernel == "mne":
            super(CSP, self).fit(X, y)
        elif self.kernel == "custom":
            self.custom_fit(X, y)
        else:
            raise InvalidKernel(self.kernel)

        return self

    # <--- CUSTOM CSP METHODS --->

    def custom_fit(self, X, y):
        y = y.astype(int)
        classes = np.unique(y)
        self.n_classes = len(classes)

        if self.n_classes < 2:
            print("Must have at least 2 tasks for filtering.")
            return self

        n_samples, n_signals, window_length = np.shape(X)

        if self.mode == "1vall":
            filters = np.zeros([self.n_classes, n_signals, n_signals])

            for i, class_id in enumerate(classes):
                x = X[y == class_id]
                not_x = X[y != class_id]

                Rx = np.average(np.array([np.cov(sample) for sample in x]), axis=0)
                not_Rx = np.average(np.array([np.cov(sample) for sample in not_x]), axis=0)

                SFx = self.spatial_filter(Rx, not_Rx)
                filters[i] = SFx

                # Special case: only two tasks, no need to compute any more mean variances
                if self.n_classes == 2:
                    filters[1] = self.spatial_filter(not_Rx, Rx)
                    break
        elif self.mode == "1v1":
            n_filters = (self.n_classes ** 2 - self.n_classes) // 2
            filters = np.zeros([n_filters, n_signals, n_signals])

            idx = 0

            for i in range(len(classes)):
                for j in range(i):
                    xi = X[y == classes[i]]
                    xj = X[y == classes[j]]

                    Rxi = np.average(np.array([np.cov(sample) for sample in xi]), axis=0)
                    Rxj = np.average(np.array([np.cov(sample) for sample in xj]), axis=0)

                    SFx = self.spatial_filter(Rxi, Rxj)
                    filters[idx] = SFx
                    idx += 1
        else:
            raise Exception("Invalid CSP mode: {}".format(self.mode))

        self.filters_ = filters

    def custom_transform(self, X, *args):
        if self.n_classes == 2:
            filters = self.filters_[0, :self.n_components]
        else:
            filters = self.filters_[:, :self.n_components]

        n_samples, n_signals, window_length = np.shape(X)
        res = np.zeros([n_samples, self.n_components, len(filters), window_length])
        for i, sample in enumerate(X):
            for j, filter in enumerate(filters):
                res[i, :, j, :] = np.dot(filter, sample)

        return np.reshape(res, [n_samples, self.n_components * len(filters), window_length])

    def spatial_filter(self, R1, R2):
        R = (R1 + R2)

        E, U = la.eig(R)

        order = np.flip(np.argsort(E), axis=0)
        E = E[order]
        U = U[:, order]

        D = np.diag(E)

        # Whitening transformation matrix
        P = dot([np.sqrt(la.inv(D)), U.T])

        S1 = dot([P, R1, P.T])
        S2 = dot([P, R2, P.T])

        E1, U1 = la.eig(S1)
        E2, U2 = la.eig(S2)

        order = np.flip(np.argsort(E1), axis=0)
        E1 = E1[order]
        U1 = U1[:, order]

        F = dot([U1.T, P]).astype(np.float32)

        return F
