import numpy as np
import pyhht
from scipy.spatial.distance import minkowski, euclidean, correlation, cityblock
from sklearn.base import BaseEstimator, TransformerMixin

distance_functions = {
    "minkowski": minkowski,
    "euclidean": euclidean,
    "correlation": correlation,
    "cityblock": cityblock
}


class EMD(BaseEstimator, TransformerMixin):
    def __init__(self, mode, n_imfs=1, max_iter=2000, subtract_residue=False):
        self.mode = mode

        if self.mode == "set_max":
            self.max_imfs = n_imfs
            self.n_imfs = n_imfs
        elif self.mode == "minkowski":
            self.max_imfs = 0
            self.n_imfs = n_imfs
        else:
            raise Exception("Invalid EMD mode: {}".format(self.mode))

        self.shape = None
        self.max_iter = max_iter
        self.subtract_residue = subtract_residue

    def transform(self, X, *args):
        n_samples, n_signals, window_length = np.shape(X)
        X_t = np.empty([n_samples, n_signals * self.n_imfs, window_length])
        for i, sample in enumerate(X):
            X_t[i] = self._transform_sample(sample)

        self.shape = np.shape(X_t)
        return X_t

    def _transform_sample(self, sample):
        n_signals, window_length = np.shape(sample)
        res = np.empty([n_signals, window_length, self.n_imfs])

        for i, signal in enumerate(sample):
            res[i] = self._transform_signal(signal, window_length=window_length)

        return np.reshape(res, [n_signals * self.n_imfs, window_length])

    def _transform_signal(self, signal, window_length):
        decomposer = pyhht.EMD(signal, n_imfs=self.max_imfs, maxiter=self.max_iter)
        imfs = decomposer.decompose()

        if self.mode == "set_max":
            imfs = imfs[:-1]
        elif self.mode == "minkowski":
            distances = []
            s = signal - imfs[-1] * int(self.subtract_residue)

            for imf in imfs[:-1]:
                distances.append(minkowski(s, imf))

            idx = np.argpartition(distances, np.min([len(distances) - 1, self.n_imfs]))

            imfs = imfs[idx[:self.n_imfs]]

        imfs_zeros = np.zeros([window_length, self.n_imfs])
        imfs_zeros[:, :len(imfs)] = imfs.T

        return imfs_zeros

    def fit(self, *_):
        return self
