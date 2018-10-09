import numpy as np
import pyhht
from sklearn.base import BaseEstimator, TransformerMixin


class EMD(BaseEstimator, TransformerMixin):
    def __init__(self, n_imfs):
        self.n_imfs = n_imfs
        self.shape = None

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
            decomposer = pyhht.EMD(signal, n_imfs=self.n_imfs)
            imfs = decomposer.decompose()

            imfs_zeros = np.zeros([window_length, self.n_imfs])
            imfs_zeros[:, 0:len(imfs) - 1] = imfs[0:-1].T

            res[i] = imfs_zeros
        return np.reshape(res, [n_signals * self.n_imfs, window_length])

    def fit(self, *_):
        return self
