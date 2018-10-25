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
    def __init__(self, n_imfs=0, max_imfs=0, imf_picks=None, max_iter=2000, subtract_residue=False):

        if isinstance(imf_picks, list):
            self.imf_picks = imf_picks
            self.n_imfs = len(imf_picks)
        elif isinstance(imf_picks, str) and ',' not in imf_picks:
            self.imf_picks = imf_picks
            self.n_imfs = n_imfs
        elif isinstance(imf_picks, str):
            self.imf_picks = [int(t) for t in imf_picks.split(",")]
            self.n_imfs = len(self.imf_picks)
        else:
            self.imf_picks = None
            self.n_imfs = n_imfs

        self.max_imfs = max_imfs

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

        if len(imfs) == 2:
            imfs = imfs[:-1]
        elif self.imf_picks == "n_imfs":
            imfs = imfs[:-1]
            imfs = imfs[:self.n_imfs]
        elif isinstance(self.imf_picks, str):
            distances = []
            distance_func = distance_functions[self.imf_picks]
            for imf in imfs[:-1]:
                s = signal - imfs[-1] * int(self.subtract_residue)
                distances.append(distance_func(s, imf))

            imfs = imfs[[np.argmin(distances)]]
        else:
            imf_picks = [idx for idx in self.imf_picks if idx < len(imfs)]
            imfs = imfs[imf_picks]

        imfs_zeros = np.zeros([window_length, self.n_imfs])
        imfs_zeros[:, :len(imfs)] = imfs.T

        return imfs_zeros

    def fit(self, *_):
        return self
