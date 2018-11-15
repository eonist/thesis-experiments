import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin

from utils.prints import Print


class Wavelet(BaseEstimator, TransformerMixin):
    def __init__(self, n_dimensions=1, wavelet='db1'):
        self.n_dimensions = n_dimensions
        print(wavelet)
        self.wavelet = pywt.Wavelet(wavelet)
        self.sample_shape = None

    def transform(self, X, *args):
        n_samples, n_signals, window_length = np.shape(X)

        X_t = np.zeros([n_samples, self.sample_shape[0], self.sample_shape[1]])

        for i, sample in enumerate(X):
            X_t[i, :, :] = self._transform_sample(sample)

        self.shape = np.shape(X_t)

        return X_t

    def _transform_sample(self, sample):
        res = np.zeros([self.sample_shape[0], self.sample_shape[1]])

        if self.n_dimensions == 1:

            for i, signal in enumerate(sample):
                cA, cD = pywt.dwt(signal, self.wavelet)
                res[i * 2, :] = cA
                res[i * 2 + 1, :] = cD

        elif self.n_dimensions == 2:
            coeffs = pywt.dwt2(sample, self.wavelet)
            cA, (cH, cV, cD) = coeffs

            res = np.vstack((cA, cH, cV, cD))
        else:
            raise Exception("Invalid n_dimensions: {}".format(self.n_dimensions))

        return res

    def fit(self, X, *args):
        sample = X[0]
        signal = sample[0]

        if self.n_dimensions == 1:
            cA, cD = pywt.dwt(signal, self.wavelet)

            cAshape = np.shape(cA)
            cDshape = np.shape(cD)

            Print.data(cAshape)
            Print.data(cDshape)

            self.sample_shape = (2 * len(sample), cAshape[0])

        if self.n_dimensions == 2:
            coeffs = pywt.dwt2(sample, self.wavelet)
            cA, (cH, cV, cD) = coeffs

            res = np.vstack((cA, cH, cV, cD))

            self.sample_shape = np.shape(res)

        return self