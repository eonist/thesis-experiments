import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin

from models.session import Session
from utils.prints import Print


class DWT(BaseEstimator, TransformerMixin):
    def __init__(self, dim=1, wavelet='db1'):
        self.dim = dim
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

        if self.dim == 1:

            for i, signal in enumerate(sample):
                cA, cD = pywt.dwt(signal, self.wavelet)
                res[i * 2, :] = cA
                res[i * 2 + 1, :] = cD

        elif self.dim == 2:
            coeffs = pywt.dwt2(sample, self.wavelet)
            cA, (cH, cV, cD) = coeffs
            res = np.vstack((cA, cH, cV, cD))

        else:
            raise Exception("Invalid dim: {}".format(self.dim))

        return res

    def fit(self, X, *args):
        sample = X[0]
        signal = sample[0]

        if self.dim == 1:
            cA, cD = pywt.dwt(signal, self.wavelet)

            cAshape = np.shape(cA)
            cDshape = np.shape(cD)

            self.sample_shape = (2 * len(sample), cAshape[0])

        if self.dim == 2:
            coeffs = pywt.dwt2(sample, self.wavelet)
            cA, (cH, cV, cD) = coeffs

            res = np.vstack((cA, cH, cV, cD))

            self.sample_shape = np.shape(res)

        return self


if __name__ == '__main__':

    ds = list(Session.full_dataset_gen(window_length=10))[0]

    indices = []

    for class_idx in [0, 1]:
        for i, y in enumerate(ds.y):
            if y == class_idx:
                indices.append(i)
                break

    print(indices)

    wavelet = DWT(dim=1, wavelet='db1')
    wavelet.fit(ds.X)

    for i in indices:
        sample = ds.X[i]
        sample_t = wavelet._transform_sample(sample)

        Print.pandas(sample)
        Print.pandas(sample_t)
