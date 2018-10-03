import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MeanPower(BaseEstimator, TransformerMixin):
    def __init__(self, log=False):
        self.log = log

        self.mean = None
        self.std = None

    def transform(self, X, *args):
        X = (X ** 2).mean(axis=2)

        if self.log:
            X = np.log(X)
        else:
            X -= self.mean
            X /= self.std

        return X

    def fit(self, X, *args):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        return self
