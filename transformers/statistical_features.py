import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin


class StatisticalFeatures(BaseEstimator, TransformerMixin):
    keywords = {
        "__all__": ["mean", "var", "std", "kurt", "skew", "max", "min", "median"],
        "__fast__": ["mean", "var", "std"]
    }

    feature_map = {
        "mean": np.mean,
        "var": np.var,
        "std": np.std,
        "kurt": kurtosis,
        "skew": skew,
        "max": np.max,
        "min": np.min,
        "median": np.median
    }

    def __init__(self, features):
        if type(features) == str and features in self.keywords:
            self.features = self.keywords[features]
        else:
            self.features = features

    def transform(self, X, *args):
        n_samples, n_signals, window_length = np.shape(X)
        X_t = np.empty([n_samples, len(self.features) * n_signals])

        for i, sample in enumerate(X):
            X_t[i] = self._transform_sample(sample)

        return X_t

    def _transform_sample(self, sample):
        n_signals, window_length = np.shape(sample)
        res = np.empty([n_signals, len(self.features)])
        for i, signal in enumerate(sample):
            for j, feature in enumerate(self.features):
                func = self.feature_map[feature]
                res[i, j] = func(signal)

        return np.reshape(res, [n_signals * len(self.features)])

    def fit(self, *_):
        return self
