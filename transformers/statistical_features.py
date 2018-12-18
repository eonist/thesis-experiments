import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin


def mean_power(X):
    return (X ** 2).mean()


class StatisticalFeatures(BaseEstimator, TransformerMixin):
    keywords = {
        "__all__": ["mean", "mean_power", "var", "std", "kurt", "skew", "max", "min", "median"],
        "__fast__": ["mean", "var", "std"],
        "__env__": ["min", "max", "var"],
        "__env2__": ["min", "max", "mean_power"],
        "__exp__": ["kurt", "skew"]
    }

    feature_map = {
        "mean": np.mean,
        "mean_power": mean_power,
        "var": np.var,
        "std": np.std,
        "kurt": kurtosis,
        "skew": skew,
        "max": np.max,
        "min": np.min,
        "median": np.median
    }

    def __init__(self, features, splits=1):
        if isinstance(features, str):
            if features in self.keywords:
                self.features = self.keywords[features]
            else:
                self.features = [features]
        else:
            self.features = features

        self.splits = splits

        self.shape = None

    def transform(self, X, *args):
        n_samples, n_signals, window_length = np.shape(X)
        X_t = np.empty([n_samples, len(self.features) * n_signals * self.splits])

        for i, sample in enumerate(X):
            X_t[i] = self._transform_sample(sample)

        self.shape = np.shape(X_t)

        return X_t

    def _transform_sample(self, sample):
        n_signals, window_length = np.shape(sample)
        res = np.empty([n_signals, self.splits, len(self.features)])
        for i, signal in enumerate(sample):
            r = window_length / self.splits
            signal_parts = [signal[int(r * x):int(r * (x + 1))] for x in range(self.splits)]
            for j in range(self.splits):
                for k, feature in enumerate(self.features):
                    func = self.feature_map[feature]
                    res[i, j, k] = func(signal_parts[j])

        return np.reshape(res, [n_signals * len(self.features) * self.splits])

    def fit(self, *_):
        return self
