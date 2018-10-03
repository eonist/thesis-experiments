import numpy as np
from mne.decoding import TransformerMixin
from mne.fixes import BaseEstimator


class AvgPower(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, *args):
        return (X ** 2).mean(axis=2)

    def fit(self, *_):
        return self
