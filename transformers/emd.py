import numpy as np
import pyhht
import requests
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from config import URL
from models.session import Session
from transformers.statistical_features import StatisticalFeatures
from utils.prints import print_time


class EMD(BaseEstimator, TransformerMixin):
    def __init__(self, n_imfs):
        self.n_imfs = n_imfs

    def transform(self, X, *args):
        n_samples, n_signals, window_length = np.shape(X)
        X_t = np.empty([n_samples, n_signals * self.n_imfs, window_length])

        for i, sample in enumerate(X):
            X_t[i] = self._transform_sample(sample)

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


if __name__ == '__main__':
    def create_dataset():
        r = requests.get(URL.sessions)

        json_data = r.json()

        dataset = Session(**json_data[0]).dataset()

        for i in range(len(json_data)):
            if i > 0:
                session = Session(**json_data[i])
                dataset = dataset + session.dataset()

        return dataset


    ds = create_dataset()
    pipeline = make_pipeline(
        EMD(1),
        StatisticalFeatures(features="__all__"),
        svm.SVC(kernel='linear')
    )


    @print_time
    def cross_val(clf, X, Y):
        score = cross_val_score(clf, X, Y, cv=5, scoring="accuracy")
        print(score)


    cross_val(pipeline, ds.X, ds.Y)
