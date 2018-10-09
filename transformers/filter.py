import mne
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.base import BaseEstimator, TransformerMixin


class Filter(BaseEstimator, TransformerMixin):
    name = "filter"

    def __init__(self, kernel="mne", l_freq=7, h_freq=30, picks=None):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.picks = picks
        self.fs = 250
        self.kernel = kernel
        self.shape = None

    def fit(self, *_):
        return self

    def transform(self, X, *args):
        if self.kernel == "mne":
            return self.mne_transform(X, *args)
        elif self.kernel == "custom":
            return self.custom_transform(X, *args)
        else:
            raise Exception("Filter kernel \"{}\" not recognized".format(self.kernel))

    # <--- MNE FILTER METHODS --->

    def mne_transform(self, X, *args):
        info = mne.create_info(
            ch_names=['C3', 'C4', 'P3', 'P4'],
            ch_types=['eeg', 'eeg', 'eeg', 'eeg'],
            sfreq=self.fs
        )

        raws = [mne.io.RawArray(x, info) for x in X]
        raws = [raw.filter(self.l_freq, self.h_freq, fir_design='firwin', skip_by_annotation='edge') for raw in raws]

        X = np.array([data for data, times in [raw[:] for raw in raws]])

        self.shape = np.shape(X)

        return X

    # <--- CUSTOM FILTER METHODS --->

    def custom_transform(self, X, *args):
        X = np.asarray([self.butter_bandpass_filter(x, self.l_freq, self.h_freq, self.fs) for x in X])
        self.shape = np.shape(X)
        return X

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype='band', output='ba')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data, axis=0)
        return y
