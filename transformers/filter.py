import numpy as np
from PyQt5.QtWidgets import QWidget, QFormLayout, QSpinBox, QPushButton, QLabel
from scipy.signal import butter, lfilter
from sklearn.base import BaseEstimator, TransformerMixin

from utils.prints import Print


class Filter(BaseEstimator, TransformerMixin):
    name = "Filter"

    def __init__(self, l_freq=7, h_freq=30, picks=None):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.picks = picks
        self.fs = 250

    def transform(self, X, *args):
        if self.l_freq >= self.h_freq:
            Print.warning("Invalid filter: l_freq: {}, h_freq: {}".format(self.l_freq, self.h_freq))
            return X

        X = np.asarray([self.butter_bandpass_filter(x, self.l_freq, self.h_freq, self.fs) for x in X])
        return X

    def fit(self, *_):
        return self

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

    @staticmethod
    def control_panel():
        print("creating panel")
        res = QWidget()
        form_layout = QFormLayout()

        form_layout.addRow("l_freq", QSpinBox())
        form_layout.addRow("h_freq", QSpinBox())

        button = QPushButton()
        form_layout.addRow("Submit", button)

        res.setLayout(form_layout)
        print(res)
        return QLabel("Hei")
