import mne
import numpy as np
from PyQt5.QtWidgets import QWidget, QFormLayout, QSpinBox, QPushButton, QLabel
from sklearn.base import BaseEstimator, TransformerMixin


class MneFilter(BaseEstimator, TransformerMixin):
    name = "MneFilter"

    def __init__(self, l_freq=7, h_freq=30, picks=None):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.picks = picks
        self.fs = 250

    def transform(self, X, *args):
        n_channels = 4
        info = mne.create_info(
            ch_names=['C3', 'C4', 'P3', 'P4'],
            ch_types=['eeg', 'eeg', 'eeg', 'eeg'],
            sfreq=self.fs
        )

        raws = [mne.io.RawArray(x, info) for x in X]
        raws = [raw.filter(self.l_freq, self.h_freq, fir_design='firwin', skip_by_annotation='edge') for raw in raws]

        return np.array([data for data, times in [raw[:] for raw in raws]])

    def fit(self, *_):
        return self

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
