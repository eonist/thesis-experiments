import binascii
import json

import numpy as np
from mne.decoding import CSP
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from config import CV_SPLITS, TEST_SIZE
from models.session import Session
from transformers.mne_filter import MneFilter
from utils.progress_bar import ProgressBar

pipeline_classes = {
    "svm": svm.SVC,
    "lda": LinearDiscriminantAnalysis,
    "csp": CSP,
    "mne_filter": MneFilter
}


class Experiment:
    def __init__(self, pipeline_items, **kwargs):
        self.cv_splits = CV_SPLITS
        self.test_size = TEST_SIZE
        self.raw_params = kwargs.get('raw_params', None)
        self.dataset_type = kwargs.get('dataset_type', "arm-foot")

        self.pipeline_items = pipeline_items
        self.pipeline_params = kwargs.get('pipeline_params', dict())

        self.pipeline = self.create_pipeline()

        self.results = {}

    @classmethod
    def from_params(cls, params):
        pipeline_items = params["preprocessor"].split(";")
        pipeline_items.append(params["classifier"])
        return cls(pipeline_items, **params, raw_params=params)

    @classmethod
    def from_hex(cls, hex_str):
        json_params = binascii.unhexlify(hex_str).decode("utf-8")
        raw_params = json.loads(json_params)
        return cls.from_params(raw_params)

    def hex_string(self):
        json_params = json.dumps(self.raw_params, sort_keys=True)
        bytes = json_params.encode()
        res = binascii.hexlify(bytes)
        return res

    def __str__(self):
        return "{}: {}".format(self.dataset_type, " -> ".join(self.pipeline_items))

    def create_pipeline(self):
        pipeline_input = []

        for item in self.pipeline_items:
            params = self.pipeline_params[item] if item in self.pipeline_params else {}
            initializer = pipeline_classes[item]
            pipeline_input.append((item, initializer(**params)))

        return Pipeline(pipeline_input)

    def run(self):
        accuracies = []
        c_matrix = np.zeros([2, 2])

        pb_id = "exp_run"
        pb = ProgressBar.include(pb_id, total=self.cv_splits)

        for i in range(self.cv_splits):
            ds = Session.full_dataset()
            ds = ds.binary_dataset(self.dataset_type)
            ds.shuffle()
            ds_train, ds_test = ds.split(include_val=False)

            self.pipeline.fit(ds_train.X, ds_train.y)

            accuracy = self.pipeline.score(ds_test.X, ds_test.y)
            predictions = self.pipeline.predict(ds_test.X)

            c_matrix = confusion_matrix(ds_test.y, predictions)

            accuracies.append(accuracy)
            c_matrix += c_matrix

            pb.increment(pb_id)

        self.results["accuracy"] = np.mean(accuracies)
        self.results["confusion_matrix"] = c_matrix


if __name__ == '__main__':
    exp = Experiment(["mne_filter", "csp", "svm"], pipeline_params={"svm": {"kernel": "linear"}})

    exp.run()
