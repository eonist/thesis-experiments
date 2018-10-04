import binascii
import json
import time

import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import CV_SPLITS, TEST_SIZE
from models.session import Session
from transformers import Filter, CSP, StatisticalFeatures, MeanPower, EMD
from utils.progress_bar import ProgressBar

pipeline_classes = {
    "svm": svm.SVC,
    "lda": LinearDiscriminantAnalysis,
    "random_forest": RandomForestClassifier,
    "bagging": BaggingClassifier,
    "tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "gaussian": GaussianNB,
    "csp": CSP,
    "filter": Filter,
    "emd": EMD,
    "stats": StatisticalFeatures,
    "mean_power": MeanPower
}


class Experiment:
    def __init__(self, pipeline_items, **kwargs):
        self.cv_splits = CV_SPLITS
        self.test_size = TEST_SIZE
        self.raw_params = kwargs.get('raw_params', dict())
        self.dataset_type = kwargs.get('dataset_type', "arm_foot")

        self.pipeline_items = pipeline_items

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
            params = self.raw_params[item] if item in self.raw_params else {}
            initializer = pipeline_classes[item]
            pipeline_input.append((item, initializer(**params)))

        return Pipeline(pipeline_input)

    def run(self):
        accuracies = []
        c_matrix = np.zeros([2, 2])

        start_time = time.time()

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

            c_matrix += confusion_matrix(ds_test.y, predictions)

            accuracies.append(round(accuracy, 3))

            pb.increment(pb_id)

        self.results["avg_time"] = round((time.time() - start_time) / self.cv_splits, 3)
        self.results["accuracy"] = round(float(np.mean(accuracies)), 2)
        self.results["accuracies"] = accuracies
        self.results["confusion_matrix"] = c_matrix


if __name__ == '__main__':
    exp = Experiment(["filter", "csp", "svm"], raw_params={"svm": {"kernel": "linear"}})

    exp.run()
