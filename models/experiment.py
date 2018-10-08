import binascii
import json
import time

import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import CV_SPLITS, TEST_SIZE
from models.neural_network import NeuralNetwork
from models.session import Session
from transformers import Filter, CSP, StatisticalFeatures, MeanPower, EMD
from utils.enums import DSType
from utils.prints import Print
from utils.utils import avg_dict

pipeline_classes = {
    "svm": svm.SVC,
    "lda": LinearDiscriminantAnalysis,
    "random_forest": RandomForestClassifier,
    "bagging": BaggingClassifier,
    "tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "gaussian": GaussianNB,
    "nn": NeuralNetwork,
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
        self.dataset_type = DSType(kwargs.get('dataset_type', "arm_foot"))

        self.pipeline_items = pipeline_items
        self.pipeline = self.create_pipeline()

        self.datasets = []

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
        kappas = []
        c_matrix = np.zeros([2, 2])
        reports = []

        start_time = time.time()
        fit_time = 0
        predict_time = 0

        try:
            if self.pipeline_items == ["emd", "stats", "svm"]:
                raise Exception("emd;stats;svm should not be used together")

            if not self.datasets:
                for i in range(self.cv_splits):
                    ds = Session.full_dataset()
                    ds = ds.binary_dataset(self.dataset_type)
                    ds.shuffle()
                    self.datasets.append(ds)

            for ds in self.datasets:
                ds_train, ds_test = ds.split(include_val=False)

                start_fit_time = time.time()
                self.pipeline.fit(ds_train.X, ds_train.y)
                fit_time += time.time() - start_fit_time

                start_predict_time = time.time()
                predictions = self.pipeline.predict(ds_test.X)
                predict_time += time.time() - start_predict_time

                accuracy = self.pipeline.score(ds_test.X, ds_test.y)

                kappas.append(self.mod_kappa(ds_train.y, accuracy))
                c_matrix += confusion_matrix(ds_test.y, predictions)

                accuracies.append(accuracy)

                reports.append(classification_report(y_true=ds_test.y, y_pred=predictions, output_dict=True,
                                                     target_names=self.dataset_type.labels))

        except Exception as e:
            print("")
            Print.warning("Skipping experiment: {}".format(e))
            Print.ex(e)
            self.results["success"] = False
            return

        self.results["time"] = {
            "exp": (time.time() - start_time) / self.cv_splits,
            "fit": fit_time / self.cv_splits,
            "predict": predict_time / self.cv_splits
        }
        self.results["accuracy"] = np.mean(accuracies)
        self.results["accuracies"] = accuracies
        self.results["kappa"] = np.mean(kappas)
        self.results["confusion_matrix"] = c_matrix
        self.results["avg_report"] = avg_dict(reports)
        self.results["cv_splits"] = self.cv_splits
        self.results["success"] = True

    def mod_kappa(self, y_train, accuracy):
        unique, counts = np.unique(y_train, return_counts=True)
        p_e = counts[0] / len(y_train)

        return (accuracy - p_e) / (1 - p_e)


if __name__ == '__main__':
    exp = Experiment(["filter", "csp", "svm"], raw_params={"svm": {"kernel": "linear"}})

    exp.run()
