import binascii
import json
import multiprocessing as mp
import time
from queue import Empty

import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from config import CV_SPLITS, TEST_SIZE
from models.my_pipeline import CustomPipeline
from models.neural_network import NeuralNetwork
from models.session import Session
from transformers import Filter, CSP, StatisticalFeatures, MeanPower, EMD
from transformers.wavelet import Wavelet
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
    "wavelet": Wavelet,
    "stats": StatisticalFeatures,
    "mean_power": MeanPower
}


class Experiment:
    def __init__(self, pipeline_items, **kwargs):
        self.cv_splits = CV_SPLITS
        self.test_size = TEST_SIZE
        self.raw_params = kwargs.get('raw_params', dict())
        self.dataset_type = DSType.from_string(kwargs["dataset_type"])
        self.window_length = kwargs["window_length"]

        self.pipeline_items = pipeline_items
        self.pipeline = self.create_pipeline()

        self.multiprocessing = False

        self.datasets = None
        self.sessions = None
        self.cv_reports = []

        self.report = dict(
            dataset_type=self.dataset_type,
            raw_params=self.raw_params,
            time={}
        )

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

    def set_datasets(self, ds_collection):
        self.datasets = ds_collection.datasets(self.dataset_type, self.window_length)

    def __str__(self):
        return "{}: {}".format(self.dataset_type, " -> ".join(self.pipeline_items))

    def create_pipeline(self):
        pipeline_input = []

        for item in self.pipeline_items:
            params = self.raw_params[item] if item in self.raw_params else {}
            initializer = pipeline_classes[item]
            pipeline_input.append((item, initializer(**params)))

        return CustomPipeline(pipeline_input)

    def run(self, sessions=None):
        start_time = time.time()

        try:
            if "stats" in self.pipeline_items and "svm" in self.pipeline_items:
                raise Exception("stats and svm should not be used together")

            if self.datasets is None:
                self.datasets = list()
                for i in tqdm(range(self.cv_splits), desc="Fetching Datasets"):
                    ds = Session.full_dataset(window_length=self.window_length)
                    ds = ds.reduced_dataset(self.dataset_type)
                    ds = ds.normalize()
                    ds.shuffle()
                    self.datasets.append(ds)

            if self.multiprocessing:
                self.run_multi()
            else:
                for ds in tqdm(self.datasets, desc="Cross validating"):
                    self.cv_reports.append(self.run_cv(ds))

        except Exception as e:
            print("")
            Print.warning("Skipping experiment: {}".format(e))
            Print.ex(e)
            self.report["success"] = False
            return

        self.report = {**self.report, **avg_dict(self.cv_reports)}
        self.report["confusion_matrix"] = np.sum([r["confusion_matrix"] for r in self.cv_reports], 0)

        self.report["time"]["exp"] = (time.time() - start_time)
        self.report["accuracies"] = [r["accuracy"] for r in self.cv_reports]
        self.report["cv_splits"] = self.cv_splits
        # self.report["feature_vector_length"] = self.feature_vector_length()
        self.report["success"] = True
        self.report["dataset_lengths"] = [d.length for d in self.datasets]

    def run_multi(self):
        working_q = mp.Queue()
        output_q = mp.Queue()

        for i in range(len(self.datasets)):
            working_q.put(i)

        n_workers = np.min([mp.cpu_count(), self.cv_splits])

        Print.info("Using {} workers".format(n_workers))
        processes = [mp.Process(target=self.worker, args=(i, working_q, output_q, self.pipeline)) for i in
                     range(n_workers)]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        while True:
            try:
                self.cv_reports.append(output_q.get_nowait())
            except Empty:
                break

    def worker(self, i, working_queue, output_q, pipeline):
        while True:
            try:
                ds_index = working_queue.get_nowait()
                ds = self.datasets[ds_index]

                Print.progress("Worker {} is doing a job".format(i))

                cv_report = self.run_cv(ds, pipeline)

                output_q.put(cv_report)
            except Empty:
                break

        return

    def run_cv(self, dataset, pipeline=None):
        if pipeline is None:
            pipeline = self.pipeline

        cv_report = {"time": {}}

        ds_train, ds_test = dataset.split(include_val=False)

        start_fit_time = time.time()

        fit_output = pipeline.fit(ds_train.X, ds_train.y)
        cv_report["time"]["fit"] = time.time() - start_fit_time

        start_predict_time = time.time()
        predictions = pipeline.predict(ds_test.X)
        cv_report["time"]["pred"] = time.time() - start_predict_time

        accuracy = pipeline.score(ds_test.X, ds_test.y)

        cv_report["kappa"] = self.mod_kappa(ds_train.y, accuracy)
        cv_report["confusion_matrix"] = confusion_matrix(ds_test.y, predictions)
        cv_report["accuracy"] = accuracy
        cv_report["report"] = classification_report(y_true=ds_test.y, y_pred=predictions, output_dict=True,
                                                    target_names=[l.value for l in self.dataset_type.labels])

        return cv_report

    @staticmethod
    def mod_kappa(y_train, accuracy):
        unique, counts = np.unique(y_train, return_counts=True)
        p_e = counts[0] / len(y_train)

        return (accuracy - p_e) / (1 - p_e)

    def feature_vector_length(self):
        pipeline_input = []

        for item in self.pipeline_items[:-1]:
            params = self.raw_params[item] if item in self.raw_params else {}
            initializer = pipeline_classes[item]
            pipeline_input.append((item, initializer(**params)))

        pipeline = CustomPipeline(pipeline_input)

        ds = self.datasets[0]
        data = pipeline.fit_transform(ds.X, ds.y)

        return np.shape(data)[-1]


if __name__ == '__main__':
    exp = Experiment(["filter", "csp", "svm"], raw_params={"svm": {"kernel": "linear"}})

    exp.run()
