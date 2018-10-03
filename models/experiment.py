import numpy as np
from mne.decoding import CSP
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from models.session import Session
from transformers.mne_filter import MneFilter
from utils.prints import Print
from utils.progress_bar import ProgressBar

pipeline_classes = {
    "svm": svm.SVC,
    "lda": LinearDiscriminantAnalysis,
    "csp": CSP,
    "mne_filter": MneFilter
}


class Experiment:
    def __init__(self, pipeline_items, **kwargs):
        self.cv_splits = 3
        self.test_size = 0.2
        self.dataset_type = kwargs.get('dataset_type', "arm-foot")

        self.pipeline_items = pipeline_items
        self.pipeline_params = kwargs.get('pipeline_params', dict())

        self.pipeline = self.create_pipeline()

        self.results = {}

    @classmethod
    def from_params(cls, params):
        pipeline_items = params["preprocessor"].split(";")
        pipeline_items.append(params["classifier"])
        return cls(pipeline_items, **params)

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
        Print.start(self.pipeline_items)

        scores = []
        c_matrix = np.zeros([2, 2])

        pb = ProgressBar.include(id="exp", total=self.cv_splits)

        for i in range(self.cv_splits):
            ds = Session.full_dataset()
            ds = ds.binary_dataset(self.dataset_type)
            ds.shuffle()
            ds_train, ds_test = ds.split(include_val=False)

            self.pipeline.fit(ds_train.X, ds_train.y)

            score = self.pipeline.score(ds_test.X, ds_test.y)
            predictions = self.pipeline.predict(ds_test.X)

            c_matrix = confusion_matrix(ds_test.y, predictions)

            scores.append(score)
            c_matrix += c_matrix
            pb.increment()

        Print.success("Mean score: {}".format(np.mean(scores)))
        Print.pandas(c_matrix)

        self.results["score"] = np.mean(scores)
        self.results["confusion_matrix"] = c_matrix


if __name__ == '__main__':
    exp = Experiment(["mne_filter", "csp", "svm"], pipeline_params={"svm": {"kernel": "linear"}})

    exp.run()
