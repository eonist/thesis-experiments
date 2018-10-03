import numpy as np
from mne.decoding import CSP
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from models.session import Session
from transformers.csp import CSP as MyCSP
from transformers.mne_filter import MneFilter
from utils.prints import Print


class ExperimentOld:

    def __init__(self):
        self.pipeline = None
        self.cv_splits = 10
        self.test_size = 0.2
        self.dataset_type = "arm-foot"
        self.param_grid = {}

    def run(self):
        scores = []
        c_matrix = np.zeros([2, 2])
        for i in range(self.cv_splits):
            ds = Session.full_dataset()
            ds = ds.binary_dataset(self.dataset_type)
            ds.shuffle()
            ds_train, ds_test = ds.split(include_val=False)

            cv = GridSearchCV(self.pipeline, param_grid=self.param_grid)

            cv.fit(ds_train.X, ds_train.y)

            score = cv.score(ds_test.X, ds_test.y)
            predictions = cv.predict(ds_test.X)

            c_matrix = confusion_matrix(ds_test.y, predictions)

            Print.success(round(score, 2))
            Print.pandas(c_matrix)
            scores.append(score)
            c_matrix += c_matrix

        return scores


if __name__ == '__main__':
    exp = ExperimentOld()
    # exp.pipeline = make_pipeline(MneFilter(), CSP())
    csp = CSP()
    my_csp = MyCSP(avg_power=True)
    exp.pipeline = make_pipeline(MneFilter(), csp, svm.SVC())
    print(exp.pipeline)
    exp.param_grid = {"csp__log": [True, False], "svc__kernel": ["linear", "rbf", "sigmoid"]}
    exp.run()
