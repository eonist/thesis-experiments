import copy
import datetime
import multiprocessing as mp
import time
from queue import Empty

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CV_SPLITS, Path
from models.dataset_collection import DatasetCollection
from models.experiment import Experiment
from models.report import Report
from utils.enums import DSType
from utils.prints import Print
from utils.utils import datestamp_str, flatten_dict

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# <--- PARAMETER GRIDS --->


param_grid = {
    "window_length": [50, 100, 250],
    "dataset_type": [str(t) for t in DSType.variants()],
    "sample_trim": ["0;0", "0;1", "0;2", "0;3"],
    "ds_split_by": ["session", "user", "random"],
    "classifier": ["svm", "lda", "rfc", "bagging", "tree", "knn", "gaussian"],
    "preprocessor": [
        "filter;csp;mean_power",
        "emd;csp;mean_power",
        "csp;mean_power",
        "emd;stats",
        "stats",
        "mean_power",
        "dwt;stats",
        "dwt;mean_power",
        "dwt;csp;mean_power"
    ]
}

conditional_param_grid = {
    "svm": {
        "kernel": ["linear", "rbf", "sigmoid"]
    },
    "nn": {
        "verbose": 0,
        "epochs": [10, 50, 100, 200],
        "loss": ["mean_squared_error", "binary_crossentropy"],
        "n_layers": [3, 5, 10],
        "start_units": [100, 50, 20, 10]
    },
    "rfc": {
        "n_estimators": [1, 10, 100],
        "criterion": ["gini", "entropy"]
    },
    "filter": {
        "kernel": ["mne", "custom"],
        "l_freq": [7, 10, 15, 20],
        "h_freq": [30, 50, None],
        "band": ["delta", "theta", "alpha", "beta", "gamma"]
    },
    "csp": {
        "kernel": ["mne", "custom"],
        "n_components": [1, 2, 4],
        "mode": ["1vall", "1v1"]
    },
    "lda": {
        "solver": "lsqr"
    },
    "mean_power": {
        "log": [True, False]
    },
    "emd": {
        "mode": ["set_max", "minkowski"],
        "n_imfs": [1, 2],
        "max_iter": [10, 500, 2000],
        "subtract_residue": [True, False]
    },
    "stats": {
        "features": ["__all__", "__fast__"],
        "splits": 1
    },
    "dwt": {
        "dim": [1, 2],
        # "wavelet": pywt.wavelist(kind="discrete"),
        "wavelet": ["db1", "rbio6.8", "rbio2.6", "sym2", "db2", "bior2.4", "sym5"]
    }
}


class ExperimentSet:
    def __init__(self, description="", hypothesis="", cv_splits=CV_SPLITS, **kwargs):
        self.params = dict(kwargs)
        self.init_time = datetime.datetime.now()
        self.exp_params_list = []
        self.exp_reports = []
        self.best_exp = None

        self.description = description
        self.hypothesis = hypothesis

        self.run_time = None
        self.multiprocessing = None
        self.save_best = kwargs.get('save_best', False)

        self.cv_splits = cv_splits

        self.relevant_keys = []
        self.pipeline_items = []

        self.create_experiment_params()

    def filename(self, prefix, suffix):
        return "{}_{}.{}".format(prefix, datestamp_str(self.init_time, file=True), suffix)

    def reproduction_params(self, as_string=False):
        params = {}

        for key in param_grid.keys():
            if key in self.params:
                params[key] = self.params[key]
            else:
                params[key] = param_grid[key]

        for key in conditional_param_grid.keys():
            if key in self.pipeline_items:
                params[key] = {}

                if key in self.params:
                    for inner_key in conditional_param_grid[key]:
                        if inner_key in self.params[key]:
                            params[key][inner_key] = self.params[key][inner_key]
                        else:
                            params[key][inner_key] = conditional_param_grid[key][inner_key]
                else:
                    params[key] = conditional_param_grid[key]

        return params

    # <--- EXPERIMENT GENERATION --->

    def create_experiment_params(self):
        Print.point("Generating Experiments")
        for key in param_grid.keys():
            if key not in self.params:
                self.params[key] = param_grid[key]

        exp_params_list = self.recurse_flatten(self.params)

        for params in exp_params_list:
            pipeline_items = params["preprocessor"].split(";")
            pipeline_items.append(params["classifier"])
            self.pipeline_items = list(set(self.pipeline_items + pipeline_items))

            for key, val in conditional_param_grid.items():
                key = key
                if key in pipeline_items:
                    if isinstance(val, dict):
                        for val_key, val_val in val.items():
                            if key in self.params:
                                if val_key in self.params[key]:
                                    params[key][val_key] = self.params[key][val_key]
                                else:
                                    params[key][val_key] = val_val
                            else:
                                params[key] = val
                    else:
                        params[key] = self.params[key] if key in self.params else val
                else:
                    if key in params:
                        del params[key]

        exp_params_list = self.recurse_flatten(exp_params_list)

        # The following two lines remove duplicate configurations

        out = []
        for v in exp_params_list:
            if v not in out:
                out.append(v)

        exp_params_list = out
        # set_of_jsons = {json.dumps(d, sort_keys=True) for d in exp_params_list}
        # exp_params_list = [json.loads(t) for t in set_of_jsons]

        Print.start("")
        print(pd.DataFrame([flatten_dict(e) for e in exp_params_list]))
        print("\n\n")

        self.exp_params_list = exp_params_list

    def recurse_flatten(self, params):
        res = []
        if isinstance(params, list):
            for item in params:
                res += self.recurse_flatten(item)
        else:
            found_list = False
            for key, val in params.items():
                if isinstance(val, list):
                    self.relevant_keys.append(key)
                    found_list = True
                    for option in val:
                        new_params = copy.deepcopy(params)
                        new_params[key] = option
                        res += self.recurse_flatten(new_params)
                    break
                elif isinstance(val, dict):
                    for val_key, val_val in val.items():
                        if isinstance(val_val, list):
                            if key in self.pipeline_items:
                                self.relevant_keys.append("{}__{}".format(key, val_key))
                            found_list = True
                            for option in val_val:
                                new_params = copy.deepcopy(params)
                                new_params[key][val_key] = option
                                res += self.recurse_flatten(new_params)
                            break

            if not found_list:
                res.append(params)

        return res

    # <--- EXPERIMENT EXECUTION --->

    def run_experiments(self, fast_datasets=False):
        time.sleep(1)
        start_run_time = time.time()

        ds_collection = DatasetCollection.from_params(self.params, self.cv_splits, fast=fast_datasets)

        if self.multiprocessing == "exp":
            self.run_multi(ds_collection)
        else:
            for i, exp_params in enumerate(tqdm(self.exp_params_list, desc="Running Experiments")):
                exp = Experiment.from_params(exp_params)
                exp.cv_splits = self.cv_splits
                exp.index = i
                exp.set_datasets(ds_collection)

                exp.multiprocessing = (self.multiprocessing == "cv")

                exp.run()

                if self.best_exp is None or exp.report["accuracy"] > self.best_exp.report["accuracy"]:
                    Print.good("New best: {}".format(np.round(exp.report["accuracy"], 3)))
                    self.best_exp = exp

                self.exp_reports.append(exp.report)

        self.run_time = time.time() - start_run_time
        self.generate_report()

        if self.save_best:
            from sklearn.externals import joblib
            fp = Path.classifiers + '/' + "classifier1.pkl"

            joblib.dump(self.best_exp.pipeline, fp)
        # notify("ExperimentSet finished running", "")

    def generate_report(self):
        print("\n")
        Print.success("Generating Report")

        report = Report(self, self.exp_reports)
        report.generate()

    @staticmethod
    def worker(i, working_queue, output_q, cv_splits, ds_collection):
        while True:
            try:
                exp_params = working_queue.get_nowait()
                exp = Experiment.from_params(exp_params)
                exp.cv_splits = cv_splits
                exp.set_datasets(ds_collection)

                Print.progress("{}: Running Experiment".format(i))
                exp.run()
                output_q.put(exp.report)
            except Empty:
                Print.info("Queue Empty")
                break

        return

    def run_multi(self, ds_collection):
        working_q = mp.Queue()
        output_q = mp.Queue()

        for exp_params in self.exp_params_list:
            working_q.put(exp_params)

        n_workers = np.min([mp.cpu_count(), len(self.exp_params_list)])

        Print.info("Using {} workers".format(n_workers))
        processes = [mp.Process(target=self.worker, args=(i, working_q, output_q, self.cv_splits, ds_collection)) for i
                     in
                     range(n_workers)]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        while True:
            try:
                self.exp_reports.append(output_q.get_nowait())
            except Empty:
                break


# <--- RUN CODE --->


if __name__ == '__main__':
    params = {
        "window_length": 100,
        "dataset_type": "LA-RA-LF-RF",
        "sample_trim": "0;3",
        "ds_split_by": "session",
        "classifier": "rfc",
        "preprocessor": ["mean_power", "stats", "csp;mean_power", "filter;mean_power", "filter;stats", "emd;stats",
                         "dwt;stats", "filter;csp;mean_power"],
        "svm": {
            "kernel": "rbf"
        },
        "lda": {
            "solver": "lsqr"
        },
        "rfc": {
            "n_estimators": 100,
            "criterion": "entropy"
        },
        "filter": {
            "kernel": "mne",
            "l_freq": 20,
            "h_freq": None,
            "band": None
        },
        "csp": {
            "kernel": "custom",
            "n_components": 4,
            "mode": "1vall"
        },
        "emd": {
            "mode": "set_max",
            "n_imfs": 1,
            "max_iter": 10,
            "subtract_residue": True
        },
        "dwt": {
            "dim": 2,
            "wavelet": "bior2.4"
        },
        "mean_power": {
            "log": True,
        },
        "stats": {
            "features": "__env__",
            "splits": 1
        }
    }

    exp_set = ExperimentSet(cv_splits=24, **params)
    exp_set.multiprocessing = "cv"
    exp_set.run_experiments(fast_datasets=False)
