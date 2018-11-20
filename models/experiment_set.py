import copy
import datetime
import multiprocessing as mp
import time
from queue import Empty

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CV_SPLITS
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
    "dataset_type": [str(t) for t in DSType.variants()],
    "window_length": [50, 100, 250],
    "classifier": ["svm", "lda", "random_forest", "bagging", "tree", "knn", "gaussian"],
    "preprocessor": [
        "filter;csp;mean_power",
        "emd;csp;mean_power",
        "csp;mean_power",
        "emd;stats",
        "stats",
        "mean_power",
        "wavelet;stats",
        "wavelet;mean_power",
        "wavelet;csp;mean_power"
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
    "random_forest": {
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
        "n_imfs": [1, 2, 4],
        "imf_picks": ["1,2", "minkowski"],
        "max_iter": [10, 20, 100, 500, 2000],
        "subtract_residue": [True, False]
    },
    "stats": {
        "features": ["__all__", "__fast__"]
    },
    "wavelet": {
        "n_dimensions": [1, 2],
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

        self.description = description
        self.hypothesis = hypothesis

        self.run_time = None
        self.multiprocessing = None

        self.cv_splits = cv_splits

        self.relevant_keys = []

        self.create_experiment_params()

    def filename(self, prefix, suffix):
        return "{}_{}.{}".format(prefix, datestamp_str(self.init_time, file=True), suffix)

    # <--- EXPERIMENT GENERATION --->

    def create_experiment_params(self):
        for key in param_grid.keys():
            if key not in self.params:
                self.params[key] = param_grid[key]

        exp_params_list = self.recurse_flatten(self.params)

        for params in exp_params_list:
            pipeline_items = params["preprocessor"].split(";")
            pipeline_items.append(params["classifier"])

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

    def run_experiments(self):
        time.sleep(1)
        start_run_time = time.time()

        ds_collection = DatasetCollection.from_params(self.params, self.cv_splits)

        if self.multiprocessing == "exp":
            self.run_multi(ds_collection)
        else:
            for exp_params in tqdm(self.exp_params_list, desc="Running Experiments"):
                exp = Experiment.from_params(exp_params)
                exp.cv_splits = self.cv_splits
                exp.set_datasets(ds_collection)

                exp.multiprocessing = (self.multiprocessing == "cv")

                exp.run()
                self.exp_reports.append(exp.report)

        self.run_time = time.time() - start_run_time
        self.generate_report()
        # notify("ExperimentSet finished running", "")

    def generate_report(self):
        print("\n")
        Print.success("Generating Report")

        report = Report(self, self.exp_reports)
        report.generate()
        report.generate_latex()

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
        "window_length": [10, 100, 250],
        "dataset_type": "none_arm/left_arm/right_foot/left_foot/right",
        "classifier": ["svm", "lda", "random_forest", "bagging", "tree", "knn", "gaussian"],
        "preprocessor": "filter;csp;mean_power",
        "svm": {
            "kernel": ["linear", "rbf", "sigmoid"]
        },
        "random_forest": {
            "n_estimators": 100,
            "criterion": ["gini", "entropy"]
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
            "mode": ["1vall", "1v1"]
        },
        "mean_power": {
            "log": True,
        }
    }

    exp_set = ExperimentSet(cv_splits=8, **params)
    exp_set.multiprocessing = "cv"
    exp_set.run_experiments()
