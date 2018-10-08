import copy
import datetime
import json

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from config import Path, CV_SPLITS, DECIMALS
from models.experiment import Experiment
from models.session import Session
from utils.enums import DSType
from utils.prints import Print
from utils.utils import create_path_if_not_existing, datestamp_str, flatten_dict

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# <--- PARAMETER GRIDS --->

param_grid = {
    "dataset_type": ["none_rest", "arm_foot", "left_right"],
    "classifier": ["svm", "lda", "random_forest", "bagging", "tree", "knn", "gaussian", "nn"],
    "preprocessor": [
        "filter;csp;mean_power",
        "csp;mean_power",
        "emd;stats",
        "stats",
        "mean_power"
    ]
}

conditional_param_grid = {
    "svm": {
        "kernel": ["linear", "rbf", "sigmoid"]
    },
    "nn": {
        "epochs": [100, 200],
        "verbose": 0
    },
    "filter": {
        "kernel": ["mne", "custom"],
        "l_freq": [1, 7, 10],
        "h_freq": [12, 30, None]
    },
    "csp": {
        "kernel": ["mne", "custom"],
        "n_components": [1, 2, 4]
    },
    "lda": {
        "solver": ["svd", "lsqr", "eigen"]
    },
    "mean_power": {
        "log": [True, False]
    },
    "emd": {
        "n_imfs": [1, 2, 4]
    },
    "stats": {
        "features": ["__all__", "__fast__"]
    }
}


class ExperimentSet:
    def __init__(self, description="", hypothesis="", cv_splits=CV_SPLITS, **kwargs):
        self.params = dict(kwargs)
        self.init_time = datetime.datetime.now()
        self.exp_params_list = []
        self.experiments = []

        self.description = description
        self.hypothesis = hypothesis

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
                self.relevant_keys.append(key)

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
                                    self.relevant_keys.append("{}__{}".format(key, val_key))
                                    params[key][val_key] = val_val
                            else:
                                self.relevant_keys.append(key)
                                params[key] = val
                    else:
                        params[key] = self.params[key] if key in self.params else val
                else:
                    if key in params:
                        del params[key]

        exp_params_list = self.recurse_flatten(exp_params_list)

        # The following two lines remove duplicate configurations
        set_of_jsons = {json.dumps(d, sort_keys=True) for d in exp_params_list}
        exp_params_list = [json.loads(t) for t in set_of_jsons]

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
                    found_list = True
                    for option in val:
                        new_params = copy.deepcopy(params)
                        new_params[key] = option
                        res += self.recurse_flatten(new_params)
                    break

                elif isinstance(val, dict):
                    for val_key, val_val in val.items():
                        if isinstance(val_val, list):
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

        datasets = []
        for ds in tqdm(Session.full_dataset_gen(count=self.cv_splits), total=self.cv_splits, desc="Fetching DataSets"):
            ds = ds.binary_dataset(self.params["dataset_type"])
            ds.shuffle()
            datasets.append(ds)

        for exp_params in tqdm(self.exp_params_list, desc="Running Experiments"):
            exp = Experiment.from_params(exp_params)
            exp.cv_splits = self.cv_splits
            exp.datasets = datasets

            exp.run()
            self.experiments.append(exp)

        self.generate_report()
        # notify("ExperimentSet finished running", "")

    def generate_report(self):
        print("\n")
        Print.success("Generating Report")
        create_path_if_not_existing(Path.exp_logs)

        fn = self.filename("exp_set_results", "md")
        fp = "/".join([Path.exp_logs, fn])

        with open(fp, 'w+') as file:
            res = "# Experiment Set\n"
            res += "{}\n".format(datestamp_str(self.init_time))
            res += "\n\n"
            if self.description:
                res += "#### Description\n"
                res += self.description + "\n"

            if self.hypothesis:
                res += "#### Hypothesis\n"
                res += self.hypothesis + "\n"

            res += "\n\n"
            res += "## Performance by configuration\n\n"

            experiments = [exp for exp in self.experiments if exp.results["success"]]
            experiments = sorted(experiments, key=lambda x: x.results["kappa"], reverse=True)

            for exp in experiments:
                flat_params = flatten_dict(exp.raw_params)

                res += "---\n\n"
                res += "### Kappa: {}\n".format(np.round(exp.results["kappa"], DECIMALS))

                res += "* **Dataset type:** {}\n".format(exp.dataset_type)
                res += "* **Accuracy:** {}\n".format(np.round(exp.results["accuracy"], DECIMALS))
                res += "* **Average Time:** {}\n".format(np.round(exp.results["time"]["exp"], DECIMALS))
                res += "* **CV Splits:** {}\n".format(exp.results["cv_splits"])
                res += "\n"

                res += "{}\n".format(np.round(exp.results["accuracies"], DECIMALS))

                res += "### Config\n"
                res += "**Relevant Parameters**\n\n"
                relevant_params = {key: flat_params[key] for key in self.relevant_keys if key in flat_params}
                params_df = pd.DataFrame([relevant_params])
                res += tabulate(params_df, tablefmt="pipe", headers="keys", showindex=False) + "\n"

                res += "**All Parameters**\n\n"
                params_df = pd.DataFrame([flat_params])
                res += tabulate(params_df.round(DECIMALS), tablefmt="pipe", headers="keys", showindex=False) + "\n"

                res += "### Details\n"

                res += "**Confusion Matrix**\n\n"
                c_matrix_df = pd.DataFrame(exp.results["confusion_matrix"],
                                           columns=["Pred: {}".format(l) for l in exp.dataset_type.labels],
                                           index=["__True: {}__".format(l) for l in exp.dataset_type.labels])
                res += tabulate(c_matrix_df, tablefmt="pipe", headers="keys", showindex=True) + "\n"

                res += "**Report**\n\n"
                report = exp.results["avg_report"]
                report_df = pd.DataFrame.from_dict(report)
                report_key = list(report.keys())[0]
                index = ["__{}__".format(key) for key in report[report_key].keys()]
                res += tabulate(report_df.round(DECIMALS), tablefmt="pipe", headers="keys", showindex=index) + "\n"

                res += "**Time**\n\n"
                time_df = pd.DataFrame([exp.results["time"]])
                res += tabulate(time_df.round(DECIMALS), tablefmt="pipe", headers="keys", showindex=False) + "\n"

            file.write(res)


# <--- RUN CODE --->


if __name__ == '__main__':
    params = {
        "dataset_type": str(DSType.NONE_REST),
        "classifier": "nn",
        "preprocessor": "filter;csp;mean_power",
        "svm": {
            "kernel": "linear"
        },
        "lda": {
            "solver": "lsqr"
        },
        "filter": {
            "kernel": "mne",
            "l_freq": 7,
            "h_freq": 30
        },
        "csp": {
            "kernel": "mne",
            "n_components": 4
        },
        "mean_power": {
            "log": True,
        },
        "emd": {
            "n_imfs": 1
        },
        "stats": {
            "features": "__fast__"
        }
    }

    exp_set = ExperimentSet(cv_splits=10, **params)

    exp_set.run_experiments()
