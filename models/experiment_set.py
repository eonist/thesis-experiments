import copy
import datetime
import json

import pandas
from tabulate import tabulate

from config import Path, CV_SPLITS
from models.experiment import Experiment
from utils.enums import DSType
from utils.prints import Print
from utils.progress_bar import ProgressBar
from utils.utils import create_path_if_not_existing, datestamp_str, flatten_dict

pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

# <--- PARAMETER GRIDS --->

param_grid = {
    "dataset_type": ["none_rest", "arm_foot", "left_right"],
    "classifier": ["svm", "lda"],
    "preprocessor": [
        "filter;csp;mean_power",
        "csp;mean_power",
        "emd;stats"
    ]
}

conditional_param_grid = {
    "svm": {
        "kernel": ["linear", "rbf", "sigmoid"]
    },
    "filter": {
        "kernel": ["mne", "custom"],
        "l_freq": [6, 8, 10],
        "h_freq": [12, 20, 30]
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
        print(pandas.DataFrame([flatten_dict(e) for e in exp_params_list]))
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
        pb = ProgressBar.include("exp_set", iterable=self.exp_params_list)

        for exp_params in self.exp_params_list:
            exp = Experiment.from_params(exp_params)
            exp.cv_splits = self.cv_splits

            exp.run()
            self.experiments.append(exp)

        pb.close()

        self.generate_report()
        # notify("ExperimentSet finished running", "")

    def generate_report(self):
        create_path_if_not_existing(Path.exp_logs)

        fn = self.filename("exp_set_results", "md")
        fp = "/".join([Path.exp_logs, fn])

        with open(fp, 'w+') as file:
            Print.point("Writing File Header")
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

            experiments = sorted(self.experiments, key=lambda x: x.results["accuracy"], reverse=True)

            for exp in experiments:
                flat_params = flatten_dict(exp.raw_params)

                res += "---\n\n"
                res += "### Accuracy: {}\n".format(round(exp.results["accuracy"], 2))

                res += "### Relevant parameters\n"
                relevant_params = {key: flat_params[key] for key in self.relevant_keys}
                params_df = pandas.DataFrame([relevant_params])
                res += tabulate(params_df, tablefmt="pipe", headers="keys", showindex=False) + "\n"

                res += "### All parameters\n"
                params_df = pandas.DataFrame([flat_params])
                res += tabulate(params_df, tablefmt="pipe", headers="keys", showindex=False) + "\n"

                res += "#### Confusion Matrix\n"

                c_matrix_df = pandas.DataFrame(exp.results["confusion_matrix"])
                res += tabulate(c_matrix_df, tablefmt="pipe", headers="keys", showindex=False) + "\n"

            file.write(res)


# <--- RUN CODE --->


if __name__ == '__main__':
    params = {
        "dataset_type": DSType.NONE_REST.value,
        "classifier": "svm",
        "preprocessor": "filter;csp;mean_power",
        "svm": {
            "kernel": "linear"
        },
        "filter": {
            "kernel": "mne",
            "l_freq": 6,
            "h_freq": 30
        },
        "csp": {
            # "kernel": "mne",
            # "n_components": 2
        },
        "mean_power": {
            # "log": True,
        },
        "lda": {
            "solver": "svd"
        }

    }

    exp_set = ExperimentSet(cv_splits=3, **params)

    exp_set.run_experiments()
