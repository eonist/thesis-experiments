import copy
import datetime
import json

import pandas
from tabulate import tabulate

from config import Path, CV_SPLITS
from models.experiment import Experiment
from utils.prints import Print
from utils.progress_bar import ProgressBar
from utils.utils import create_path_if_not_existing, datestamp_str, flatten

pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)


class ExperimentSet:
    param_grid = {
        "dataset_type": ["none-rest", "arm-foot", "right-left"],
        "classifier": ["svm", "lda"],
        "preprocessor": [
            "mne_filter;csp",
            "csp"
        ]
    }

    conditional_param_grid = {
        "svm": {
            "kernel": ["linear", "rbf", "sigmoid"]
        },
        "mne_filter": {
            "l_freq": [6, 10],
            "h_freq": [20, 30]
        },
        "csp": {
            "log": [True, False],
            "n_components": [2, 4, 6]
        },
        "lda": {
            "solver": ["svd", "lsqr", "eigen"]
        }
    }

    def __init__(self, description="", hypothesis="", cv_splits=CV_SPLITS, **kwargs):
        self.params = dict(kwargs)
        self.init_time = datetime.datetime.now()
        self.exp_params_list = []
        self.experiments = []

        self.description = description
        self.hypothesis = hypothesis

        self.cv_splits = cv_splits

        self.create_experiment_params()

    def filename(self, prefix, suffix):
        return "{}_{}.{}".format(prefix, datestamp_str(self.init_time, file=True), suffix)

    def create_experiment_params(self):
        for key in self.param_grid.keys():
            if key not in self.params:
                self.params[key] = self.param_grid[key]

        exp_params_list = self.recurse_flatten(self.params)

        for params in exp_params_list:
            pipeline_items = params["preprocessor"].split(";")
            pipeline_items.append(params["classifier"])

            for key, val in self.conditional_param_grid.items():
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

        set_of_jsons = {json.dumps(d, sort_keys=True) for d in exp_params_list}
        exp_params_list = [json.loads(t) for t in set_of_jsons]

        print("")
        print(pandas.DataFrame(exp_params_list))
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

    def run_experiments(self):
        pb = ProgressBar.include("exp_set", iterable=self.exp_params_list)

        for exp_params in self.exp_params_list:
            exp = Experiment.from_params(exp_params)
            exp.cv_splits = self.cv_splits

            exp.run()
            self.experiments.append(exp)

        for exp in self.experiments:
            print("\n\n")
            Print.success("Experiment")
            print(pandas.DataFrame([exp.raw_params]))
            print("")
            Print.point("Results\n")
            print("Accuracy: {}\n".format(round(exp.results["accuracy"], 2)))
            print(pandas.DataFrame(exp.results["confusion_matrix"]))
            print("\n")

        pb.close()

        self.generate_report2()

    def generate_report2(self):
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

            experiments = sorted(self.experiments, key=lambda x: x.results["accuracy"], reverse=True)

            for exp in experiments:
                res += "---\n\n"
                res += "### Accuracy: {}\n".format(round(exp.results["accuracy"], 2))

                print(exp.raw_params)
                print(flatten(exp.raw_params))
                params_df = pandas.DataFrame([flatten(exp.raw_params)])
                res += tabulate(params_df, tablefmt="pipe", headers="keys", showindex=False) + "\n"

                res += "#### Confusion Matrix\n"

                c_matrix_df = pandas.DataFrame(exp.results["confusion_matrix"])
                res += tabulate(c_matrix_df, tablefmt="pipe", headers="keys", showindex=False) + "\n"

            file.write(res)

    def generate_report(self):
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
            res += "### Performance by configuration\n\n"

            results = [exp.results for exp in self.experiments]

            df_perf1 = pandas.DataFrame(results, copy=True)
            df_perf1 = df_perf1.drop("confusion_matrix", axis=1)
            # df_perf1["Config Summary"] = [" - ".join(exp_config.summary()) for exp_config in exp_configs]
            df_perf1.sort_values(by=["accuracy"], axis=0, ascending=False, inplace=True)
            res += tabulate(df_perf1, tablefmt="pipe", headers="keys", showindex=False) + "\n"

            res += "<!---\nResults in LaTeX\n"

            res += tabulate(df_perf1, tablefmt="latex", headers="keys", showindex=False) + "\n"
            res += "--->\n"

            file.write(res)


if __name__ == '__main__':
    params = {
        "dataset_type": "arm-foot",
        # "classifier": "svm",
        "preprocessor": "mne_filter;csp",
        "svm": {
            # "kernel": "linear"
        },
        "mne_filter": {
            "l_freq": 6,
            # "h_freq": 30
        },
        "csp": {
            # "log": True,
            # "n_components": 2
        },
        "lda": {
            # "solver": "svd"
        }
    }

    exp_set = ExperimentSet(cv_splits=3, **params)

    exp_set.run_experiments()
