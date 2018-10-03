import copy
import json

import pandas

from models.experiment import Experiment
from utils.progress_bar import ProgressBar

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

    def __init__(self, **kwargs):
        self.params = dict(kwargs)

        for key in self.param_grid.keys():
            if key not in self.params:
                self.params[key] = self.param_grid[key]

        exp_params_list = self.create_all_options(self.params)

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

        exp_params_list = self.create_all_options(exp_params_list)

        set_of_jsons = {json.dumps(d, sort_keys=True) for d in exp_params_list}
        exp_params_list = [json.loads(t) for t in set_of_jsons]

        print("")
        print(pandas.DataFrame(exp_params_list))

        self.exp_params_list = exp_params_list

    def create_all_options(self, params):
        res = []
        if isinstance(params, list):
            for item in params:
                res += self.create_all_options(item)
        else:
            found_list = False
            for key, val in params.items():
                if isinstance(val, list):
                    found_list = True
                    for option in val:
                        new_params = copy.deepcopy(params)
                        new_params[key] = option
                        res += self.create_all_options(new_params)
                    break

                elif isinstance(val, dict):
                    for val_key, val_val in val.items():
                        if isinstance(val_val, list):
                            found_list = True
                            for option in val_val:
                                new_params = copy.deepcopy(params)
                                new_params[key][val_key] = option
                                res += self.create_all_options(new_params)
                            break

            if not found_list:
                res.append(params)

        return res

    def run_experiments(self):

        ProgressBar.include(id="exp_set", iterable=self.exp_params_list)

        for exp_params in self.exp_params_list:
            exp = Experiment.from_params(exp_params)

            exp.run()

            print(exp.results)


if __name__ == '__main__':
    exp_set = ExperimentSet(
        classifier="svm",
        dataset_type="arm-foot",
        svm={"kernel": "linear"},
        csp={"n_components": 2},
        mne_filter={"h_freq": 30}
    )

    exp_set.run_experiments()
