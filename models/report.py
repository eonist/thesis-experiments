import json

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import Path, DECIMALS
from utils.prints import Print
from utils.utils import create_path_if_not_existing, datestamp_str, flatten_dict, format_array


class Report:
    def __init__(self, exp_set, exp_reports):
        self.exp_reports = [r for r in exp_reports if r["success"]]

        try:
            self.exp_reports.sort(key=lambda r: r["accuracy"], reverse=True)
        except:
            pass

        self.exp_set = exp_set

    def filename(self, prefix, suffix):
        return "{}_{}.{}".format(prefix, datestamp_str(self.exp_set.init_time, file=True), suffix)

    @property
    def dir_name(self):
        return "exp_set_{}".format(datestamp_str(self.exp_set.init_time, file=True))

    @property
    def path(self):
        return "/".join([Path.exp_logs, self.dir_name])

    def generate(self):
        create_path_if_not_existing(self.path)

        self.generate_detail()
        self.generate_overview()
        self.generate_reproduction()

    def generate_detail(self):
        fn = self.filename("exp_set_detail", "md")
        Print.data(fn)
        fp = "/".join([self.path, fn])

        relevant_keys = list(set(self.exp_set.relevant_keys))

        res = "# Experiment Set Detail\n"
        res += "{}\n\n".format(datestamp_str(self.exp_set.init_time))
        res += "* **Runtime:** {}s\n".format(np.round(self.exp_set.run_time, 1))
        res += "* **Multiprocessing:** {}\n".format(self.exp_set.multiprocessing)
        res += "\n\n"
        if self.exp_set.description:
            res += "#### Description\n"
            res += self.exp_set.description + "\n"

        if self.exp_set.hypothesis:
            res += "#### Hypothesis\n"
            res += self.exp_set.hypothesis + "\n"

        res += "\n\n"
        res += "## Performance by configuration\n\n"

        for i, exp_report in enumerate(self.exp_reports):
            flat_params = flatten_dict(exp_report["raw_params"])

            res += "---\n\n"
            res += "### Entry {} accuracy: {}\n".format(i + 1, np.round(exp_report["accuracy"], DECIMALS))
            res += "* **Kappa:** {}\n".format(np.round(exp_report["kappa"], DECIMALS))
            res += "* **Average Experiment Time:** {}s\n".format(np.round(exp_report["time"]["exp"], 2))
            res += "* **Dataset type:** {}\n".format(exp_report["dataset_type"])
            res += "* **Dataset avg length:** {}\n".format(
                np.round(np.mean(exp_report["dataset_lengths"]), DECIMALS))
            # res += "* **Feature Vector Length:** {}\n".format(exp_report["feature_vector_length"])
            res += "* **CV Splits:** {}\n".format(exp_report["cv_splits"])
            res += "\n"

            res += "{}\n".format(np.round(exp_report["accuracies"], DECIMALS))

            res += "### Config\n"
            res += "**Relevant Parameters**\n\n"
            relevant_params = {key: flat_params[key] for key in relevant_keys if key in flat_params}
            params_df = pd.DataFrame([relevant_params])
            res += tabulate(params_df, tablefmt="pipe", headers="keys", showindex=False) + "\n"

            res += "**All Parameters**\n\n"
            params_df = pd.DataFrame([flat_params])
            res += tabulate(params_df.round(DECIMALS), tablefmt="pipe", headers="keys", showindex=False) + "\n"

            res += "### Details\n"

            res += "**Confusion Matrix**\n\n"
            c_matrix = exp_report["confusion_matrix"]
            class_names = exp_report["dataset_type"].labels
            c_matrix_df = pd.DataFrame(c_matrix,
                                       columns=["Pred: {}".format(l) for l in class_names],
                                       index=["__True: {}__".format(l) for l in class_names])
            res += tabulate(c_matrix_df, tablefmt="pipe", headers="keys", showindex=True) + "\n"

            res += "<!---\nConfusion Matrix in LaTeX\n"
            res += tabulate(c_matrix_df, tablefmt="latex", headers="keys", showindex=False) + "\n"
            res += "--->\n"

            # Formats the confusion matrix as
            res += "<!---\nConfusion Matrix Raw\n"
            res += "c_matrix = np.array({})\n".format(format_array(c_matrix))
            res += "class_names = {}\n".format(format_array(class_names))
            res += "--->\n"

            # res += "**Report**\n\n"
            # report = exp_report["report"]
            # report_df = pd.DataFrame.from_dict(report)
            # report_key = list(report.keys())[0]
            # index = ["__{}__".format(key) for key in report[report_key].keys()]
            # res += tabulate(report_df.round(DECIMALS), tablefmt="pipe", headers="keys", showindex=index) + "\n"

            res += "**Time**\n\n"
            time_df = pd.DataFrame([exp_report["time"]])
            res += tabulate(time_df.round(DECIMALS), tablefmt="pipe", headers="keys", showindex=False) + "\n"

        with open(fp, 'w+') as file:
            file.write(res)

    def generate_overview(self):
        fn = self.filename("exp_set_overview", "md")
        Print.data(fn)
        fp = "/".join([self.path, fn])
        relevant_keys = list(set(self.exp_set.relevant_keys))
        Print.data(relevant_keys)
        exp_summary = np.empty(shape=[len(self.exp_reports), 3 + len(relevant_keys)], dtype="U25")

        res = "# Experiment Set Overview\n"

        res += "## Performance by relevant params\n\n"

        param_performances = {param: self.param_performance(param) for param in self.all_relevant_params()}

        for param_name, param_vals in param_performances.items():
            res += "### {}\n\n".format(param_name)

            param_vals_list = sorted(list(param_vals.items()), key=lambda x: x[1], reverse=True)

            res += "\n".join(["* **{}:** {}".format(e[0], np.round(e[1], DECIMALS)) for e in param_vals_list])
            res += "\n\n"

        res += "\n\n"

        res += "## Performance Overview\n\n"

        for i, exp_report in enumerate(self.exp_reports):
            flat_params = flatten_dict(exp_report["raw_params"])
            relevant_params = np.empty(shape=[len(relevant_keys)], dtype="U25")

            for j, key in enumerate(relevant_keys):
                if key in flat_params:
                    relevant_params[j] = flat_params[key]
                else:
                    relevant_params[j] = "-"

            acc_string = "{}%".format(np.round(100 * exp_report["accuracy"], 1))
            kappa_string = "{}".format(np.round(exp_report["kappa"], 3))
            time_string = "{}s".format(np.round(exp_report["time"]["exp"], 2))

            exp_summary[i, :3] = [acc_string, kappa_string, time_string]
            exp_summary[i, 3:] = relevant_params

        df_perf1 = pd.DataFrame(exp_summary, columns=["Accuracy", "Kappa", "Avg Time"] + relevant_keys,
                                copy=True)
        df_perf1.sort_values(by=["Accuracy"], axis=0, ascending=False, inplace=True)
        res += tabulate(df_perf1, tablefmt="pipe", headers="keys", showindex=False) + "\n"

        res += "<!---\nResults in LaTeX\n"
        res += tabulate(df_perf1, tablefmt="latex", headers="keys", showindex=False) + "\n"
        res += "--->\n"

        with open(fp, 'w+') as file:
            file.write(res)

    def generate_reproduction(self):
        fn = self.filename("exp_set_reproduction", "md")
        Print.data(fn)
        fp = "/".join([self.path, fn])
        relevant_keys = list(set(self.exp_set.relevant_keys))
        exp_summary = np.empty(shape=[len(self.exp_reports), 3 + len(relevant_keys)], dtype="U25")

        res = "# Experiment Set Reproduction\n"

        res += "## Code\n"

        res += "```\n"
        code = "params = "
        code += json.dumps(self.exp_set.reproduction_params(), indent=4) + "\n\n"
        code += "exp_set = ExperimentSet(cv_splits={},  **params)\n".format(self.exp_set.cv_splits)
        code += "exp_set.multiprocessing = \"cv\"\n"
        code += "exp_set.run_experiments()"
        res += code + "\n"
        res += "```\n\n"

        res += "<!--- Figure in LaTeX\n"
        res += self.python_figure(code) + "\n"
        res += "--->\n"

        with open(fp, 'w+') as file:
            file.write(res)

    # <--- HELPER METHODS --->

    def python_figure(self, code):
        res = "\\begin{figure}[h!]\n\\centering\n\\begin{python}\n"
        res += code + "\n"
        res += "\\end{python}\n\\caption{Code for executing Experiment \\ref{}}\n\\label{code:exp_[id]_run}\n\\end{figure}"

        return res

    def relevant_params(self, flat_params):
        relevant_keys = list(set(self.exp_set.relevant_keys))
        return {key: flat_params[key] for key in relevant_keys if key in flat_params}

    def all_relevant_params(self):
        res = []

        for exp_report in self.exp_reports:
            flat_params = flatten_dict(exp_report["raw_params"])
            relevant_params = self.relevant_params(flat_params)

            for param in relevant_params:
                if param not in res:
                    res.append(param)

        return res

    def param_performance(self, param):
        res = {}

        for exp_report in self.exp_reports:
            flat_params = flatten_dict(exp_report["raw_params"])

            if param in flat_params:
                param_val = flat_params[param]

                if param_val not in res:
                    res[param_val] = []

                res[param_val].append(exp_report["accuracy"])

        for key, val in res.items():
            res[key] = np.mean(val)

        return res
