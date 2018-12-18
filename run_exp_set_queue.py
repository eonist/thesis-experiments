import json
import os

from config import Path
from models.experiment_set import ExperimentSet
from utils.prints import Print

path = Path.exp_set_queue


def pri_from_fn(fn):
    fn = fn.split(".")[0]
    last_token = fn.split("_")[3]
    priority = int(last_token[3:])
    return priority


def name_from_fn(fn):
    return fn.split("_")[2]


def run_queue():
    fns = os.listdir(path)

    fns = [fn for fn in fns if "done" not in fn]
    fns.sort(key=lambda x: pri_from_fn(x), reverse=True)

    fps = ["/".join([path, fn]) for fn in fns if "done" not in fn]

    for fp in fps:
        print("\n\n")
        Print.start("Running ExperimentSet ({})".format(name_from_fn(fp.split("/")[-1])))
        with open(fp, "r") as infile:
            json_data = json.load(infile)

            exp_set = ExperimentSet(cv_splits=8, **json_data)
            exp_set.multiprocessing = "cv"
            exp_set.run_experiments()

        tokens = fp.split(".")
        tokens[0] += "_done"
        new_fp = ".".join(tokens)
        os.rename(fp, new_fp)


if __name__ == '__main__':
    run_queue()
