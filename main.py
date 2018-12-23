from models.experiment_set import ExperimentSet


def main():
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


if __name__ == '__main__':
    main()
