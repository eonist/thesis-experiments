from tqdm import tqdm

from models.session import Session
from utils.utils import as_list


class DatasetCollection:
    def __init__(self, dataset_types, window_lengths, sample_trims, cv_splits, fast=False):
        self.dataset_types = as_list(dataset_types)
        self.window_lengths = as_list(window_lengths)
        self.sample_trims = as_list(sample_trims)
        self.cv_splits = cv_splits
        self.fast = fast

        self.count = len(self.window_lengths) * len(self.dataset_types) * len(self.sample_trims) * cv_splits
        self.value = None

        self._create_datasets()

    @classmethod
    def from_params(cls, params, cv_splits, fast=False):
        return cls(params["dataset_type"], params["window_length"], params["sample_trim"], cv_splits, fast=fast)

    def _create_datasets(self):
        self.value = {}

        max_sessions = 10 if self.fast else None
        sessions = Session.fetch_all(only_real=True, include_timeframes=True, max=max_sessions)

        pbar = tqdm(total=self.count, desc="Creating Datasets{}".format(" (fast)" if self.fast else ""))

        for wl in self.window_lengths:
            self.value[str(wl)] = {}

            wl_datasets = list(
                Session.full_dataset_gen(window_length=wl, count=self.cv_splits, sessions=sessions))

            for sample_trim in self.sample_trims:
                self.value[str(wl)][str(sample_trim)] = {}
                st_datasets = [ds.trim_none_seconds(sample_trim, return_copy=True) for ds in wl_datasets]

                for ds_type in self.dataset_types:
                    dt_datasets = []

                    for st_ds in st_datasets:
                        ds = st_ds.copy()
                        ds = ds.reduced_dataset(ds_type)
                        ds = ds.normalize()
                        ds.shuffle()

                        dt_datasets.append(ds)
                        pbar.update(1)

                    self.value[str(wl)][str(sample_trim)][str(ds_type)] = dt_datasets
        pbar.close()

    def datasets(self, dataset_type, window_length, sample_trim):
        return self.value[str(window_length)][str(sample_trim)][str(dataset_type)]
