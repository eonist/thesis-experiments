from tqdm import tqdm

from models.session import Session


class DatasetCollection:
    def __init__(self, dataset_types, window_lengths, cv_splits):
        if isinstance(dataset_types, str): dataset_types = [dataset_types]
        if isinstance(window_lengths, int): window_lengths = [window_lengths]

        self.dataset_types = dataset_types
        self.window_lengths = window_lengths
        self.cv_splits = cv_splits

        self.count = len(window_lengths) * len(dataset_types) * cv_splits
        self.value = None

        self._create_datasets()

    @classmethod
    def from_params(cls, params, cv_splits):
        return cls(params["dataset_type"], params["window_length"], cv_splits)

    def _create_datasets(self):
        self.value = {}
        pbar = tqdm(total=self.count, desc="Creating Datasets")
        sessions = Session.fetch_all(only_real=True, include_timeframes=True)

        for wl in self.window_lengths:
            self.value[str(wl)] = {}

            wl_datasets = list(
                Session.full_dataset_gen(window_length=wl, count=self.cv_splits, sessions=sessions))

            for ds_type in self.dataset_types:
                dt_datasets = []

                for wl_ds in wl_datasets:
                    ds = wl_ds.copy()
                    ds = ds.reduced_dataset(ds_type)
                    ds = ds.normalize()
                    ds.shuffle()

                    dt_datasets.append(ds)
                    pbar.update(1)

                self.value[str(wl)][str(ds_type)] = dt_datasets
        pbar.close()

    def datasets(self, dataset_type, window_length):
        return self.value[str(window_length)][str(dataset_type)]
