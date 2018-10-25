import numpy as np

from config import BASE_LABEL_MAP
from utils.enums import DSType
from utils.prints import Print


class DataSet:
    split_ratio = (0.6, 0.8)

    def __init__(self, X, y, is_child=False, label_map=BASE_LABEL_MAP):
        self.X = np.array(X)
        self.y = np.array(y)
        self.is_child = is_child
        self.label_map = label_map

    @classmethod
    def empty(cls):
        return cls([], [])

    @classmethod
    def from_dict(cls, d):
        return cls(d["X"], d["y"], is_child=d["is_child"], label_map=d["label_map"])

    def split(self, include_val=True):
        if self.is_child:
            raise Print.build_except("Tried to split a child dataset.", self)

        if include_val:
            indices = [int(fraction * self.length) for fraction in self.split_ratio]
        else:
            indices = [int(self.split_ratio[1] * self.length)]

        X_parts = np.split(self.X, indices)
        y_parts = np.split(self.y, indices)

        res = []

        for X_part, Y_part in zip(X_parts, y_parts):
            res.append(DataSet(X_part, Y_part, is_child=True, label_map=self.label_map))

        return res

    def split_by_labels(self):
        datasets = {}

        labels = np.unique(self.y)

        for label in labels:
            p = [self.y == label]
            y_l = self.y[p]
            X_l = self.X[p]
            datasets[label] = (DataSet(X_l, y_l))

    def trim(self, new_length):
        if self.length > new_length:
            self.X = self.X[:new_length]
            self.y = self.y[:new_length]

    def shuffle(self):
        p = np.random.permutation(self.length)
        self.X = self.X[p]
        self.y = self.y[p]

    @property
    def length(self):
        return len(self.y)

    @property
    def one_hot(self):
        n_categories = len(set(self.y))
        Yoh = np.empty([self.length, n_categories])
        for i in range(self.length):
            row = np.zeros([n_categories])
            index = int(self.y[i])
            row[index] = 1
            Yoh[i, :] = row

        return Yoh

    def distribution(self):
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))

    def reduced_dataset(self, ds_type):
        if isinstance(ds_type, str):
            ds_type = DSType.from_string(ds_type)

        p = [y_val in ds_type.base_integer_list for y_val in self.y]

        ds = DataSet(self.X[p], self.y[p])
        ds.y = ds_type.adjust_y(ds.y)
        ds.label_map = ds_type.label_map

        return ds

    def normalize(self):
        labels, count = np.unique(self.y, return_counts=True)
        min_count = min(count)

        norm_ds = DataSet.empty()
        for label in labels:
            p = [self.y == label]
            y_l = self.y[p]
            X_l = self.X[p]

            ds_l = DataSet(X_l, y_l)
            ds_l.shuffle()
            ds_l.trim(min_count)

            norm_ds += ds_l

        norm_ds.shuffle()
        return norm_ds

    def label_str(self, label_id):
        for (key, val) in self.label_map.items():
            if val == label_id:
                return key

    def __add__(self, other):
        if len(self.X) != len(self.y) or len(other.X) != len(other.y):
            raise Exception("Mismatch in X and y length in DataSet")

        if self.label_map != other.label_map:
            raise Exception("Cannot add DataSets with different label_maps")

        if self.length == 0: return other
        if other.length == 0: return self

        X = np.concatenate((self.X, other.X))
        y = np.concatenate((self.y, other.y))
        is_child = (self.is_child or other.is_child)

        return DataSet(X, y, is_child, label_map=self.label_map)
