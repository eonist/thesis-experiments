import numpy as np

from config import label_map
from utils.prints import Print


class DataSet:
    split_ratio = (0.6, 0.8)

    def __init__(self, X, y, is_child=False):
        self.X = np.array(X)
        self.y = np.array(y)
        self.is_child = is_child
        self.label_map = label_map

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
            res.append(DataSet(X_part, Y_part, is_child=True))

        # order: ds_train, ds_val, ds_test
        return res

    def shuffle(self):
        p = np.random.permutation(len(self.X))
        self.X = self.X[p]
        self.y = self.y[p]

    @property
    def length(self):
        return len(self.y)

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

    def binary_dataset(self, type):
        if type == "none-rest":
            ds = self
            ds.y = np.array([0 if val == 0 else 1 for val in ds.y])
            ds.label_map = {"none": 0, "event": 1}
        elif type == "arm-foot":
            ds = self.reduced_dataset([1, 2, 3, 4])
            ds.y = np.array([0 if val in [1, 2] else 1 for val in ds.y])
            ds.label_map = {"arm": 0, "foot": 1}
        elif type == "right-left":
            ds = self.reduced_dataset([1, 2, 3, 4])
            ds.y = np.array([0 if val % 2 == 0 else 1 for val in ds.y])
            ds.label_map = {"left": 0, "right": 1}
        elif type == "right-left-arm":
            ds = self.reduced_dataset([1, 2])
            ds.y = np.array([0 if val % 2 == 0 else 1 for val in ds.y])
            ds.label_map = {"arm/left": 0, "arm/right": 1}
        elif type == "right-left-foot":
            ds = self.reduced_dataset([3, 4])
            ds.y = np.array([0 if val % 2 == 0 else 1 for val in ds.y])
            ds.label_map = {"foot/left": 0, "foot/right": 1}
        else:
            ds = None
            Print.failure("Binary label type not found: {}".format(type))

        return ds

    def reduced_dataset(self, labels):
        y = np.array([self.y[i] for i in range(len(self.y)) if self.y[i] in labels])
        X = np.array([self.X[i] for i in range(len(self.y)) if self.y[i] in labels])

        return DataSet(X, y)

    def normalize(self):
        min_label_count = min(list(self.distribution().values()))
        print(min_label_count)

        for label in np.unique(self.y):
            while self.distribution()[label] > min_label_count:
                # TODO: Finish this
                pass

    def label_str(self, label_id):
        for (key, val) in self.label_map.items():
            if val == label_id:
                return key

    def __add__(self, other):
        X = np.concatenate((self.X, other.X))
        Y = np.concatenate((self.y, other.y))
        is_child = (self.is_child or other.is_child)

        return DataSet(X, Y, is_child)
