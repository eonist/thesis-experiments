import numpy as np

from config import BASE_LABEL_MAP, SAMPLING_RATE
from utils.enums import DSType
from utils.prints import Print
from utils.utils import as_list, points_around, find_nearest


class Dataset:
    split_ratio = (0.6, 0.8)

    def __init__(self, X, y, u, s, is_child=False, label_map=BASE_LABEL_MAP, original_order=True):
        self.X = np.array(X)
        self.y = np.array(y)
        self.u = np.array(as_list(u, length=self.length))  # User id list
        self.s = np.array(as_list(s, length=self.length))  # Session id list

        self.is_child = is_child
        self.label_map = label_map
        self.original_order = original_order

    @classmethod
    def empty(cls):
        return cls([], [], [], [])

    @classmethod
    def from_dict(cls, d):
        return cls(d["X"], d["y"], d["u"], d["s"], is_child=d["is_child"], label_map=d["label_map"])

    def child_from_mask(self, mask):
        child = self.copy(mask=mask)
        child.is_child = True
        return child

    def apply_mask(self, p):
        self.X = self.X[p]
        self.y = self.y[p]
        self.u = self.u[p]
        self.s = self.s[p]
        self.original_order = False

    def split_random(self, include_val=False):
        if self.is_child:
            raise Print.build_except("Tried to split a child dataset.", self)

        split_ratio = np.asarray(self.split_ratio if include_val else [self.split_ratio[1]])
        splits = (self.length * split_ratio).astype(int)

        p = np.random.permutation(self.length)
        p_parts = np.split(p, splits)

        return [self.child_from_mask(p) for p in p_parts]

    def split_by_user(self, user_id=None):
        if user_id is None:
            user_ids = np.unique(self.u)
            user_id = np.random.choice(user_ids)

        user_p = self.u == user_id
        rest_p = self.u != user_id

        return [self.child_from_mask(p) for p in [rest_p, user_p]]

    def split_by_session(self):
        unique, counts = np.unique(self.s, return_counts=True)

        test_set_length_goal = (1 - self.split_ratio[1]) * self.length

        split_index = None
        while split_index is None:
            p = np.random.permutation(len(unique))
            unique = unique[p]
            counts = counts[p]

            test_set_length = 0

            for i in range(len(counts)):
                test_set_length += counts[i]
                if np.abs(test_set_length - test_set_length_goal) <= self.length * 0.01:
                    split_index = i + 1
                    break

        test_p = np.isin(self.s, unique[:split_index])
        train_p = np.isin(self.s, unique[split_index:])

        return [self.child_from_mask(p) for p in [train_p, test_p]]

    def trim(self, new_length, shuffle=False):
        if shuffle:
            self.shuffle()

        if self.length > new_length:
            p = self.range < new_length
            self.apply_mask(p)

    def shuffle(self):
        p = np.random.permutation(self.length)
        self.apply_mask(p)
        self.original_order = False

    @property
    def length(self):
        return len(self.y)

    def distribution(self):
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))

    def reduced_dataset(self, ds_type):
        if isinstance(ds_type, str):
            ds_type = DSType.from_string(ds_type)

        p = np.isin(self.y, ds_type.base_integer_list)

        ds = self.copy(mask=p)
        ds.y = ds_type.adjust_y(ds.y)
        ds.label_map = ds_type.label_map

        return ds

    def user_datasets(self):
        users = np.unique(self.u)

        for user in users:
            p = self.u == user
            yield self.copy(mask=p)

    def normalize(self):
        labels, count = np.unique(self.y, return_counts=True)
        min_count = min(count)

        norm_ds = Dataset.empty()
        norm_ds.label_map = self.label_map
        for label in labels:
            p = self.y == label

            ds_l = self.copy(mask=p)
            ds_l.trim(min_count, shuffle=True)

            norm_ds += ds_l

        norm_ds.shuffle()
        return norm_ds

    def copy(self, mask=None):
        if mask is not None:
            copy = self.copy()
            copy.apply_mask(mask)
            return copy

        return Dataset(self.X, self.y, self.u, self.s, is_child=self.is_child, label_map=self.label_map,
                       original_order=self.original_order)

    def label_str(self, label_id):
        for (key, val) in self.label_map.items():
            if val == label_id:
                return key

    def trim_none_seconds(self, sample_trim, return_copy=False):
        if not self.original_order:
            Print.warning("Skipped trim_none_seconds since dataset was not in original order")
            if return_copy:
                return self.copy()
            return

        if isinstance(sample_trim, str):
            sample_trim = [float(t) for t in sample_trim.split(";")]

        relabel_seconds = sample_trim[0]
        remove_seconds = np.max(sample_trim)

        if remove_seconds == 0:
            if return_copy:
                return self.copy()
            return

        change_points = []
        action_labels = []
        for i in range(1, self.length, 1):
            if self.y[i] != self.y[i - 1]:
                change_points.append(i)
                action_labels.append(np.max(self.y[i - 1:i + 1]))

        relabel_dist = int((SAMPLING_RATE * relabel_seconds) / self.window_length)
        remove_dist = int((SAMPLING_RATE * remove_seconds) / self.window_length)

        y_none_mask = self.y == 0

        relabel_mask = np.isin(self.range, points_around(change_points, relabel_dist))
        relabel_mask = np.logical_and(relabel_mask, y_none_mask)

        remove_mask = np.isin(self.range, points_around(change_points, remove_dist))
        remove_mask = np.logical_and(remove_mask, y_none_mask)
        remove_mask = np.logical_xor(remove_mask, relabel_mask)

        for i in self.range:
            if relabel_mask[i]:
                self.y[i] = action_labels[find_nearest(change_points, i, return_index=True)]

        keep_mask = np.invert(remove_mask)

        if return_copy:
            return self.copy(keep_mask)
        else:
            self.apply_mask(keep_mask)

    def __add__(self, other):
        if len(self.X) != len(self.y) or len(other.X) != len(other.y):
            raise Exception("Mismatch in X and y length in Dataset")

        if self.label_map != other.label_map:
            raise Exception("Cannot add Datasets with different label_maps")

        if self.length == 0: return other
        if other.length == 0: return self

        X = np.concatenate((self.X, other.X))
        y = np.concatenate((self.y, other.y))
        u = np.concatenate((self.u, other.u))
        s = np.concatenate((self.s, other.s))

        is_child = (self.is_child or other.is_child)

        return Dataset(X, y, u, s, is_child, label_map=self.label_map)

    @property
    def window_length(self):
        return np.shape(self.X)[2]

    @property
    def range(self):
        return np.arange(self.length)

    @property
    def one_hot(self):
        n_categories = len(set(self.y))
        y_oh = np.empty([self.length, n_categories])
        for i in self.range:
            row = np.zeros([n_categories])
            index = int(self.y[i])
            row[index] = 1
            y_oh[i, :] = row

        return y_oh
