import enum
import itertools

import numpy as np

from config import BASE_LABEL_MAP


class DSLabel(enum.Enum):
    NONE = "N"
    ANY_EVENT = "E"
    ARM = "A"
    FOOT = "F"
    LEFT = "L"
    RIGHT = "R"
    ARM_LEFT = "LA"
    ARM_RIGHT = "RA"
    FOOT_LEFT = "LF"
    FOOT_RIGHT = "RF"

    def __str__(self):
        return self.value

    @property
    def base_integer_list(self):
        if self == self.ANY_EVENT:
            return [1, 2, 3, 4]
        else:
            res = []
            for key, val in BASE_LABEL_MAP.items():
                if self.value == key or self.value in list(key):
                    res.append(val)

            return res

    def has_overlap(self, other):
        result = False
        for x in self.base_integer_list:
            for y in other.base_integer_list:
                if x == y:
                    return True
        return result


class DSType:
    separator = "-"

    def __init__(self, labels):
        labels = [DSLabel(l) for l in labels]

        # Do it this way to ensure deterministic order
        self.labels = [DSLabel(l) for l in DSLabel if l in labels]

        for a, b in itertools.combinations(self.labels, 2):
            if a.has_overlap(b):
                raise Exception("DSType has overlap between classes")

    def __str__(self):
        return self.separator.join([l.value for l in self.labels])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    @classmethod
    def variants(cls):
        return [
            cls(["N", "E"]),
            cls(["A", "F"]),
            cls(["N", "A", "F"]),
            cls(["L", "R"]),
            cls(["LA", "RA"]),
            cls(["LF", "RF"]),
            cls(["N", "LA", "RA", "LF", "RF"])
        ]

    @classmethod
    def from_string(cls, string):
        tokens = string.split(cls.separator)
        return cls([DSLabel(t) for t in tokens])

    @property
    def base_integer_list(self):
        res = []
        for label in self.labels:
            res += label.base_integer_list
        return res

    @property
    def label_map(self):
        res = dict()

        i = 0
        for label in DSLabel:
            if label in self.labels:
                res[label.value] = i
                i += 1

        return res

    @property
    def n_classes(self):
        return len(self.labels)

    def base_to_new_y(self, y_val):
        for label in self.labels:
            if y_val in label.base_integer_list:
                return self.label_map[label.value]

        return None

    def adjust_y(self, y):
        res = []
        for y_val in y:
            y_new = self.base_to_new_y(y_val)
            if y_new is not None:
                res.append(y_new)

        return np.array(res)
