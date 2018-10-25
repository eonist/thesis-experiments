import enum
import itertools

import numpy as np

from config import BASE_LABEL_MAP


class DSLabel(enum.Enum):
    NONE = "none"
    ANY_EVENT = "event"
    ARM = "arm"
    FOOT = "foot"
    LEFT = "left"
    RIGHT = "right"
    ARM_LEFT = "arm/left"
    ARM_RIGHT = "arm/right"
    FOOT_LEFT = "foot/left"
    FOOT_RIGHT = "foot/right"

    def __str__(self):
        return self.value

    @property
    def base_integer_list(self):
        if self == self.ANY_EVENT:
            return [1, 2, 3, 4]
        else:
            res = []
            for key, val in BASE_LABEL_MAP.items():
                if self.value == key or self.value in key.split("/"):
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
    def __init__(self, labels):
        labels = [DSLabel(l) for l in labels]

        # Do it this way to ensure deterministic order
        self.labels = [DSLabel(l) for l in DSLabel if l in labels]

        for a, b in itertools.combinations(self.labels, 2):
            if a.has_overlap(b):
                raise Exception("DSType has overlap between classes")

    def __str__(self):
        return "_".join([l.value for l in self.labels])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    @classmethod
    def variants(cls):
        return [
            cls(["none", "event"]),
            cls(["arm", "foot"]),
            cls(["none", "arm", "foot"]),
            cls(["left", "right"]),
            cls(["arm/left", "arm/right"]),
            cls(["foot/left", "foot/right"]),
            cls(["none", "arm/left", "arm/right", "foot/left", "foot/right"])
        ]

    @classmethod
    def from_string(cls, string):
        tokens = string.split("_")
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


if __name__ == '__main__':
    ds_type = DSType(["none", "arm", "foot"])

    y = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    new_y = ds_type.adjust_y(y)

    print(ds_type.label_map)
    print(new_y)
