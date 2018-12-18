import collections
import copy
import datetime
import os
import pathlib
import random
import string

import numpy as np

from config import TIME_FORMAT, DATE_FILE_FORMAT, DATE_FORMAT, DECIMALS, API_TO_LABEL_MAP, BASE_LABEL_MAP
from utils.enums import DSLabel


def func_name():
    import traceback
    return traceback.extract_stack(None, 2)[0][2]


def rand_string(length, rng=None):
    if rng is not None:
        all_chars = list(string.ascii_uppercase + string.digits)
        rng.shuffle(all_chars)
        return ''.join(all_chars[:7])
    else:
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def timestamp_str(timestamp):
    str_repr = timestamp.strftime(TIME_FORMAT)
    return str_repr


def datestamp_str(timestamp, file=False):
    str_repr = timestamp.strftime(DATE_FILE_FORMAT if file else DATE_FORMAT)
    return str_repr


def timestamp_obj(timestamp_str):
    return datetime.datetime.strptime(timestamp_str, TIME_FORMAT)


def create_path_if_not_existing(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def np_append(arr, value):
    if len(arr) == 0:
        return np.array(value)
    else:
        return np.vstack([arr, value])


def max_elem_length(arr):
    temp_max_length = 0
    if isinstance(arr[0], (list, np.ndarray)):
        for elem in arr:
            res = max_elem_length(elem)
            if res > temp_max_length:
                temp_max_length = res
    else:
        for elem in arr:
            length = len(str(elem))
            if length > temp_max_length:
                temp_max_length = length

    return temp_max_length


def format_array(arr):
    np_array = np.asarray(arr)

    dim = len(np.shape(np_array))

    if dim == 1:
        elems = ['\"{}\"'.format(e) if isinstance(e, (str, DSLabel)) else str(e) for e in np_array]
        return "[{}]".format(",".join(elems))
    elif dim == 2:
        rows_string = ",\n".join([format_array(row) for row in np_array])
        return "[\n{}\n]".format(rows_string)


def notify(title="Notification", text=""):
    os.system("""osascript -e 'display notification "{}" with title "{}"'""".format(text, title))


# <--- DICT OPERATIONS --->

def flatten_dict(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_set(data, multilevel_key, value, index=0):
    """ Use double underscores (__) in key to access deeper in the dict """

    keys = multilevel_key.split("__")
    keys = [int(k) if is_number(k) else k for k in keys]

    if index < len(keys) - 1:
        next_data = data[keys[index]]
        return dict_set(next_data, multilevel_key, value, index=index + 1)
    else:
        data[keys[index]] = value


def dict_get(data, multilevel_key, index=0):
    if multilevel_key == "":
        return data

    keys = multilevel_key.split("__")
    keys = [int(k) if is_number(k) else k for k in keys]

    if index < len(keys):
        return dict_get(data[keys[index]], multilevel_key, index=index + 1)
    else:
        return data


def inverse_dict(d):
    return {value: key for key, value in d.items()}


def is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def find_number_paths(data, key_path=None):
    number_paths = []

    if isinstance(data, dict):
        for key, val in data.items():
            new_key_path = key_path + "__{}".format(key) if key_path is not None else key
            number_paths += find_number_paths(val, new_key_path)
    elif isinstance(data, list):
        for i in range(len(data)):
            new_key_path = key_path + "__{}".format(i) if key_path is not None else str(i)
            number_paths += find_number_paths(data[i], new_key_path)
    elif is_number(data):
        number_paths.append(key_path)

    return number_paths


def avg_dict(dict_list, decimals=None):
    """ Assumes all the dicts in the list are structurally identical """

    if len(dict_list) <= 0:
        return {}

    d = dict_list[0]
    number_paths = find_number_paths(d)

    res = copy.deepcopy(d)

    for ml_key in number_paths:
        avg = np.mean([dict_get(d, ml_key) for d in dict_list])
        if decimals is not None:
            avg = np.round(avg, decimals)

        dict_set(res, ml_key, avg)

    return res


def round_dict(d, decimals=DECIMALS):
    number_paths = find_number_paths(d)

    res = copy.deepcopy(d)

    for key in number_paths:
        rounded_num = np.round(dict_get(res, multilevel_key=key), decimals=decimals)
        dict_set(res, key, rounded_num)
    return res


def api_label_to_val(api_label):
    api_label = api_label.strip().lower()
    base_label = API_TO_LABEL_MAP[api_label]
    label_val = BASE_LABEL_MAP[base_label]

    return label_val


def as_list(elem, length=1):
    is_list = isinstance(elem, list) or isinstance(elem, np.ndarray)
    return elem if is_list else [elem] * length


def points_around(points, distance):
    if distance == 0:
        return np.array([])

    if isinstance(points, int):
        return np.arange(points - distance, points + distance + 1, 1)
    elif isinstance(points, (list, np.ndarray)):
        return np.concatenate([points_around(p, distance) for p in points])


def find_nearest(array, value, return_index=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    if return_index:
        return idx
    else:
        return array[idx]


if __name__ == '__main__':
    dict1 = {'arm': {'precision': [1.1323234, 6.1493832, 10], 'recall': 2},
             'foot': {'precision': 1, 'recall': 1}}

    dict2 = {'arm': {'precision': [3, 14, 100], 'recall': 3},
             'foot': {'precision': 2, 'recall': 0}}

    res = round_dict(dict1)
    print(res)
