import collections
import copy
import datetime
import os
import pathlib
import random
import string
from queue import Empty

import numpy as np

from config import TIME_FORMAT, DATE_FILE_FORMAT, DATE_FORMAT, DECIMALS
from utils.prints import Print


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


# <--- MULTIPROCESSING --->

def worker(i, working_queue, output_q, func):
    while True:
        try:
            job = working_queue.get_nowait()
            Print.progress("Worker {} is doing a job".format(i))
            output_q.put(func(job))
        except Empty:
            break

    return


if __name__ == '__main__':
    dict1 = {'arm': {'precision': [1.1323234, 6.1493832, 10], 'recall': 2},
             'foot': {'precision': 1, 'recall': 1}}

    dict2 = {'arm': {'precision': [3, 14, 100], 'recall': 3},
             'foot': {'precision': 2, 'recall': 0}}

    res = round_dict(dict1)
    print(res)
