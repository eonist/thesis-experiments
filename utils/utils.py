import collections
import datetime
import os
import pathlib
import random
import string

import numpy as np

from config import TIME_FORMAT, DATE_FILE_FORMAT, DATE_FORMAT


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

    if index < len(keys) - 1:
        next_data = data[keys[index]]
        return dict_set(next_data, multilevel_key, value, index=index + 1)
    else:
        data[keys[index]] = value


def dict_get(data, multilevel_key, index=0):
    keys = multilevel_key.split("__")
    if index < len(keys):
        return dict_get(data[keys[index]], multilevel_key, index=index + 1)
    else:
        return data
