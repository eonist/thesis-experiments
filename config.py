# <--- API COMMUNICATION --->
import os

dev_mode = True

# None is 0
# Arm events are 1 and 2, foot events are 3 and 4
# Right events are odd, left events are even
label_map = {
    "None": 0,
    "arm/right": 1,
    "arm/left": 2,
    "foot/right": 3,
    "foot/left": 4
}

inv_label_map = {v: k for k, v in label_map.items()}

WINDOW_LENGTH = 250


class URL:
    _base = "http://127.0.0.1:8000" if dev_mode else ""

    timeframes = "/".join([_base, "time-frames"])
    sessions = "/".join([_base, "sessions"])


class Path:
    _project_root = os.path.dirname(os.path.realpath(__file__))

    data_dir = "{}/data".format(_project_root)
