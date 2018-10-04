# <--- PROJECT CONFIG --->

DEV_MODE = True
SHOW_PROGRESS_BAR = True

# <--- NUMBER CONSTANTS --->

CV_SPLITS = 3
TEST_SIZE = 0.2

# <--- FORMAT --->

TIME_FORMAT = '%H:%M:%S'
DATE_FORMAT = '%d.%m.%Y %H:%M'
DATE_FILE_FORMAT = '%d%m%H%M%S'

# <--- API COMMUNICATION --->
import os

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
    _base = "http://127.0.0.1:8000" if DEV_MODE else ""

    timeframes = "/".join([_base, "time-frames"])
    sessions = "/".join([_base, "sessions"])


class Path:
    _project_root = os.path.dirname(os.path.realpath(__file__))

    data_dir = "{}/data".format(_project_root)

    session_cache = "{}/data/session_cache".format(_project_root)
    exp_logs = "{}/data/exp_logs".format(_project_root)
