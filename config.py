# <--- PROJECT CONFIG --->

DEV_MODE = True
SHOW_PROGRESS_BAR = True

# <--- NUMBER CONSTANTS --->

CV_SPLITS = 1
TEST_SIZE = 0.2
DECIMALS = 3

# <--- FORMAT --->

TIME_FORMAT = '%H:%M:%S'
DATE_FORMAT = '%d.%m.%Y %H:%M'
# DATE_FILE_FORMAT = '%d%m%H%M%S'
DATE_FILE_FORMAT = '%y%m%d_%H%M%S'

# <--- API COMMUNICATION --->
import os

# None is 0
# Arm events are 1 and 2, foot events are 3 and 4
# Left events are odd, right events are even
BASE_LABEL_MAP = {
    "N": 0,
    "LA": 1,
    "RA": 2,
    "LF": 3,
    "RF": 4
}

API_TO_LABEL_MAP = {
    "none": "N",
    "arm/left": "LA",
    "arm/right": "RA",
    "foot/left": "LF",
    "foot/right": "RF"
}

WINDOW_LENGTH = 250
CH_NAMES = ["C3", "C4", "P3", "P4"]

SAMPLING_RATE = 250


class URL:
    _base = "http://127.0.0.1:8000" if DEV_MODE else ""

    timeframes = "/".join([_base, "time-frames"])
    sessions = "/".join([_base, "sessions"])


class Path:
    _project_root = os.path.dirname(os.path.realpath(__file__))

    data_dir = "{}/data".format(_project_root)

    session_cache = "{}/data/session_cache".format(_project_root)
    exp_logs = "{}/data/exp_logs".format(_project_root)
    plots = "{}/data/plots".format(_project_root)
    exp_set_queue = "{}/data/exp_set_queue".format(_project_root)

    classifiers = "{}/data/classifiers".format(_project_root)

    templates = "{}/data/templates".format(_project_root)
