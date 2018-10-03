# -*- coding: UTF-8 -*-

import inspect
import json
import sys
import time
import traceback
# -- SYMBOLS --
from functools import wraps

import numpy as np
import pandas as pd
from pygments import highlight
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.lexers.data import JsonLexer

START = "✅"
MIDDLE = "🌠"
END = "🔵"


# |-- PRIVATE HELPER FUNCTIONS --|

def wrapped_print(array, TYPE):
    array.insert(0, TYPE + " " + START + " Start")
    array.append(TYPE + " " + END + " End")

    print("")
    for item in array:
        if type(item) is dict:
            print_json_color(item)
        else:
            print(item)
    print("")


def varName(data):
    function_call = inspect.stack()[2][4][0]
    input_name = function_call[function_call.find("(") + 1:function_call.rfind(")")]
    return '{0}'.format(input_name.replace('.', '. '))


def print_json_color(obj):
    print(highlight(json.dumps(obj, indent=4, sort_keys=True), JsonLexer(), Terminal256Formatter()))


def flag_print(emoji, text):
    print(" {} --- {}".format(emoji, str(text)))


# |-- UTILITY CLASS --|

class Print:
    def __init__(self, text):
        self.point(text)

    @staticmethod
    def ex(e):
        TYPE = "❗"
        array = [type(e), e, MIDDLE, traceback.format_exc()]
        wrapped_print(array, TYPE)

    @staticmethod
    def data(data, one_line=True):
        TYPE = "📳"
        if one_line:
            flag_print(TYPE, "{}  --- {}".format(varName(data), data))
        else:
            array = [varName(data), MIDDLE, data]
            wrapped_print(array, TYPE)

    @staticmethod
    def pandas(data):
        TYPE = "📳"

        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        else:
            data = pd.DataFrame(data)

        array = [varName(data), MIDDLE, data]
        wrapped_print(array, TYPE)

    @staticmethod
    def build_except(message, data):
        return Exception({'message': '{0}: {1}'.format(message, str(type(data))), 'data': str(data)})

    @staticmethod
    def json(data):
        Print.data(data, one_line=False)

    # -- FLAG PRINTS --
    @staticmethod
    def point(text):
        flag_print("⛳", text)

    @staticmethod
    def warning(text):
        flag_print("🚸", text)

    @staticmethod
    def failure(text):
        flag_print("❌", text)

    @staticmethod
    def success(text):
        flag_print("✅", text)

    @staticmethod
    def info(text):
        flag_print("🔘", text)

    @staticmethod
    def start(text):
        flag_print("✨", text)

    @staticmethod
    def progress(text):
        flag_print("🚙", text)

    @staticmethod
    def stop(text):
        flag_print("🛑", text)


# [-- DECORATOR METHODS --]

def print_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        time_start = time.time()

        the_class = get_class_that_defined_method(func)

        if the_class:
            func_name = '{0}.{1}()'.format(the_class.__name__, func.__name__)
        else:
            func_name = '{0}()'.format(func.__name__)

        res = None
        print("")
        Print.start('Starting {0}'.format(func_name))
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            Print.failure('Failed to finish {0}'.format(func_name))
            Print.ex(e)
            sys.exit()
        else:
            time_end = time.time()
            print("")
            Print.progress('Finished {0} in {1}s'.format(func_name, round(time_end - time_start, 1)))
            return res

    return wrapper


def print_init(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__

        if len(args) > 1 and type(args[1]) is str:
            name = args[1]
        else:
            name = None

        if not class_name:
            class_name = '<Unknown class>'

        print("\n\n\n\n")
        if name:
            flag_print("🚀", "Creating {0} instance with name \"{1}\"".format(class_name, name))
        else:
            flag_print("🚀", "Creating {0} instance".format(class_name))

        try:
            res = func(*args, **kwargs)
        except Exception as e:
            Print.failure('Failed to create {0}'.format(class_name))
            Print.ex(e)
            sys.exit()
        else:
            return res

    return wrapper


def print_section(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        time_start = time.time()

        the_class = get_class_that_defined_method(func)

        if the_class:
            func_name = '{0}.{1}()'.format(the_class.__name__, func.__name__)
        else:
            func_name = '{0}()'.format(func.__name__)

        res = None
        print("\n\n")
        Print.progress("Starting {0}".format(func_name))

        try:
            res = func(*args, **kwargs)
        except Exception as e:
            Print.failure('Failed to finish {0}'.format(func_name))
            Print.ex(e)
            sys.exit()
        else:
            time_delta = time.time() - time_start
            time_str = "{}m ".format(int(np.floor(time_delta / 60))) if time_delta / 60 > 1 else ""
            time_str += "{}s".format(int(time_delta % 60))
            Print.success('Finished {0} in {1}'.format(func_name, time_str))
            return res

    return wrapper


def get_class_that_defined_method(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth), meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return None  # not required since None would have been implicitly returned anyway
