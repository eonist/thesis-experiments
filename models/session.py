import json
import os
import random

import numpy as np
import requests
from tqdm import tqdm

from config import URL, Path, WINDOW_LENGTH
from models.dataset import Dataset
from utils.plotting import plot_matrix
from utils.prints import Print
from utils.utils import api_label_to_val


class Session:
    def __init__(self):
        self.id = None
        self.person_id = None
        self.ch_names = None
        self.created = None
        self.n_timeframes = None
        self.labels = None
        self.is_real_data = None
        self.timeframes = None

    def __str__(self):
        return "{id} - {created}".format(id=self.id, created=self.created)

    def cache_fp(self, only_fn=False):
        fn = "session-{}-timeframes".format(self.id)
        if only_fn:
            return fn
        else:
            return "/".join([Path.session_cache, fn])

    @classmethod
    def from_api(cls, id):
        url = "/".join([URL.sessions, str(id)])
        json_data = requests.get(url).json()
        return cls.from_json(json_data)

    @classmethod
    def from_json(cls, json):
        obj = cls()
        obj.id = int(json["id"])
        obj.person_id = int(json["person"])

        if obj.person_id == 112:  # person 112 is also Olav
            obj.person_id = 6

        obj.ch_names = json["ch_names"]
        obj.created = json["created"]
        obj.n_timeframes = json["timeframe_count"]
        obj.labels = json["labels"]
        obj.is_real_data = json["is_real_data"]
        return obj

    def save_cache(self, json_data):
        fp = self.cache_fp()

        with open(fp, "w") as outfile:
            json.dump(json_data, outfile)

    def is_cached(self):
        fns = os.listdir(Path.session_cache)
        return self.cache_fp(only_fn=True) in fns

    @classmethod
    def fetch_all(cls, only_real=True, include_timeframes=False, max=None):
        params = {"real": True} if only_real else {}

        r = requests.get(URL.sessions, params=params)
        json_data = r.json()

        sessions = [cls.from_json(d) for d in json_data]

        if max is not None:
            sessions = np.random.choice(sessions, max)

        if include_timeframes:
            [s.fetch_timeframes() for s in tqdm(sessions, desc="Fetching Sessions")]

        return sessions

    def fetch_timeframes(self):
        if self.is_cached():
            with open(self.cache_fp(), "r") as infile:
                json_data = json.load(infile)
        else:
            r = requests.get(URL.timeframes, params={"session": self.id})
            json_data = r.json()
            self.save_cache(json_data)

        if len(json_data) == 0:
            return np.array([])

        n_channels = len(self.ch_names)

        m = np.zeros([len(json_data), n_channels + 2])

        time_zero = json_data[0]["timestamp"]

        for i, data in enumerate(json_data):
            m[i, 0:n_channels] = data["sensor_data"]
            m[i, n_channels] = data["timestamp"] - time_zero
            m[i, n_channels + 1] = api_label_to_val(data["label"]["name"])

        self.timeframes = m
        return m

    def window_gen(self, window_length=WINDOW_LENGTH, allowed_labels=None):
        if self.timeframes is None:
            self.fetch_timeframes()

        i = random.randint(0, window_length / 2)
        n_timeframes = np.shape(self.timeframes)[0]

        if i > n_timeframes - window_length:
            return

        while i < n_timeframes - window_length:
            window = self.timeframes[i:i + window_length, :]

            if allowed_labels is not None:
                label = int(max(window[:, -1]))
                if label in allowed_labels:
                    yield window
            else:
                yield window

            # TODO: Try without overlapping windows
            i += window_length + random.randint(-window_length / 2, window_length / 2)

    def dataset(self, windows, remove_seconds=0):
        if len(windows) == 0:
            return Dataset.empty()

        n_samples = len(windows)
        n_channels = len(self.ch_names)
        window_length = np.shape(windows)[1]

        X = np.empty([n_samples, n_channels, window_length])
        y = np.empty([n_samples], dtype=np.int8)

        for i, window in enumerate(windows):
            X[i] = window[:, 0:n_channels].T
            y[i] = int(max(window[:, -1]))

        if remove_seconds > 0:
            change_points = []
            action_labels = []
            for i in range(1, len(y), 1):
                if y[i] != y[i - 1]:
                    change_points.append(i)
                    action_labels.append(np.max(y[i - 1:i + 1]))

            remove_distance = (250 * remove_seconds) / window_length
            keep_indices = []
            for i in range(len(y)):
                label = y[i]
                if label == 0:
                    viable = True
                    for point in change_points:
                        if np.abs(i - point) <= remove_distance:
                            viable = False
                    if viable:
                        keep_indices.append(i)
                else:
                    keep_indices.append(i)

            X = X[keep_indices]
            y = y[keep_indices]

        return Dataset(X, y, self.person_id, self.id)

    @classmethod
    def combined_dataset(cls, ids, window_length):
        dataset = Dataset.empty()
        for id in ids:
            session = cls.from_api(id)
            windows = list(session.window_gen(window_length=window_length))
            dataset = dataset + session.dataset(windows)

        return dataset

    @classmethod
    def full_dataset(cls, window_length):
        sessions = cls.fetch_all(only_real=True)
        ids = [s.id for s in sessions]
        return cls.combined_dataset(ids, window_length=window_length)

    @classmethod
    def full_dataset_gen(cls, window_length, count=1, sessions=None):

        if sessions is None:
            Print.info("Fetching sessions")
            sessions = Session.fetch_all(only_real=True, include_timeframes=True)

        for _ in range(count):
            dataset = Dataset.empty()
            for session in sessions:
                windows = list(session.window_gen(window_length=window_length))
                dataset = dataset + session.dataset(windows=windows)

            yield dataset


if __name__ == '__main__':
    Print.start("Starting")
    sessions = Session.fetch_all()

    session = random.choice(sessions)
    n_channels = len(session.ch_names)

    session.fetch_timeframes()

    X = session.timeframes[:, :n_channels]
    y = session.timeframes[:, n_channels + 1]

    Print.data(np.mean(y))

    X_pow = X ** 2

    res = np.zeros([len(y), n_channels + 1])
    res[:, :n_channels] = X_pow
    res[:, n_channels] = y

    res = res / res.max(axis=0)
    res = res.T

    Print.data(np.mean(res[-1, :]))

    Print.data(np.shape(res))
    plot_matrix(res)
