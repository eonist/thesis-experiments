import json
import os

import numpy as np
import requests

from config import URL, label_map, WINDOW_LENGTH, Path
from models.dataset import DataSet


class Session:
    def __init__(self):
        self.id = None
        self.person_id = None
        self.ch_names = None
        self.created = None
        self.n_timeframes = None
        self.labels = None
        self.is_real_data = None

    def __str__(self):
        return "{id} - {created}".format(id=self.id, created=self.created)

    def cache_fp(self, only_fn=False):
        fn = "session-{}-timeframes".format(self.id)
        if only_fn:
            return fn
        else:
            return "/".join([Path.data_dir, fn])

    @classmethod
    def from_api(cls, id):
        url = "/".join([URL.sessions, str(id)])
        json_data = requests.get(url).json()
        return cls.from_json(json_data)

    @classmethod
    def from_json(cls, json):
        obj = cls()
        obj.id = json["id"]
        obj.person_id = json["person"]
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
        fns = os.listdir(Path.data_dir)

        return self.cache_fp(only_fn=True) in fns

    @classmethod
    def fetch_all(cls, only_real=False):
        params = {"real": True} if only_real else {}

        r = requests.get(URL.sessions, params=params)
        json_data = r.json()

        return [cls.from_json(d) for d in json_data]

    def get_timeframes(self):
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

            label_name = data["label"]["name"].strip()
            m[i, n_channels + 1] = label_map[label_name]

        return m

    def window_gen(self, window_length):
        m = self.get_timeframes()
        i = np.random.randint(0, window_length / 2)
        n_timeframes = np.shape(m)[0]

        while i < n_timeframes - window_length:
            window = m[i:i + window_length, :]
            # if window[0] == 0 and window[0] <= window[-1]:
            yield window

            i += window_length + np.random.randint(-window_length / 2, window_length / 2)

    def dataset(self, window_length=WINDOW_LENGTH):
        windows = list(self.window_gen(window_length))
        n_samples = len(windows)
        n_channels = len(self.ch_names)

        X = np.empty([n_samples, n_channels, window_length])
        Y = np.empty([n_samples], dtype=np.int8)

        for i, window in enumerate(windows):
            X[i] = window[:, 0:n_channels].T

            unique_labels = set(window[:, -1])
            Y[i] = int(max(unique_labels))

        return DataSet(X, Y)

    @classmethod
    def combined_dataset(cls, ids):
        dataset = cls.from_api(ids[0]).dataset()
        for id in ids[1:]:
            dataset = dataset + cls.from_api(id).dataset()

        return dataset

    @classmethod
    def full_dataset(cls):
        sessions = cls.fetch_all(only_real=True)
        ids = [s.id for s in sessions]
        return cls.combined_dataset(ids)
