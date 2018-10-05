import enum

from tqdm import tqdm

from config import SHOW_PROGRESS_BAR


class State(enum.Enum):
    READY = 0
    RUNNING = 1
    FINISHED = 2


class ProgressBar:
    __instance = None

    @staticmethod
    def get_instance():
        if ProgressBar.__instance is None:
            ProgressBar()
        return ProgressBar.__instance

    def __init__(self):
        if ProgressBar.__instance is not None:
            raise Exception("This class is a singleton!")

        ProgressBar.__instance = self

        self.progress = 0
        self.total = 1
        self.tqdm = None
        self.pb_ids = []
        self.state = State.READY

    @classmethod
    def include(cls, pb_id, iterable=None, total=None):
        pb = cls.get_instance()

        if not SHOW_PROGRESS_BAR:
            return pb

        if pb.state == State.FINISHED:
            pb.reset()

        if pb.state == State.READY:
            if pb_id not in pb.pb_ids:
                pb.pb_ids.append(pb_id)
                if total is not None:
                    pb.total *= total
                elif iterable is not None:
                    pb.total *= len(iterable)

        return pb

    def increment(self, pb_id=None, count=1):
        if not SHOW_PROGRESS_BAR:
            return

        if pb_id is not None and pb_id == self.pb_ids[-1]:
            if self.state == State.READY:
                self.tqdm = tqdm()
                self.tqdm.total = self.total

                self.state = State.RUNNING

            if self.state == State.RUNNING:
                self.progress += 1
                self.tqdm.update(count)

                if self.progress >= self.total:
                    self.state = State.FINISHED

    def reset(self):
        self.state = State.READY
        self.progress = 0
        self.total = 1
        self.tqdm = None
        self.pb_ids = []

    def close(self):
        self.reset()
        self.__instance = None
