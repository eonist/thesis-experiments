from tqdm import tqdm


class ProgressBar:
    __instance = None

    @staticmethod
    def get_instance():
        if ProgressBar.__instance is None:
            ProgressBar()
        return ProgressBar.__instance

    def __init__(self, **kwargs):
        if ProgressBar.__instance is not None:
            raise Exception("This class is a singleton!")

        super().__init__(**kwargs)
        ProgressBar.__instance = self

        self.progress = 0
        self.total = 1
        self.bar = tqdm()
        self.ids = []

    @classmethod
    def include(cls, id, iterable=None, total=None):
        pb = cls.get_instance()

        if id not in pb.ids:
            pb.ids.append(id)
            if total is not None:
                pb.total *= total
            elif iterable is not None:
                pb.total *= len(iterable)

            pb.bar.total = pb.total

        return pb

    def increment(self):
        self.progress += 1
        self.bar.update(1)

    def reset(self):
        self.progress = 0
