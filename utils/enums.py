import enum


class DSType(enum.Enum):
    NONE_REST = "none_rest"
    ARM_FOOT = "arm_foot"
    LEFT_RIGHT = "left_right"
    ARM_LEFT_RIGHT = "arm/left_arm/right"
    FOOT_LEFT_RIGHT = "foot/left_foot/right"

    def __str__(self):
        return self.value

    @property
    def label_map(self):
        if self == self.NONE_REST:
            return {"none": 0, "event": 1}
        elif self == self.ARM_FOOT:
            return {"arm": 0, "foot": 1}
        elif self == self.LEFT_RIGHT:
            return {"left": 0, "right": 1}
        elif self == self.ARM_LEFT_RIGHT:
            return {"arm/left": 0, "arm/right": 1}
        elif self == self.FOOT_LEFT_RIGHT:
            return {"foot/left": 0, "foot/right": 1}

    @property
    def labels(self):
        return self.value.split("_")
