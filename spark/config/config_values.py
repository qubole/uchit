from abc import ABCMeta, abstractmethod
from enum import Enum


class ConfigValues:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_values(self):
        pass

    @abstractmethod
    def get_type(self):
        pass


class ConfigType(Enum):
    INT = 1
    FLOAT = 2


class RangeValues(ConfigValues):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__value = None

    def get_min(self):
        self.__min

    def get_max(self):
        self.__max

    def get_step(self):
        self.__step

    def get_values(self):
        if self.__value is not None:
            self.__value = range(self.__min, self.__max, self.__step)
        return self.__value


class IntRangeValues(RangeValues):
    def __init__(self, min_val, max_val, step):
        RangeValues.__init__(self)
        self.__min = int(min_val)
        self.__max = int(max_val)
        self.__step = int(step)


    def get_values(self):
        if self.__value is not None:
            self.__value = range(self.__min, self.__max, self.__step)
        return self.__value

    def get_type(self):
        return ConfigType.INT


class FloatRangeValues(RangeValues):
    def __init__(self, min_val, max_val, step):
        RangeValues.__init__(self)
        self.__min = float(min_val)
        self.__max = float(max_val)
        self.__step = float(step)

    def get_type(self):
        return ConfigType.FLOAT


class PointValue(ConfigValues):
    __metaclass__ = ABCMeta

    def __init__(self, value):
        self.__value = value

    def get_values(self):
        return self.__value


class IntPointValue(PointValue):
    def __init__(self, value):
        PointValue.__init__(self, int(value))

    def get_type(self):
        return ConfigType.INT


class FloatPointValue(PointValue):
    def __init__(self, value):
        PointValue.__init__(self, float(value))

    def get_type(self):
        return ConfigType.FLOAT
