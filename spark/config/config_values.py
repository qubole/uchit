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

    @abstractmethod
    def get_num_elements(self):
        pass

    @abstractmethod
    def get_max_for_normalization(self):
        pass

    @abstractmethod
    def get_min_for_normalization(self):
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

    def get_num_elements(self):
        self.get_step()

    def get_max_for_normalization(self):
        return self.__max_norm

    def get_min_for_normalization(self):
        return self.__min_norm


class IntRangeValues(RangeValues):
    def __init__(self, min_val, max_val, step,
                 min_norm, max_norm # these are min and max possible values for normalization
                 ):
        RangeValues.__init__(self)
        self.__min = int(min_val)
        self.__max = int(max_val)
        self.__step = int(step)
        self.__min_norm = int(min_norm)
        self.__max_norm = int(max_norm)

    def get_values(self):
        if self.__value is not None:
            self.__value = range(self.__min, self.__max, self.__step)
        return self.__value

    def get_type(self):
        return ConfigType.INT


class FloatRangeValues(RangeValues):
    def __init__(self, min_val, max_val, step,
                 min_norm, max_norm  # these are min and max possible values for normalization
                 ):
        RangeValues.__init__(self)
        self.__min = float(min_val)
        self.__max = float(max_val)
        self.__step = float(step)
        self.__min_norm = min_val
        self.__max_norm = max_norm

    def get_type(self):
        return ConfigType.FLOAT


class PointValue(ConfigValues):
    __metaclass__ = ABCMeta

    def __init__(self, value, min_norm, max_norm):
        self.__values = value
        self.__min_norm = min_norm
        self.__max_norm = max_norm

    def get_values(self):
        return self.__values

    def get_num_elements(self):
        len(self.__values)

    def get_max_for_normalization(self):
        return self.__max_norm

    def get_min_for_normalization(self):
        return self.__min_norm


class IntPointValue(PointValue):
    def __init__(self, value, min_norm, max_norm):
        PointValue.__init__(self, map(int, value), min_norm, max_norm)

    def get_type(self):
        return ConfigType.INT


class FloatPointValue(PointValue):
    def __init__(self, value, min_norm, max_norm):
        PointValue.__init__(self, map(float, value), min_norm, max_norm)

    def get_type(self):
        return ConfigType.FLOAT


