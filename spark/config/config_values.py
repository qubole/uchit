from abc import ABCMeta, abstractmethod
from enum import Enum

from spark.discretizer.normalizer import ConfigNormalizer


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
        self._values = None

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def get_step(self):
        return self._step

    def get_values(self):
        if self._values is None:
            self._values = range(self._min + self._step, self._max, self._step)
        return self._values

    def get_num_elements(self):
        self.get_step()

    def get_max_for_normalization(self):
        return self._max_norm

    def get_min_for_normalization(self):
        return self._min_norm


class IntRangeValues(RangeValues):
    def __init__(self, min_val, max_val, step,
                 min_norm, max_norm  # these are min and max possible values for normalization
                 ):
        RangeValues.__init__(self)
        self._min = int(min_val)
        self._max = int(max_val)
        self._step = int(step)
        self._min_norm = int(min_norm)
        self._max_norm = int(max_norm)

    def get_type(self):
        return ConfigType.INT


class FloatRangeValues(RangeValues):
    def __init__(self, min_val, max_val, step,
                 min_norm, max_norm  # these are min and max possible values for normalization
                 ):
        RangeValues.__init__(self)
        self._min = float(min_val)
        self._max = float(max_val)
        self._step = float(step)
        self._min_norm = min_norm
        self._max_norm = max_norm

    def get_type(self):
        return ConfigType.FLOAT


class PointValue(ConfigValues):
    __metaclass__ = ABCMeta

    def __init__(self, value, min_norm, max_norm):
        self._values = value
        self._min_norm = min_norm
        self._max_norm = max_norm

    def get_values(self):
        return self._values

    def get_num_elements(self):
        return len(self._values)

    def get_max_for_normalization(self):
        return self._max_norm

    def get_min_for_normalization(self):
        return self._min_norm

    def get_normalized_value(self):
        normalizer = ConfigNormalizer.norm_function(self._min_norm, self._max_norm)
        return normalizer(self._values)

    def get_type(self):
        return None

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


