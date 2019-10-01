from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy


class Domain:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_possible_values(self):
        pass

    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def get_num_elements(self):
        pass

    @abstractmethod
    def get_max(self):
        pass

    @abstractmethod
    def get_min(self):
        pass


class DomainType(Enum):
    INT = 1
    FLOAT = 2


class RangeDomain(Domain):
    __metaclass__ = ABCMeta

    def __init__(self, min_val, max_val, step):
        self._values = list(numpy.arange(min_val, max_val, step))

        if not self._values:
            raise ValueError("Illegal Arguments - min value: "
                             + str(min_val) + " max value: " + str(max_val)
                             + " step: " + str(step))
        self._min = self._values[0]
        self._max = self._values[len(self._values) - 1]
        self._step = step

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def get_step(self):
        return self._step

    def get_possible_values(self):
        return self._values

    def get_num_elements(self):
        return len(self.get_possible_values())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            res = self.get_min() == other.get_min()\
                   and self.get_max() == other.get_max()\
                   and self.get_step() == other.get_step()
            return res
        return NotImplemented

    def __hash__(self):
        return hash(self._min) ^ hash(self._max) ^ hash(self._step) ^ hash(self.get_type())


class IntRangeDomain(RangeDomain):
    # both min_val and mx_val are inclusive
    def __init__(self, min_val, max_val, step):
        RangeDomain.__init__(self, int(min_val), int(max_val) + 1, int(step))

    def get_type(self):
        return DomainType.INT


class FloatRangeDomain(RangeDomain):
    def __init__(self, min_val, max_val, step):
        RangeDomain.__init__(self, float(min_val), float(max_val), float(step))

    def get_type(self):
        return DomainType.FLOAT
