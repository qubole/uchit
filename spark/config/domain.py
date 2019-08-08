from abc import ABCMeta, abstractmethod
from enum import Enum


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

    def __init__(self):
        self._values = None

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def get_step(self):
        return self._step

    def get_possible_values(self):
        if self._values is None:
            self._values = range(self._min + self._step, self._max, self._step)
        return self._values

    def get_num_elements(self):
        self.get_step()


class IntRangeDomain(RangeDomain):
    def __init__(self, min_val, max_val, step):
        RangeDomain.__init__(self, int(min_val), int(max_val), int(step))

    def get_type(self):
        return DomainType.INT


class FloatRangeDomain(RangeDomain):
    def __init__(self, min_val, max_val, step):
        RangeDomain.__init__(self, float(min_val), float(max_val), float(step))

    def get_type(self):
        return DomainType.FLOAT
