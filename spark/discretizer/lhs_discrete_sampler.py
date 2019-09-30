import numpy
from ..error import UnsupportedError
from pyDOE import *


class LhsDiscreteSampler:
    # normalized_configs list of list - list of possible config values normalized between 0 and 1.
    # sample_size - total number of sample points required. this should be less than total_points
    def __init__(self,
                 normalized_configs,
                 sample_size=None):
        self._normalized_configs = normalized_configs
        if sample_size is None:
            max_sample_size = max(list(map(lambda x: len(x), normalized_configs)))
            self._sample_size = max_sample_size
        else:
            self._sample_size = sample_size
        self._config_size_array = map(len, self._normalized_configs)

    def get_samples(self, seed_value):
        numpy.random.seed(seed_value)
        total_num_configs = len(self._normalized_configs)
        lhd = lhs(total_num_configs, samples=self._sample_size, criterion='center')
        return_config = []
        for config_set_index in range(len(lhd)):
            config_set = []
            for config_index in range(len(lhd[config_set_index])):
                # Since config value lengths are unequal we sampled using maximum points
                # To find the actual point from sampled points p,
                # we will find index = int(round(p * actual_size/max_size))
                actual_index = int(round(lhd[config_set_index][config_index] * self._config_size_array[config_index]))
                # actual_index = int(round(len(lhd[config_index])
                #                          * (self._config_size_array[config_index]/float(max_points))))
                # ToDO: Fix the below hack to ensure that actual_index is never equal to
                #  self._config_size_array[config_index] to avoid list index out of range
                if actual_index == self._config_size_array[config_index]:
                    actual_index = actual_index - 1
                config_set.append(self._normalized_configs[config_index][actual_index])
            return_config.append(config_set)
        return return_config
