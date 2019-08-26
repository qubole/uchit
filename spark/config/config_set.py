from domain import IntRangeDomain
from parameter import Parameter


class ConfigSet:
    def __init__(self):
        self.param_list = list()

    def add_param(self, param):
        self.param_list.append(param)
        return self

    def get_params(self):
        return self.param_list

    def get_size(self):
        return len(self.param_list)


# Set of all possible Parameter Values that can be used to tune a Job
# Name is derived from Universal Set in Set Theory which is defined
# as a set that contains all objects.
class UniversalConfigSet(ConfigSet):

    def __init__(self, num_cores,  # int
                 total_memory):
        ConfigSet.__init__(self)
        if num_cores < 2 or total_memory < 512*1.1:
            raise ValueError("Number of cores less than 2 and total memory less than 564 (1.1*512) is unsupported")
        self.num_cores = num_cores
        self.total_memory = total_memory
        self.__init_default_params()

    def _get_min_executor_mem(self):
        return int(self._get_max_executor_mem() / self.num_cores)

    def _get_max_executor_mem(self):
        return int(self.total_memory * 0.9)

    def _get_max_driver_memory(self):
        # This will give a very small range for possible values of driver memory :thinking:
        return int(self.total_memory / self.num_cores)

    def _get_max_broadcast_threshold(self):
        return int(self._get_max_driver_memory() * 0.2)

    def __init_default_params(self):
        self.add_param(Parameter('spark.sql.shuffle.partitions', IntRangeDomain(10, 2000, 50)))\
            .add_param(Parameter('spark.executor.memory',
                                 IntRangeDomain(self._get_min_executor_mem(),  # min executor memory
                                                self._get_max_executor_mem(),  # max executor memory
                                                512)))\
            .add_param(Parameter('spark.driver.memory',
                                 IntRangeDomain(256, self._get_max_driver_memory(), 256))) \
            .add_param(Parameter('spark.executor.cores',
                                 IntRangeDomain(1, self.num_cores, 1)))
