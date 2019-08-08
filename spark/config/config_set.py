from domain import IntRangeDomain
from parameter import Parameter


class ConfigSet:
    def __init__(self):
        self.param_list = list()

    def add_param(self, param):
        self.param_list.append(param)
        return self

    def get_params(self):
        self.param_list


# Set of all possible Parameter Values that can be used to tune a Job
# Name is derived from Universal Set in Set Theory which is defined
# as a set that contains all objects.
class UniversalConfigSet(ConfigSet):

    def __init__(self, num_cores,  # int
                 total_memory):
        ConfigSet.__init__(self)
        self.num_cores = num_cores
        self.total_memory = total_memory
        self.__init_default_params()

    def _get_min_executor_mem(self):
        return int(self.total_memory / self.num_cores)

    def _get_max_executor_mem(self):
        return int(self.total_memory * 0.9)

    def _get_max_driver_memory(self):
        return int(self.total_memory / self.num_cores)

    def _get_max_broadcast_threshold(self):
        return int(self._get_max_driver_memory() * 0.2)

    def __init_default_params(self):
        self.add_param(Parameter('spark.sql.shuffle.partitions', IntRangeDomain(10, 2000, 50, 10, 2000)))\
            .add_param(Parameter('spark.executor.memory',
                                 IntRangeDomain(self._get_min_executor_mem(),  # min executor memory
                                                self._get_max_executor_mem(),  # max executor memory
                                                512, self._get_min_executor_mem(),
                                                self._get_max_executor_mem())))\
            .add_param(Parameter('spark.driver.memory',
                                 IntRangeDomain(256, self._get_max_driver_memory(), 256, 256,
                                                self._get_max_driver_memory()))) \
            .add_param('spark.executor.cores', IntRangeDomain(2, self.num_cores, 1, 1, self.num_cores))