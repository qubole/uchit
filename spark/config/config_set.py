from domain import IntRangeDomain
from parameter import Parameter
from spark.model.mathmodels.default_config_estimator import DefaultConfigEstimator


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

    """
    ToDo: Fix the min/max driver/executor memory initialization. Currently, the max executor memory suggested by the \
    calculation can be greater than the actual max executor memory (max memory of the node, for the case when only one executor per node.

    To fix the calculation would it be a good idea to take the number of worker nodes as user input :thinking:??
    """


    def __init__(self, num_cores,  # int
                 total_memory):
        ConfigSet.__init__(self)
        if num_cores < 1 or total_memory < 1024*1.1:
            raise ValueError("Number of cores less than 1 and total memory less than 1126 (1.1*1024) is unsupported")
        self.config_estimator = DefaultConfigEstimator(num_cores, total_memory)
        self.num_cores = num_cores
        self.total_memory = total_memory
        self.__init_default_params()

    def _get_min_executor_mem(self):
        return self.config_estimator.get_min_executor_mem()

    def _get_max_executor_mem(self):
        return self.config_estimator.get_max_executor_mem()

    def _get_max_driver_memory(self):
        # This will give a very small range for possible values of driver memory :thinking:
        return max(1024, self.config_estimator.get_min_executor_mem())

    def _get_max_broadcast_threshold(self):
        return self.config_estimator.get_max_broadcast_threshold()

    def __init_default_params(self):
        executor_step = self._get_min_executor_mem()
        while (executor_step > 512):
            executor_step = executor_step / 2

        self.add_param(Parameter('spark.sql.shuffle.partitions', IntRangeDomain(10, 2000, 10)))\
            .add_param(Parameter('spark.executor.memory',
                                 IntRangeDomain(self._get_min_executor_mem(),  # min executor memory
                                                self._get_max_executor_mem(),  # max executor memory
                                                executor_step)))\
            .add_param(Parameter('spark.driver.memory',
                                 IntRangeDomain(1024, self._get_max_driver_memory(), 256))) \
            .add_param(Parameter('spark.executor.cores',
                                 IntRangeDomain(1, self.num_cores, 1))) \
            .add_param(Parameter('spark.sql.autoBroadcastJoinThreshold',
                                 IntRangeDomain(10, self._get_max_broadcast_threshold(), 5))) \
            .add_param(Parameter('spark.sql.statistics.fallBackToHdfs',
                                 IntRangeDomain(0, 1, 1)))  # TODO Create Boolean Domain Type
