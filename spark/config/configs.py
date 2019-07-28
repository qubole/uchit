from spark.config.config_values import IntRangeValues, PointValue
from spark.config.configs_element import ConfigsElement


class Configs:
    """Individual configs being auto-tuned for a particular spark job"""

    def __init__(self, num_cores,  # int
                 total_memory,  # int represents MB
                 ):
        self.num_cores = num_cores
        self.total_memory = total_memory
        self.config = dict()
        self.configs_elements = ConfigsElement()

    def add_config(self, name, config_value):
        self.config[name] = config_value
        return self

    def add_configs_element(self, names, values, out):
        assert len(names) == len(values)
        temp = []
        for (name, value) in zip(names, values):
            config_value = self.get_config_dict().get(name)
            if config_value is None:
                raise Exception("Conf not found")

            temp.append((name, PointValue(value,
                                          config_value.get_min_for_normalization(),
                                          config_value.get_max_for_normalization())))
        self.configs_elements.add_config(temp, out)

    # WARNING: returns shallow copy which is fine as keys are String and values are ConfigValues
    # and both are immutable. If we change that anytime then this should return deepcopy
    def get_config_dict(self):
        return self.config.copy()

    def _get_min_executor_mem(self):
        return int(self.total_memory / self.num_cores)

    def _get_max_executor_mem(self):
        return int(self.total_memory * 0.9)

    def _get_max_driver_memory(self):
        return int(self.total_memory / self.num_cores)

    def _get_max_broadcast_threshold(self):
        return int(self._get_max_driver_memory() * 0.2)

    def get_config_names(self):
        return self.config.keys()


class DefaultConfigs(Configs):
    __instance = None

    def __init__(self, num_cores,  # int
                 total_memory):
        """ Virtually private constructor. """
        if DefaultConfigs.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Configs.__init__(self, num_cores, total_memory)
            self.add_default_config()
            DefaultConfigs.__instance = self

    def add_default_config(self):
        self.add_config('spark.sql.shuffle.partitions', IntRangeValues(10, 2000, 50, 10, 2000))\
            .add_config('spark.executor.memory',
                        IntRangeValues(self._get_min_executor_mem(),  # min executor memory
                                       self._get_max_executor_mem(),  # max executor memory
                                       512, self._get_min_executor_mem(),
                                       self._get_max_executor_mem())) \
            .add_config('spark.driver.memory',
                        IntRangeValues(256,
                                       self._get_max_driver_memory(),
                                       256, 256, self._get_max_driver_memory())) \
            .add_config('spark.executor.cores', IntRangeValues(2, self.num_cores, 1, 1, self.num_cores))

    @staticmethod
    def get_instance(num_cores,  # int
                     total_memory):
        if DefaultConfigs.__instance is None:
            DefaultConfigs(num_cores, total_memory)

        return DefaultConfigs.__instance
