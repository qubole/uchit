from spark.config.config_values import ConfigType, IntRangeValues, FloatRangeValues
from spark.error import UnsupportedError
from enum import Enum


class DefaultPerConfigs:
    """Individual configs being auto-tuned for a particular spark job"""

    def __init__(self, format,  # string
                 num_cores,  # int
                 total_memory,  # int represents MB
                 ):
        if format.lower not in self.__options.keys:
            UnsupportedError("Unsupported format " + format +
                             ". Please specify one of these: " +
                             ",".join(self.__options.keys()))
        self.__options[format.lower]()
        self.num_cores = num_cores
        self.total_memory = total_memory
        self.default_config = self.__default_config()

    def __default_config(self):
        config = dict
        config['spark.sql.shuffle.partitions'] = \
            IntRangeValues(10, 2000, 50)
        config['spark.executor.memory'] = \
            IntRangeValues(self.__get_min_executor_mem(),  # min executor memory
                           self.__get_max_executor_mem(),  # max executor memory
                           self.__get_min_executor_mem())  # steps to increase or decrease memory
        config['spark.driver.memory'] = \
            IntRangeValues(1024,  # 1024 MB is minimum driver memory
                           self.__get_min_executor_mem())  # min executor memory in MB
        config['spark.executor.cores'] = IntRangeValues(2, self.num_cores, 1)
        config['spark.sql.autoBroadcastJoinThreshold'] = \
            IntRangeValues(10, self.__get_max_broadcast_threshold(), 20)
        return config

    def __get_min_executor_mem(self):
        int(self.total_memory / self.num_cores)

    def __get_max_executor_mem(self):
        int(self.total_memory * 0.9)

    def __get_max_driver_memory(self):
        int(self.total_memory / self.num_cores)

    def __get_max_broadcast_threshold(self):
        int(self.__get_max_driver_memory() * 0.2)

    def get_default_config(self):
        return self.default_config

    def get_config_names(self):
        return self.default_config.keys()
