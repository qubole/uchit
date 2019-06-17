from spark.config.config_values import IntRangeValues


class Configs:
    """Individual configs being auto-tuned for a particular spark job"""

    def __init__(self, num_cores,  # int
                 total_memory,  # int represents MB
                 ):
        self.num_cores = num_cores
        self.total_memory = total_memory
        self.config = dict()

    def add_config(self, name, config_value):
        self.config[name] = config_value
        return self

    # WARNING: returns shallow copy which is fine as keys are String and values are ConfigValues
    # and both are immutable. If we change that anytime then this should return deepcopy
    def get_config_dict(self):
        return self.config.copy()

    def __get_min_executor_mem(self):
        int(self.total_memory / self.num_cores)

    def __get_max_executor_mem(self):
        int(self.total_memory * 0.9)

    def __get_max_driver_memory(self):
        int(self.total_memory / self.num_cores)

    def __get_max_broadcast_threshold(self):
        int(self.__get_max_driver_memory() * 0.2)

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
        self.add_config('spark.sql.shuffle.partitions', IntRangeValues(10, 2000, 50))\
            .add_config('spark.executor.memory',
                        IntRangeValues(self.__get_min_executor_mem(),  # min executor memory
                                       self.__get_max_executor_mem(),  # max executor memory
                                       self.__get_min_executor_mem()))\
            .add_config('spark.driver.memory',
                        IntRangeValues(1024,  # 1024 MB is minimum driver memory
                                       self.__get_min_executor_mem(), 512))\
            .add_config('spark.executor.cores', IntRangeValues(2, self.num_cores, 1))\
            .add_config('spark.sql.autoBroadcastJoinThreshold',
                        IntRangeValues(10, self.__get_max_broadcast_threshold(), 20))

    @staticmethod
    def get_instance(num_cores,  # int
                     total_memory):
        if DefaultConfigs.__instance is None:
            DefaultConfigs(num_cores, total_memory)

        return DefaultConfigs.__instance
