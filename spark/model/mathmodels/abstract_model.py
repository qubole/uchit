from spark.discretizer.normalizer import ConfigNormalizer
from spark.model.mathmodels.default_config_estimator import DefaultConfigEstimator


class AbstractModel:

    def __init__(self, config_set, training_data, num_cores, total_memory):
        self.config_set = config_set
        self.training_data = training_data
        self.normalizer = ConfigNormalizer(config_set)
        self.configEstimator = DefaultConfigEstimator(num_cores, total_memory)
        self.num_cores = num_cores
        self.total_memory = total_memory

    def get_pruned_config(self, normalized_lhs_configs):
        final_config = []
        denorm_configs = []
        for normalized_config in normalized_lhs_configs:
            denorm_configs.append(self.normalizer.denormalize_config(normalized_config))
        for denorm_config in denorm_configs:
            if not self.prune(denorm_config):
                final_config.append(self.normalizer.normalize_config(denorm_config))
        return final_config

    def get_optimal_executor_mem_per_core(self):
        return self.configEstimator.get_min_executor_mem()

    def get_max_driver_mem(self):
        """
        :rtype: int
        """
        return self.configEstimator.get_max_driver_memory()

    def get_min_driver_mem(self):
        return 1024  # 1 GB

    def prune(self, denorm_config):
        """

        :param denorm_config: Denormalized Configuration which will be checked if it can be pruned from config space
        :return: True, if config can be pruned otherwise False
        """
        params = self.config_set.get_params()
        executor_mem = executors_per_core = driver_mem = broadcast_threshold = -404  # NOT FOUND
        for index in range(len(params)):
            if params[index].get_name() == 'spark.executor.memory':
                executor_mem = denorm_config[index]
            if params[index].get_name() == 'spark.executor.cores':
                executors_per_core = denorm_config[index]
            if params[index].get_name() == 'spark.driver.memory':
                driver_mem = denorm_config[index]
            if params[index].get_name() == 'spark.sql.autoBroadcastJoinThreshold':
                broadcast_threshold = denorm_config[index]

        if not executors_per_core == -404 and \
                not (self.num_cores % executors_per_core == 0):
            return True

        upper_executor_memory = self.get_optimal_executor_mem_per_core() * 1.02
        lower_executor_memory = self.get_optimal_executor_mem_per_core() * 0.98
        if not executor_mem == -404 and \
                not executors_per_core == -404 and \
                not (lower_executor_memory <= (executor_mem / executors_per_core) <= upper_executor_memory):
            return True

        if not driver_mem == -404 and \
                not (self.get_min_driver_mem() <= driver_mem <= self.get_max_driver_mem()):
            return True

        if not broadcast_threshold == -404 and \
                broadcast_threshold > self.configEstimator.get_max_broadcast_of_driver_mem(executor_mem):
            return True

        return False

    def get_best_config(self):
        config = {}
        training_data_point = self.training_data.get_training_data()[0]
        param_dict = training_data_point["configs"].get_params_dict()
        params = training_data_point["configs"].get_all_params()
        driver_param = [x for x in params if x.get_name() == 'spark.driver.memory'][0]
        for param, value in param_dict.items():
            name = param.get_name()
            if name == 'spark.executor.memory':
                config[name] = self.configEstimator.get_max_executor_mem()
            elif name == 'spark.executor.cores':
                config[name] = self.num_cores
            elif name == 'spark.sql.autoBroadcastJoinThreshold':
                config[name] = self.configEstimator\
                    .get_max_broadcast_of_driver_mem(param_dict[driver_param])
            elif name == 'spark.sql.statistics.fallBackToHdfs':
                config[name] = 1
            else:
                config[name] = value
        return config
