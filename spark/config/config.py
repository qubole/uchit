import copy


class Config:
    """Individual configs being auto-tuned for a particular spark job"""

    def __init__(self, num_cores,  # int
                 total_memory,  # int represents MB
                 ):
        self.num_cores = num_cores
        self.total_memory = total_memory
        self.params = dict()

    def add_param(self, param, value):
        self.params[param] = value
        return self

    def get_params_dict(self):
        return copy.deepcopy(self.params)

    def _get_min_executor_mem(self):
        return int(self.total_memory / self.num_cores)

    def _get_max_executor_mem(self):
        return int(self.total_memory * 0.9)

    def _get_max_driver_memory(self):
        return int(self.total_memory / self.num_cores)

    def _get_max_broadcast_threshold(self):
        return int(self._get_max_driver_memory() * 0.2)

    def get_all_params(self):
        return copy.deepcopy(self.params).keys()

    def get_all_param_names(self):
        return list(map(lambda x: x.get_name(), self.params.keys()))

    def get_all_param_values(self):
        return self.params.values()
