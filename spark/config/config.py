import copy
from collections import OrderedDict


class Config:
    """Individual configs being auto-tuned for a particular spark job"""

    def __init__(self):
        self.params = OrderedDict()

    def add_param(self, param, value):
        self.params[param] = value
        return self

    def get_params_dict(self):
        return copy.deepcopy(self.params)

    def get_all_params(self):
        return copy.deepcopy(self.params).keys()

    def get_all_param_names(self):
        return list(map(lambda x: x.get_name(), self.params.keys()))

    def get_all_param_values(self):
        return self.params.values()
