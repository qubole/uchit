from spark.config.config import Config
from spark.config.config_set import UniversalConfigSet
from spark.model.gaussian_model import GaussianModel
from spark.model.mathmodels.abstract_model import AbstractModel
from spark.model.training_data import TrainingData


class Combiner:
    def __init__(self, num_cores, total_memory, training_data=TrainingData()):
        self.num_cores = num_cores
        self.total_memory = total_memory
        self.config_set = UniversalConfigSet(num_cores, total_memory)
        self.training_data = training_data
        if self.training_data.size() > 0:
            self.ml_model = GaussianModel(self.config_set, training_data)
            self.math_model = AbstractModel(self.config_set, training_data, num_cores, total_memory)

    def add_training_data(self, training_sample, output):
        self.training_data.add_training_data(self._get_training_config(training_sample), output)
        self.ml_model = GaussianModel(self.config_set, self.training_data)
        self.math_model = AbstractModel(self.config_set, self.training_data, self.num_cores, self.total_memory)

    def _get_training_config(self, training_sample):
        conf_names_params_mapping = {}
        for param in self.config_set.get_params():
            conf_names_params_mapping[param.get_name()] = param
        training_config = Config()
        for config_name, config_value in training_sample.items():
            training_config.add_param(conf_names_params_mapping[config_name], training_sample[config_name])
        return training_config

    def get_best_config(self):
        if self.training_data.size() == 0:
            raise ValueError("Training Data Not Provided")
        if self.training_data.size() == 1:
            return self.math_model.get_best_config()
        self.ml_model.train()
        sampled_configs = self.ml_model.get_sampled_configs()
        pruned_configs = self.math_model.get_pruned_config(sampled_configs)
        return self.ml_model.get_best_config_for_config_space(pruned_configs)

