from spark.config.config_set import UniversalConfigSet
from spark.model.gaussian_model import GaussianModel
from spark.model.mathmodels.abstract_model import AbstractModel


class Combiner:
    def __init__(self, num_cores, total_memory, training_data):
        self.config_set = UniversalConfigSet(num_cores, total_memory)
        self.ml_model = GaussianModel(self.config_set, training_data)
        self.math_model = AbstractModel(self.config_set, training_data, num_cores, total_memory)
        self.training_data = training_data

    def get_best_config(self):
        if self.training_data.size() == 0:
            raise ValueError("Training Data Not Provided")
        if self.training_data.size() == 1:
            return self.math_model.get_best_config()
        self.ml_model.train()
        sampled_configs = self.ml_model.get_sampled_configs()
        pruned_configs = self.math_model.get_pruned_config(sampled_configs)
        return self.ml_model.get_best_config_for_config_space(pruned_configs)

