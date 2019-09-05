from spark.model.model import Model
from spark.model.training_data import TrainingData


class SparkTuner:

    def __init__(self, config_set, model_name="Gaussian", training_data=TrainingData()):
        self.model = SparkTuner.get_model(config_set, model_name, training_data)

    @staticmethod
    def get_model(config_set, model_name, training_data):
        return Model.get_model(config_set, model_name, training_data)

    def add_sample_to_train_data(self, training_sample, out):
        self.model.add_sample_to_train_data(training_sample, out)
        self.model._is_updated = False

    def get_next_best_config(self):
        if not self.model.is_updated():
            self.model.train()

        return self.model.get_best_config()

