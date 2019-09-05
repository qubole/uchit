import copy
from abc import abstractmethod


class Model(object):

    def __init__(self):
        self._is_updated = False

    def is_updated(self):
        return self._is_updated

    @staticmethod
    def get_model_mapping():
        from spark.model.gaussian_model import GaussianModel
        model_mapping = {
            "gaussian": copy.deepcopy(GaussianModel)
        }
        return model_mapping

    @classmethod
    def valid_models(cls):
        return cls.get_model_mapping().keys()

    @classmethod
    def get_model(cls, config_set, model_name, training_data):
        if model_name.lower() not in cls.valid_models():
            raise Exception("Invalid model name. Valid models are: %s" % ", ".join(cls.valid_models()))

        return copy.deepcopy(cls.get_model_mapping()[model_name.lower()](config_set, training_data))

    @abstractmethod
    def add_sample_to_train_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_best_config(self):
        pass

