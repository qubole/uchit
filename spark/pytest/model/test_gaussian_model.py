import copy
import pytest

from spark.config.config_set import UniversalConfigSet
from spark.model.gaussian_model import GaussianModel
from spark.model.training_data import TrainingData


@pytest.fixture(scope="class", autouse=True)
def gaussian_model():
    training_data = TrainingData()
    config_set = UniversalConfigSet(10, 1024 * 10)
    yield GaussianModel(training_data, config_set)


class TestGaussianModel:

    def test_gaussian_model_no_data(self, gaussian_model):
        model = copy.deepcopy(gaussian_model)
        with pytest.raises(Exception, match="No training data found"):
            assert model.train()

    def test_gaussian_model_invalid_training_data(self, gaussian_model):
        model = copy.deepcopy(gaussian_model)
        training_sample = {
            "spark.sql.shuffle.partitions": 100,
            "spark.executor.memory": 1024 * 5,
            "spark.driver.memory": 1024,
            "invalid_config": 123
        }
        with pytest.raises(Exception, match="Invalid config to be added as training data. "
                                            "Missing config: spark.executor.cores, Extra config: invalid_config"):
            assert model.add_sample_to_train_data(training_sample, 12)

    def test_gaussian_model_training(self, gaussian_model):
        model = copy.deepcopy(gaussian_model)
        training_sample = {
            "spark.executor.memory": 1024 * 5,
            "spark.sql.shuffle.partitions": 100,
            "spark.executor.cores": 4,
            "spark.driver.memory": 512
        }
        model.add_sample_to_train_data(training_sample, 12)
        model.add_sample_to_train_data(training_sample, 12)
        # Ensure the order in training data is same as the one in config_set
        assert model.training_data.get_training_data()[0]["configs"].get_all_param_names() == \
               ["spark.sql.shuffle.partitions", "spark.executor.memory", "spark.driver.memory", "spark.executor.cores"]

        assert model.training_data.get_training_data()[0]["configs"].get_all_param_values() == \
               [100, 1024 * 5, 512, 4]

        model.train()

        # Ensure that normalised values are between 0 and 1
        for normalized_config_value in model.training_inp_normalized[0]:
            assert 0 <= normalized_config_value <= 1

        # Ensure order is maintained
        assert model.training_inp_normalized[0] == model.training_inp_normalized[1]
