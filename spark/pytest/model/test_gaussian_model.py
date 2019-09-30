import copy
import pytest

from spark.config.config_set import UniversalConfigSet
from spark.model.gaussian_model import GaussianModel
from spark.model.training_data import TrainingData
from spark.config.config import Config


@pytest.fixture(scope="class", autouse=True)
def gaussian_model():
    training_data = TrainingData()
    config_set = UniversalConfigSet(10, 1024 * 10)
    yield GaussianModel(config_set, training_data)


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
                                            "Missing config: spark.executor.cores, spark.sql.statistics.fallBackToHdfs, spark.sql.autoBroadcastJoinThreshold, Extra config: invalid_config"):
            assert model.add_sample_to_train_data(training_sample, 12)

    def test_gaussian_model_training(self, gaussian_model):
        model = copy.deepcopy(gaussian_model)
        training_sample = {
            "spark.executor.memory": 1024 * 5,
            "spark.sql.shuffle.partitions": 100,
            "spark.executor.cores": 4,
            "spark.driver.memory": 1024,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        model.add_sample_to_train_data(training_sample, 12)
        model.add_sample_to_train_data(training_sample, 12)
        # Ensure the order in training data is same as the one in config_set
        assert model.training_data.get_training_data()[0]["configs"].get_all_param_names() == \
               ["spark.sql.shuffle.partitions", "spark.executor.memory", "spark.driver.memory", "spark.executor.cores",
                "spark.sql.autoBroadcastJoinThreshold", "spark.sql.statistics.fallBackToHdfs"]

        assert model.training_data.get_training_data()[0]["configs"].get_all_param_values() == \
               [100, 1024 * 5, 1024, 4, 10, 0]

        model.train()

        # Ensure that normalised values are between 0 and 1
        for normalized_config_value in model.training_inp_normalized[0]:
            assert 0 <= normalized_config_value <= 1

        # Ensure order is maintained
        assert (model.training_inp_normalized[0] == model.training_inp_normalized[1]).all()

    def test_gaussian_model_predict(self):
        training_data = TrainingData()
        config_set = UniversalConfigSet(4, 26544)
        model = GaussianModel(config_set, training_data)
        training_sample_1 = {
            "spark.executor.memory": 11945,
            "spark.sql.shuffle.partitions": 200,
            "spark.executor.cores": 2,
            "spark.driver.memory": 1024 * 4,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        training_sample_2 = {
            "spark.executor.memory": 5972,
            "spark.sql.shuffle.partitions": 300,
            "spark.executor.cores": 1,
            "spark.driver.memory": 1024 * 2,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        training_sample_3 = {
            "spark.executor.memory": 11945,
            "spark.sql.shuffle.partitions": 460,
            "spark.executor.cores": 2,
            "spark.driver.memory": 1024 * 4,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        training_sample_4 = {
            "spark.executor.memory": 10068,
            "spark.sql.shuffle.partitions": 1660,
            "spark.executor.cores": 1,
            "spark.driver.memory": 1024,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        model.add_sample_to_train_data(training_sample_1, 131)
        model.add_sample_to_train_data(training_sample_2, 143)
        model.add_sample_to_train_data(training_sample_3, 155)
        model.add_sample_to_train_data(training_sample_4, 343)
        model.train()
        config = Config()
        params = config_set.get_params()
        for param in params:
            if param.get_name() == 'spark.executor.memory':
                config.add_param(param, 10068)
            elif param.get_name() == 'spark.sql.shuffle.partitions':
                config.add_param(param, 1660)
            elif param.get_name() == 'spark.executor.cores':
                config.add_param(param, 1)
            elif param.get_name() == 'spark.driver.memory':
                config.add_param(param, 1024)
            elif param.get_name() == 'spark.sql.autoBroadcastJoinThreshold':
                config.add_param(param, 10)
            elif param.get_name() == 'spark.sql.statistics.fallBackToHdfs':
                config.add_param(param, 0)
        low, high = model.predict(model.normalizer.normalize_config(config.get_all_param_values()))
        assert low > (343 - 1)
        assert high < (343 + 1)

    def test_gaussian_model_get_best_config_value(self):
        training_data = TrainingData()
        config_set = UniversalConfigSet(4, 26544)
        model = GaussianModel(config_set, training_data)
        training_sample_1 = {
            "spark.executor.memory": 11945,
            "spark.sql.shuffle.partitions": 200,
            "spark.executor.cores": 2,
            "spark.driver.memory": 1024 * 4,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        training_sample_2 = {
            "spark.executor.memory": 5972,
            "spark.sql.shuffle.partitions": 300,
            "spark.executor.cores": 1,
            "spark.driver.memory": 1024 * 2,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        training_sample_3 = {
            "spark.executor.memory": 11945,
            "spark.sql.shuffle.partitions": 460,
            "spark.executor.cores": 2,
            "spark.driver.memory": 1024 * 4,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        training_sample_4 = {
            "spark.executor.memory": 10068,
            "spark.sql.shuffle.partitions": 1660,
            "spark.executor.cores": 1,
            "spark.driver.memory": 1024,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        model.add_sample_to_train_data(training_sample_1, 131)
        model.add_sample_to_train_data(training_sample_2, 143)
        model.add_sample_to_train_data(training_sample_3, 155)
        model.add_sample_to_train_data(training_sample_4, 343)
        model.train()
        best_config = model.get_best_config()
        assert True

