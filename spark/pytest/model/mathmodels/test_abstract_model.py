import pytest

from spark.config.config import Config
from spark.config.config_set import UniversalConfigSet
from spark.model.gaussian_model import GaussianModel
from spark.model.mathmodels.abstract_model import AbstractModel
from spark.model.training_data import TrainingData


def training_sample1():
    return ({
        "spark.executor.memory": 11945,
        "spark.sql.shuffle.partitions": 200,
        "spark.executor.cores": 2,
        "spark.driver.memory": 1024 * 4,
        "spark.sql.autoBroadcastJoinThreshold": 10,
        "spark.sql.statistics.fallBackToHdfs": 0
    }, 131)


def training_sample2():
    return ({
        "spark.executor.memory": 5972,
        "spark.sql.shuffle.partitions": 300,
        "spark.executor.cores": 1,
        "spark.driver.memory": 1024 * 2,
        "spark.sql.autoBroadcastJoinThreshold": 10,
        "spark.sql.statistics.fallBackToHdfs": 0
    }, 143)

def training_sample3():
    return ({
        "spark.executor.memory": 11945,
        "spark.sql.shuffle.partitions": 460,
        "spark.executor.cores": 2,
        "spark.driver.memory": 1024 * 4,
        "spark.sql.autoBroadcastJoinThreshold": 10,
        "spark.sql.statistics.fallBackToHdfs": 0
    }, 155)

def training_sample4():
    return ({
        "spark.executor.memory": 10068,
        "spark.sql.shuffle.partitions": 1660,
        "spark.executor.cores": 1,
        "spark.driver.memory": 1024,
        "spark.sql.autoBroadcastJoinThreshold": 10,
        "spark.sql.statistics.fallBackToHdfs": 0
    }, 343)


@pytest.fixture(scope="class", autouse=True)
def config_set():
    return UniversalConfigSet(4, 26544)


@pytest.fixture(scope="class", autouse=True,
                params=[[training_sample1(), training_sample2(), training_sample3(), training_sample4()]])
def training_samples(request):
    return request.param


@pytest.fixture(scope="class", autouse=True)
def training_data(config_set, training_samples):
    training_data = TrainingData()
    conf_names_params_mapping = {}
    for param in config_set.get_params():
        conf_names_params_mapping[param.get_name()] = param
    for training_sample in training_samples:
        training_config = Config()
        for config_name, config_value in training_sample[0].items():
            training_config.add_param(conf_names_params_mapping[config_name], training_sample[0][config_name])
        training_data.add_training_data(training_config, training_sample[1])
    return training_data


@pytest.fixture(scope="class", autouse=True)
def gaussian_model(training_data):
    config_set = UniversalConfigSet(10, 1024 * 10)
    model = GaussianModel(config_set, training_data)
    model.train()
    return model


@pytest.fixture(scope="class", autouse=True)
def abstract_model(config_set, training_data):
    yield AbstractModel(config_set, training_data, 4, 26544)


def test_abstract_model_best_config(abstract_model):
    best_config = abstract_model.get_best_config()
    assert best_config['spark.executor.cores'] == 4
    assert best_config['spark.sql.shuffle.partitions'] == 200
    assert best_config['spark.executor.memory'] == 23889
    assert best_config['spark.driver.memory'] == 4096
    assert best_config['spark.sql.autoBroadcastJoinThreshold'] == 100
    assert best_config['spark.sql.statistics.fallBackToHdfs'] == 1


def test_abstract_model_prune_config(gaussian_model, abstract_model):
    normalized_lhs_config = gaussian_model.get_sampled_configs()
    pruned_config = abstract_model.get_pruned_config(normalized_lhs_config)
    assert len(normalized_lhs_config) == 2000
    assert len(pruned_config) == 79