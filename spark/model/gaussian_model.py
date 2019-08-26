from scipy.stats import norm

import itertools
import math
import numpy as np
import sys

from spark.config.config import Config
from spark.discretizer.lhs_discrete_sampler import LhsDiscreteSampler
from spark.discretizer.normalizer import ConfigNormalizer


class GaussianModel:

    # ToDO - Fix the values initialisation
    alpha = 2.0
    beta = np.array([1, 1, 1, 1]) * pow(10, -6)
    gamma = np.array([1, 1, 1, 1])
    theta = np.array([1.0, 0.17, 0.17, 0.8])

    def __init__(self, training_data, config_set):
        self.training_data = training_data
        self.config_set = config_set
        self.training_inp = []
        self.training_out = []
        self.best_out = None
        self.training_inp_normalized = []
        self.training_pair_wise_corr = None
        self.conf_names_params_mapping = {}
        self.normalizer = ConfigNormalizer(self.config_set)

    def train(self):
        if not self.training_data or self.training_data.size() == 0:
            raise Exception("No training data found")

        self.training_inp = []
        self.training_inp_normalized = []
        self.training_out = []
        self.best_out = None
        self.training_pair_wise_corr = None

        for data_point in self.training_data.get_training_data():
            self.training_inp.append(data_point["configs"].get_all_param_values())
            param_dict = data_point["configs"].get_params_dict()
            self.training_inp_normalized.append(
                list(map(lambda x: ConfigNormalizer.norm_function(
                    x.get_domain().get_max(), x.get_domain().get_min())(param_dict[x]),
                         param_dict)))
            self.training_out.append(data_point["output"])
            self.best_out = min(self.training_out)
        # ToDo: Implement a train function to find precise values of alpha, beta and gamma

    def get_param_object(self, config_name):
        if config_name not in self.conf_names_params_mapping:
            for param in self.config_set.get_params():
                self.conf_names_params_mapping[param.get_name()] = param
        return self.conf_names_params_mapping[config_name]

    def add_sample_to_train_data(self, training_sample, out):
        """
        training_sample is expected to be of the format
        {
            "spark.sql.shuffle.partitions": 100,
            "spark.executor.memory": 5120,
            "spark.driver.memory": 1024,
            "spark.executor.cores": 4,
            .
            .
            .
        }

        Internally we will map config names to their corresponding Param objects
        """
        valid_config_names = list(map(lambda x: x.get_name(), self.config_set.get_params()))
        if set(training_sample.keys()) != set(valid_config_names):
            raise Exception("Invalid config to be added as training data. Missing config: %s, "
                            "Extra config: %s" % (", ".join(list(set(valid_config_names)-set(training_sample.keys()))),
                                                  ", ".join(list(set(training_sample.keys())-set(valid_config_names)))))
        training_config = Config(self.config_set.num_cores, self.config_set.total_memory)
        # To maintain order to be same as the one in config_set(), iterate on valid_config_names instead of
        # training_sample.keys()
        for config_name in valid_config_names:
            training_config.add_param(self.get_param_object(config_name), training_sample[config_name])
        self.training_data.add_training_data(training_config, out)

    def get_sampled_configs(self):
        # Normalize the values
        # Use LHS to get the correct values
        lhs_sampler = LhsDiscreteSampler(self.normalizer.get_all_possible_normalized_configs(), 2)
        return lhs_sampler.get_samples(2)

    def get_best_config(self):
        if self.training_inp_normalized is None:
            raise Exception("No training data found")

        normalized_values = self.get_sampled_configs()
        best_config_value = None
        best_config = {}
        best_out = sys.maxint
        # for config in list(itertools.product(*normalized_values)):
        for config in normalized_values:
            out = self.predict(config)
            if out < best_out:
                best_out = out
                best_config_value = config

        denorm_best_config = self.normalizer.denormalize_config(best_config_value)
        i = 0
        for param in self.normalizer.get_params():
            best_config[param.get_name] = denorm_best_config[i]
            ++i
        return best_config

    def get_correlation(self, var1, var2):
        correlation = 1
        for i in range(0, len(var1)):
            term = math.exp(-self.theta[i] * pow(abs(var1[i] - var2[i]), self.gamma[i]))
            correlation = correlation * term
        return correlation

    def get_training_pairwise_correlation(self):
        if self.training_pair_wise_corr is None:
            metrics = []
            for i in range(0, len(self.training_inp_normalized)):
                metrics.append([])
                for j in range(0, len(self.training_inp_normalized)):
                    metrics[i].append(
                        self.get_correlation(self.training_inp_normalized[i], self.training_inp_normalized[j]))
            self.training_inp_normalized = np.array(metrics)

        return self.training_pair_wise_corr

    def get_correlation_with_train_data(self, config):
        metrics = []
        for i in range(0, len(self.training_inp_normalized)):
            metrics.append([])
            metrics[i].append(self.get_correlation(config, self.training_inp_normalized[i]))
        return np.array(metrics)

    def get_training_params(self):
        return self.training_inp_normalized

    def get_mean(self, config):
        term1 = np.dot(config, self.beta)
        term2 = self.training_out - np.dot(self.get_training_params(), self.beta)
        term3 = np.dot(self.get_correlation_with_train_data(config).transpose(),
                       np.linalg.inv(self.get_training_pairwise_correlation()))
        term4 = np.dot(term3, term2)
        return term1 + term4

    def get_variance(self, config):
        corr_with_train_data = self.get_correlation_with_train_data(config)
        corr_pairwise_train_data = self.get_training_pairwise_correlation()
        term1 = np.dot(corr_with_train_data.transpose(), np.linalg.inv(corr_pairwise_train_data))
        term2 = np.dot(term1, corr_with_train_data)
        term3 = 1 - term2
        return np.linalg.det(pow(self.alpha, 2) * term3)

    def get_mu(self, config):
        return (self.best_out - self.get_mean(config)) / math.sqrt(self.get_variance(config))

    def predict(self, config):
        mu = self.get_mu(config)
        return math.sqrt(self.get_variance(config)) * (mu * norm.cdf(mu) + norm.pdf(mu))
