import math
import numpy as np
import sys

from collections import OrderedDict
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from spark.config.config import Config
from spark.discretizer.lhs_discrete_sampler import LhsDiscreteSampler
from spark.discretizer.normalizer import ConfigNormalizer
from spark.model.model import Model


class GaussianModel(Model):

    def __init__(self, config_set, training_data):
        super(GaussianModel, self).__init__()
        self.training_data = training_data
        self.config_set = config_set
        self.training_inp = np.empty((0, config_set.get_size()), float)
        self.training_inp_normalized = np.empty((0, config_set.get_size()), float)
        self.training_out = np.empty((0, 1), float)
        self.best_out = None
        self.training_pair_wise_corr = None
        self.conf_names_params_mapping = {}
        self.normalizer = ConfigNormalizer(self.config_set)
        # ToDO - Fix the values initialisation
        self.alpha = 2.0
        self.gamma = np.ones(config_set.get_size(), float)
        self.theta = np.ones(config_set.get_size(), float) * 0.01
        # self.linearLasso = Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
        #                     normalize=False, positive=True, precompute=False, random_state=None,
        #                     selection='cyclic', tol=0.0001, warm_start=False)
        self.linear_model = LinearRegression(fit_intercept=True)
        self.linear_regression_model = None

    def train(self):
        if not self.training_data or self.training_data.size() == 0:
            raise Exception("No training data found")

        self.training_inp = np.empty((0, self.config_set.get_size()), float)
        self.training_inp_normalized = np.empty((0, self.config_set.get_size()), float)
        self.training_out = np.empty((0, 1), float)
        self.best_out = None
        self.training_pair_wise_corr = None

        for data_point in self.training_data.get_training_data():
            self.training_inp = np.vstack((self.training_inp, data_point["configs"].get_all_param_values()))
            param_dict = data_point["configs"].get_params_dict()
            self.training_inp_normalized = np.vstack((self.training_inp_normalized,
                                                     list(map(lambda x: ConfigNormalizer.norm_function(param_dict[x],
                                                         x.get_domain().get_min(), x.get_domain().get_max()), param_dict))))

            self.training_out = np.vstack((self.training_out, data_point["output"]))
            self.best_out = min(self.training_out)
        self.linear_regression_model = self.linear_model.fit(self.training_inp_normalized, self.training_out)

    def get_param_object(self, config_name):
        if config_name not in self.conf_names_params_mapping:
            for param in self.config_set.get_params():
                self.conf_names_params_mapping[param.get_name()] = param
        return self.conf_names_params_mapping[config_name]

    # ToDo: Move this to combiner
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
        training_config = Config()
        # To maintain order to be same as the one in config_set(), iterate on valid_config_names instead of
        # training_sample.keys()
        for config_name in valid_config_names:
            training_config.add_param(self.get_param_object(config_name), training_sample[config_name])
        self.training_data.add_training_data(training_config, out)

    def get_sampled_configs(self):
        # Normalize the values
        # Use LHS to get the correct values
        # ToDo: Fix the logic on number of sample configs to be picked
        config_samples = self.normalizer.get_all_possible_normalized_configs()
        size_sample = max(list(map(lambda x: len(x), config_samples)))
        if size_sample < 2000:
            size_sample = 2000
        lhs_sampler = LhsDiscreteSampler(self.normalizer.get_all_possible_normalized_configs(), size_sample)
        lhs_samples = lhs_sampler.get_samples(2)
        training_data_list = np.ndarray.tolist(self.training_inp_normalized)
        # remove training data from samples
        return_list = [x for x in lhs_samples if x not in training_data_list]
        return return_list

    def get_best_config(self):
        return self.get_best_config_for_config_space(self.get_sampled_configs())

    def get_best_config_for_config_space(self, lhs_config_space):
        if len(self.training_inp_normalized) == 0:
            raise Exception("Model is not trained")

        normalized_values = lhs_config_space
        best_config_value = None
        best_config = OrderedDict()
        best_out = sys.maxint

        for config_value in normalized_values:
            out = self.expected_improvement(config_value, self.alpha, self.gamma, self.theta)
            if out < best_out and out != 0:
                best_out = out
                best_config_value = config_value
        if best_out == sys.maxint:
            # find second best mu
            for config_value in normalized_values:
                out_predict = self.predict(config_value)
                if out_predict[0][0] < best_out:
                    best_out = out_predict[0][0]
                    best_config_value = config_value
        denorm_best_config = self.normalizer.denormalize_config(best_config_value)
        i = 0
        for param in self.normalizer.get_params():
            best_config[param.get_name()] = denorm_best_config[i]
            i = i + 1
        return best_config

    def get_correlation(self, var1, var2, gamma, theta):
        correlation = 1
        for i in range(0, len(var1)):
            term = math.exp(-theta[i] * pow(abs(var1[i] - var2[i]), gamma[i]))
            correlation = correlation * term
        return correlation

    def get_training_pairwise_correlation(self, gamma, theta):
        if self.training_pair_wise_corr is None or len(self.training_pair_wise_corr) == 0:
            metrics = []
            for i in range(0, len(self.training_inp_normalized)):
                metrics.append([])
                for j in range(0, len(self.training_inp_normalized)):
                    metrics[i].append(
                        self.get_correlation(self.training_inp_normalized[i], self.training_inp_normalized[j],
                        gamma, theta))
            self.training_pair_wise_corr = np.array(metrics)

        return self.training_pair_wise_corr

    def get_correlation_with_train_data(self, config_value, gamma, theta):
        metrics = []
        for i in range(0, len(self.training_inp_normalized)):
            metrics.append([])
            metrics[i].append(self.get_correlation(config_value, self.training_inp_normalized[i], gamma, theta))
        return np.array(metrics)

    def get_training_params(self):
        return self.training_inp_normalized

    def get_mean(self, config_value, gamma, theta):
        config_value_array = []
        config_value_array.append(np.array(config_value))
        term1 = self.linear_regression_model.predict(np.array(config_value_array))
        term2 = self.training_out - self.linear_regression_model.predict(self.training_inp_normalized)
        term3 = np.dot(self.get_correlation_with_train_data(config_value, gamma, theta).transpose(),
                       np.linalg.inv(self.get_training_pairwise_correlation(gamma, theta)))
        term4 = np.dot(term3, term2)
        return term1[0] + term4[0]

    def get_variance(self, config_value, alpha, gamma, theta):
        corr_with_train_data = self.get_correlation_with_train_data(config_value, gamma, theta)
        corr_pairwise_train_data = self.get_training_pairwise_correlation(gamma, theta)
        term1 = np.dot(corr_with_train_data.transpose(), np.linalg.inv(corr_pairwise_train_data))
        term2 = np.dot(term1, corr_with_train_data)
        term3 = 1 - term2
        return np.linalg.det(pow(alpha, 2) * term3)

    def get_mu(self, config_value, alpha, gamma, theta):
        # print self.get_variance(config_value, alpha, gamma, theta)
        return (self.best_out - self.get_mean(config_value, gamma, theta)) / \
               math.sqrt(self.get_variance(config_value, alpha, gamma, theta))

    def expected_improvement(self, config_value, alpha, gamma, theta):
        mu = self.get_mu(config_value, alpha, gamma, theta)
        return math.sqrt(self.get_variance(config_value, alpha, gamma, theta)) * (mu * norm.cdf(mu) + norm.pdf(mu))

    def predict(self, config_value):
        predict_mean = self.get_mean(config_value, self.gamma, self.theta)
        variance = math.sqrt(self.get_variance(config_value, self.alpha, self.gamma, self.theta))
        return predict_mean - 2 * variance, predict_mean + 2 * variance
