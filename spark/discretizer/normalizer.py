from spark.config.domain import DomainType


class ConfigNormalizer:
    def __init__(self, config_set):
        self._config_set = config_set
        self._param_list = config_set.get_params()
        self._normalized_config = self.__init_normalized_config()

    def __init_normalized_config(self):
        normalized_config = []
        for param in self._param_list:
            domain = param.get_domain()
            norm_values = ConfigNormalizer.normalize_domain(domain)
            normalized_config.append(norm_values)
        return normalized_config

    def get_all_possible_normalized_configs(self):
        return self._normalized_config

    # TODO make this API clean
    # Currently it has assumption that normalized config_array will have values
    # from param in same order of self._param_list.
    def denormalize_config(self, normalized_config_array):
        assert len(normalized_config_array) == len(self._param_list)
        res = list()
        i = 0
        for param in self._param_list:
            res.append(ConfigNormalizer.denormalize(param, normalized_config_array[i]))
            i = i + 1
        return res

    def get_params(self):
        return self._param_list

    @staticmethod
    def norm_function(max_norm, min_norm):
        return lambda a: (a - min_norm) if (max_norm == min_norm) else (1/float(max_norm - min_norm)) * (a - min_norm)

    @staticmethod
    def normalize_domain(domain):
        norm_values = map(ConfigNormalizer.norm_function(domain.get_min(), domain.get_max()),
                          domain.get_possible_values())
        return norm_values

    @staticmethod
    def normalize(param, value):
        domain = param.get_domain()
        return list(map(ConfigNormalizer.norm_function(domain.get_min(), domain.get_max()), value))

    @staticmethod
    def denorm_func(min_norm, max_norm, domain_type):
        if domain_type == DomainType.INT:
            return lambda a: round(a * (max_norm - min_norm)) + min_norm
        else:
            return lambda a: float(a * (max_norm - min_norm)) + min_norm

    @staticmethod
    def denormalize(param, value):
        domain = param.get_domain()
        denormlizer_func = \
            ConfigNormalizer.denorm_func(domain.get_min(), domain.get_max(), domain.get_type())
        return map(denormlizer_func, value)

    @staticmethod
    def denormalize_value(param, value):
        domain = param.get_domain()
        denormlizer_func = ConfigNormalizer.denorm_func(domain.get_min(), domain.get_max(), domain.get_type())
        return denormlizer_func(value)
