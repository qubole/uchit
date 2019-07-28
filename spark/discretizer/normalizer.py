# from ..config.config_values import ConfigType


class ConfigNormalizer:
    def __init__(self, configs):
        self._configs = configs
        self._config_dict = configs.get_config_dict()
        self._config_name_list = list(self._config_dict.keys())
        self._normalized_config = self.__init_normalized_config()

    def __init_normalized_config(self):
        normalized_config = []
        for config_name in self._config_name_list:
            config_value = self._config_dict[config_name]
            norm_values = ConfigNormalizer.normalize(config_value.get_values(),
                                           config_value.get_min_for_normalization(),
                                           config_value.get_max_for_normalization(),
                                           config_value.get_type())
            normalized_config.append(norm_values)
        return normalized_config

    def get_normalized_config(self):
        return self._normalized_config

    def get_config_names(self):
        return self._config_name_list

    @staticmethod
    def norm_function(max_norm, min_norm):
        return lambda a: (1/float(max_norm - min_norm)) * (a - min_norm)

    @staticmethod
    def normalize(values, min_norm, max_norm, type):
        norm_values = map(ConfigNormalizer.norm_function(min_norm, max_norm), values)
        # If we want to normalize all the values to be in between 0 and 1, we might have to remove this condition
        # if type == ConfigType.INT:
        #     norm_values = map(int, norm_values)
        return norm_values


class ConfigDenormalizer:
    def __init__(self, configs, config_keys):
        self._configs = configs
        self.__config_keys = config_keys

    def denormalize_config(self, norm_config):
        return_configs = {}
        for key in self.__config_keys:
            return_configs[key] = []

    @staticmethod
    def denorm_func(min_norm, max_norm):
        return lambda a: (a * (max_norm - min_norm)) + min_norm

    @staticmethod
    def denormalize(value, min_norm, max_norm):
        denormlizer_func = ConfigDenormalizer.denorm_func(min_norm, max_norm)
        return denormlizer_func(value)

