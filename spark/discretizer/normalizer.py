from ..config.config_values import ConfigType
class ConfigNormalizer:
    def __init__(self, configs):
        self.__configs = configs
        self.__config_dict = configs.get_config_dict()
        self.__config_name_list = list(self.__config_dict.key())
        self.__normalized_config = self.__init_normalized_config()

    def __init_normalized_config(self):
        normalized_config = []
        for config_name in self.__config_name_list:
            config_value = self.__config_dict[config_name]
            norm_values = self.__normalize(config_value.get_values,
                                 config_value.get_min_for_normalization,
                                 config_value.get_max_for_normalization,
                                 config_value.get_type)
            normalized_config.append(norm_values)
        return normalized_config

    def get_normalized_config(self):
        return self.__normalized_config

    def get_config_names(self):
        return self.__config_name_list

    def __normalize(self, values, min_norm, max_norm, type):
        norm_func = lambda a: (1/float(max_norm - min_norm)) * (a - min_norm)
        norm_values = map(norm_func, values)
        if type == ConfigType.INT:
            norm_values = map(int, norm_values)
        return norm_values


class ConfigDenormalizer:
    def __init__(self, configs, config_keys):
        self.__configs = configs
        self.__config_keys = config_keys

    def denormalize_config(self, norm_config):
        return_configs = {}
        for key in self.__config_keys:
            return_configs[key] = []
