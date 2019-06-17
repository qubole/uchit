class SampledPerConfigs:
    """Sampled individual configs being auto-tuned for a particular spark job"""

    def __init__(self, norm_samples, sample_points, default_configs):
        self.__norm_samples = norm_samples
        self.__sample_points = sample_points
        self.__default_config = default_configs

    def get_sampled_configs(self):
        max = 0
        for configs in self.__default_config.values():
            if max < configs.get_num_elements:
                max = configs.get_num_elements

