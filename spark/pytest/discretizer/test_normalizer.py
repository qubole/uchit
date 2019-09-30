from spark.config.config_set import UniversalConfigSet
from spark.discretizer.normalizer import ConfigNormalizer


def test_normalization():
    config_set = UniversalConfigSet(4, 28*1024)
    normalizer = ConfigNormalizer(config_set)
    norm_configs = normalizer.get_all_possible_normalized_configs()
    assert len(norm_configs) == 6
    assert len(norm_configs[0]) == 200
    assert len(norm_configs[1]) == 49
    assert len(norm_configs[2]) == 22
    assert len(norm_configs[3]) == 4
    assert len(norm_configs[4]) == 19
    assert len(norm_configs[5]) == 2


def test_denormalization():
    config_set = UniversalConfigSet(4, 28*1024)
    normalizer = ConfigNormalizer(config_set)
    norm_configs = normalizer.get_all_possible_normalized_configs()
    denorm_config = normalizer.denormalize_config_set(norm_configs)
    i = 0
    for param in config_set.get_params():
        domain = param.get_domain()
        assert sorted(domain.get_possible_values()) == sorted(denorm_config[i])
        i = i + 1
