from spark.config.config_set import UniversalConfigSet
from spark.discretizer.lhs_discrete_sampler import LhsDiscreteSampler
from spark.discretizer.normalizer import ConfigNormalizer


def test_discretizer_for_r4_xlarge():
    config_set = UniversalConfigSet(4, 26544)
    normalizer = ConfigNormalizer(config_set)
    norm_configs = normalizer.get_all_possible_normalized_configs()
    sampler = LhsDiscreteSampler(norm_configs)
    samples = sampler.get_samples(2)
    assert max(list(map(lambda x: len(x), norm_configs))) == len(samples)
    assert all(map(lambda x: len(x) == len(norm_configs), samples))
