import pytest
from spark.config.domain import IntRangeDomain
from spark.config.domain import FloatRangeDomain
from spark.config.domain import DomainType


@pytest.mark.parametrize("domain, exp_values, exp_type, exp_size, exp_max, exp_min",
                         [(IntRangeDomain(0, 110, 20), [0, 20, 40, 60, 80, 100], DomainType.INT, 6, 100, 0),
                          (IntRangeDomain(0.0, 110.0, 20.5), [0, 20, 40, 60, 80, 100], DomainType.INT, 6, 100, 0),
                          (FloatRangeDomain(0.0, 110.0, 20.0), [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
                           DomainType.FLOAT, 6, 100, 0.0),
                          (FloatRangeDomain(0, 120, 20.0), [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
                           DomainType.FLOAT, 6, 100.0, 0.0)])
def test_range_domain(domain, exp_values, exp_type, exp_size, exp_max, exp_min):
    assert domain.get_possible_values() == exp_values
    assert domain.get_type() == exp_type
    assert domain.get_num_elements() == exp_size
    assert domain.get_min() == exp_min
    assert domain.get_max() == exp_max
