from spark.config.config_set import UniversalConfigSet
from spark.config.domain import IntRangeDomain
from spark.config.parameter import Parameter
import pytest


@pytest.mark.parametrize("num_cores, memory, param_list",
                         [(4, 28*1024, [Parameter('spark.sql.shuffle.partitions', IntRangeDomain(10, 2000, 50)),
                                        Parameter('spark.executor.memory', IntRangeDomain(7*1024*0.9, 28*1024*0.9, 512)),
                                        Parameter('spark.driver.memory', IntRangeDomain(1024, (7*1024), 256)),
                                        Parameter('spark.executor.cores', IntRangeDomain(1, 4, 1))]),
                          (2, 28*1024, [Parameter('spark.sql.shuffle.partitions', IntRangeDomain(10, 2000, 50)),
                                        Parameter('spark.executor.memory', IntRangeDomain(14*1024*0.9, 28*1024*0.9, 512)),
                                        Parameter('spark.driver.memory', IntRangeDomain(1024, 14*1024, 256)),
                                        Parameter('spark.executor.cores', IntRangeDomain(1, 2, 1))])
                          ])
def test_universal_config_set(num_cores, memory, param_list):
    univ_config = UniversalConfigSet(num_cores, memory)
    assert (univ_config.get_params() == param_list)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("num_cores, memory", [(0, 600), (1, 600), (2, 0), (-20, 1024), (5, -1024)])
def test_univarsal_config_set_exception(num_cores, memory):
    UniversalConfigSet(num_cores, memory)