from spark.combiner.combiner import Combiner
from spark.pytest.fixtures.test_framework import *


class TestCombiner:
    def test_combiner(self, training_data):
        combiner = Combiner(4, 26544, training_data)
        best_config = combiner.get_best_config()
        assert True