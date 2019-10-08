from spark.combiner.combiner import Combiner
from spark.pytest.fixtures.test_framework import *


class TestCombiner:
    def test_combiner(self, training_data):
        combiner = Combiner(4, 26544, training_data)
        best_config = combiner.get_best_config()
        assert True

    def test_combiner_temp(self):
        combiner = Combiner(4, 26544)
        training_data_1 = {
            "spark.executor.memory": 11945,
            "spark.sql.shuffle.partitions": 200,
            "spark.executor.cores": 2,
            "spark.driver.memory": 1024 * 2,
            "spark.sql.autoBroadcastJoinThreshold": 10,
            "spark.sql.statistics.fallBackToHdfs": 0
        }
        runtime_in_sec = 248
        combiner.add_training_data(training_data_1, runtime_in_sec)
        training_data_2 = {
            "spark.executor.memory": 23889,
            "spark.sql.shuffle.partitions": 200,
            "spark.executor.cores": 4,
            "spark.driver.memory": 1024 * 2,
            "spark.sql.autoBroadcastJoinThreshold": 100,
            "spark.sql.statistics.fallBackToHdfs": 1
        }
        runtime_in_sec = 92
        combiner.add_training_data(training_data_2, runtime_in_sec)
        training_data_3 = {
            "spark.executor.memory": 11940,
            "spark.sql.shuffle.partitions": 460,
            "spark.executor.cores": 2,
            "spark.driver.memory": 2304,
            "spark.sql.autoBroadcastJoinThreshold": 30,
            "spark.sql.statistics.fallBackToHdfs": 1
        }
        runtime_in_sec = 105
        combiner.add_training_data(training_data_3, runtime_in_sec)
        best_config = combiner.get_best_config()
        print best_config