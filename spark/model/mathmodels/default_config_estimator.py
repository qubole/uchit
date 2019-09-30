class DefaultConfigEstimator:

    def __init__(self, num_cores,  # int
                 total_memory):
        self.num_cores = num_cores
        self.total_memory = total_memory

    def get_min_executor_mem(self):
        """

        :return: Returns the minimum executor memory
        """
        return int(self.get_max_executor_mem() / self.num_cores)

    def get_max_executor_mem(self):
        """

        :return: Returns the maximum executor memory.
        """
        return int(self.total_memory * 0.9)

    def get_max_driver_memory(self):
        """

        :return: Returns maximum driver memory that can be set
        """
        return int(self.total_memory / self.num_cores)

    def get_max_broadcast_threshold(self):
        """

        :return: Returns maximum broadcast threshold that can be set
        """
        return self.get_max_broadcast_of_driver_mem(self.get_max_driver_memory())

    def get_max_broadcast_of_driver_mem(self, driver_memory):
        """

        :return: Returns maximum of driver memory that can be set as broadcast threshold
        """
        return min(max(int(driver_memory * 0.10), 10), 100)