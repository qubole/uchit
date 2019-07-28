class ConfigsElement:
    """
    Class to store the an unique combination of the configs and its output. Can be used for storing the training data
    """

    def __init__(self):
        self.config_elements_list = []

    def add_config(self, configs, out):
        self.config_elements_list.append({"configs": configs, "out": out})
