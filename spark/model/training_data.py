class TrainingData:
    def __init__(self):
        self.data = []

    def add_training_data(self, configs, output):
        self.data.append({"configs": configs, "output": output})

    def get_training_data(self):
        return self.data

    def size(self):
        return len(self.data)
