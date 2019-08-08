class Parameter:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain

    def get_name(self):
        return self.name

    def get_domain(self):
        return self.domain