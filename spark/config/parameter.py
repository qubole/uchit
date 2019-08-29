class Parameter:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain

    def get_name(self):
        return self.name

    def get_domain(self):
        return self.domain

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and self.domain == other.domain
        return NotImplemented

    def __hash__(self):
        return hash(self.name) ^ self.domain.__hash__()

    def __ne__(self, other):
        return not self.__eq__(other)