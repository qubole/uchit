class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class UnsupportedError(Error):
    """Exception raised for Unsupported operations.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
