class ConfigurationException(Exception):
    pass

class CastException(ConfigurationException):
    pass

class UnionCastException(CastException):
    pass

class ConfigurableCastException(CastException):
    pass

class UnknownDistributionException(ConfigurationException):
    pass

class InvalidDistributionException(ConfigurationException):
    pass
