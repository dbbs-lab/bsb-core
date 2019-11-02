class ConfigurationException(Exception):
    pass

class CastException(ConfigurationException):
    pass

class UnionCastException(CastException):
    pass

class ConfigurableCastException(CastException):
    pass

class UnknownDistributionException(ConfigurableCastException):
    pass

class InvalidDistributionException(ConfigurableCastException):
    pass

class NestException(Exception):
    pass

class NestKernelException(Exception):
    pass

class MorphologyException(Exception):
    pass

class MissingMorphologyException(MorphologyException):
    pass
