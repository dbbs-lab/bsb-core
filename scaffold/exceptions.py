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

class NestModelException(Exception):
    pass

class MorphologyException(Exception):
    pass

class MissingMorphologyException(MorphologyException):
    pass

class DynamicClassException(ConfigurationException):
    pass

class ScaffoldWarning(UserWarning):
    pass

class PlacementWarning(ScaffoldWarning):
    pass

class ConnectivityWarning(ScaffoldWarning):
    pass

class RepositoryWarning(ScaffoldWarning):
    pass

class SimulationWarning(ScaffoldWarning):
    pass

class KernelWarning(SimulationWarning):
    pass
