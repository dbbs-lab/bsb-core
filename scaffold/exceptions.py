class ScaffoldException(Exception):
    pass

## Configuration

class ConfigurationException(ScaffoldException):
    pass

class DynamicClassException(ConfigurationException):
    pass

class ConfigurableClassNotFoundException(DynamicClassException):
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

## Nest

class NestException(ScaffoldException):
    pass

class NestKernelException(NestException):
    pass

class NestModelException(NestException):
    pass

## Morphologies

class MorphologyException(ScaffoldException):
    pass

class MissingMorphologyException(MorphologyException):
    pass

## Resources (HDF5, ...)

class ResourceException(ScaffoldException):
    pass

class DatasetNotFoundException(ResourceException):
    pass

class AttributeMissingException(ResourceException):
    pass


## Warnings

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
