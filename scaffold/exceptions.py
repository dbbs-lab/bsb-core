class ScaffoldError(Exception):
    pass


class SpatialDimensionError(ScaffoldError):
    pass


## Configuration


class ConfigurationError(ScaffoldError):
    pass


class ConfigurationFormatError(ConfigurationError):
    pass


class DynamicClassError(ConfigurationError):
    pass


class ConfigurableClassNotFoundError(DynamicClassError):
    pass


class CastError(ConfigurationError):
    pass


class CastConfigurationError(ConfigurationError):
    pass


class UnionCastError(CastError):
    pass


class ConfigurableCastError(CastError):
    pass


class UnknownDistributionError(ConfigurableCastError):
    pass


class InvalidDistributionError(ConfigurableCastError):
    pass


class TypeNotFoundError(ScaffoldError):
    pass


class LayerNotFoundError(ScaffoldError):
    pass


class SimulationNotFoundError(ScaffoldError):
    pass


## Nest


class NestError(ScaffoldError):
    pass


class AdapterError(NestError):
    pass


class NestKernelError(AdapterError):
    pass


class KernelLockedError(NestKernelError):
    pass


class SuffixTakenError(KernelLockedError):
    pass


class NestModelError(NestError):
    pass


class NestModuleError(NestKernelError):
    pass


class ReceptorSpecificationError(NestError):
    pass


## Connectivity


class ConnectivityError(ScaffoldError):
    pass


## Morphologies


class MorphologyError(ScaffoldError):
    pass


class MorphologyRepositoryError(MorphologyError):
    pass


class MissingMorphologyError(MorphologyError):
    pass


class IncompleteMorphologyError(MorphologyError):
    pass


class MorphologyDataError(MorphologyError):
    pass


class CompartmentError(MorphologyError):
    pass


class TreeError(ScaffoldError):
    pass


class VoxelizationError(ScaffoldError):
    pass


class VoxelTransformError(VoxelizationError):
    pass


## Resources (HDF5, ...)


class ResourceError(ScaffoldError):
    pass


class DatasetNotFoundError(ResourceError):
    pass


class IntersectionDataNotFoundError(DatasetNotFoundError):
    pass


class DataNotFoundError(ResourceError):
    pass


class DataNotProvidedError(ScaffoldError):
    pass


class AttributeMissingError(ResourceError):
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


## Misc


class OrderError(ScaffoldError):
    pass


class ClassError(ScaffoldError):
    pass
