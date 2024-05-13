from errr import exception as _e
from errr import make_tree as _t

_t(
    globals(),
    ScaffoldError=_e(
        CodeImportError=_e(),
        CLIError=_e(
            CommandError=_e(),
            ConfigTemplateNotFoundError=_e("template", "path"),
            InputError=_e(),
            DryrunError=_e(),
        ),
        ConfigurationError=_e(
            ConfigurationFormatError=_e(),
            DynamicClassError=_e(
                DynamicObjectNotFoundError=_e(),
                DynamicClassInheritanceError=_e(),
                ClassMapMissingError=_e(),
            ),
            BootError=_e("node"),
            CastError=_e(
                "node",
                "attr",
                DistributionCastError=_e(),
                UnresolvedClassCastError=_e(),
                UnfitClassCastError=_e("node", "attr"),
            ),
            CastConfigurationError=_e(),
            IndicatorError=_e(),
            RequirementError=_e("node", "attr"),
            CfgReferenceError=_e(
                NoReferenceAttributeSignal=_e(),
            ),
            UnknownConfigAttrError=_e("attributes"),
        ),
        CompilationError=_e(
            DistributorError=_e("property", "strategy"),
            RedoError=_e(),
        ),
        DependencyError=_e(),
        GatewayError=_e(
            AllenApiError=_e(),
        ),
        JobPoolError=_e(
            JobCancelledError=_e(),
            JobPoolContextError=_e(),
            JobSchedulingError=_e(),
        ),
        TopologyError=_e(
            UnmanagedPartitionError=_e(),
            LayoutError=_e(),
        ),
        TypeHandlingError=_e(
            NoneReferenceError=_e(),
            InvalidReferenceError=_e("value"),
        ),
        NodeNotFoundError=_e("query"),
        AdapterError=_e(),
        ConnectivityError=_e(
            ExternalSourceError=_e(
                MissingSourceError=_e(),
                IncompleteExternalMapError=_e(),
                SourceQualityError=_e(),
            ),
            UnknownGIDError=_e(),
        ),
        MorphologyError=_e(
            MorphologyRepositoryError=_e(),
            MissingMorphologyError=_e(),
            IncompleteMorphologyError=_e(),
            MorphologyDataError=_e(),
            CircularMorphologyError=_e("morphology", "component").set(list_details=True),
            CompartmentError=_e(),
            EmptySelectionError=_e("selectors"),
            EmptyBranchError=_e(),
        ),
        OptionError=_e(
            ReadOnlyOptionError=_e("option", "tag"),
        ),
        PlacementError=_e(
            ChunkError=_e(),
            PlacementRelationError=_e(),
            ContinuityError=_e(),
            PackingError=_e(),
        ),
        SimulationError=_e(
            ParameterError=_e(
                "parameter",
                ReificationError=_e(),
            )
        ),
        SelectorError=_e(),
        TreeError=_e(),
        VoxelSetError=_e(
            EmptyVoxelSetError=_e(),
        ),
        StorageError=_e(
            DatasetNotFoundError=_e(
                IntersectionDataNotFoundError=_e(),
            ),
            DatasetExistsError=_e(),
            DataNotFoundError=_e(),
            AttributeMissingError=_e(),
            UnknownStorageEngineError=_e(),
            MissingActiveConfigError=_e(),
        ),
        DataNotProvidedError=_e(),
        PluginError=_e("plugin"),
        ParserError=_e(
            FileImportError=_e(),
            FileReferenceError=_e(),
        ),
        ClassError=_e(),
    ),
)


# Warnings


class ScaffoldWarning(UserWarning):
    pass


class ConfigurationWarning(ScaffoldWarning):
    pass


class PlacementWarning(ScaffoldWarning):
    pass


class PackingWarning(PlacementWarning):
    pass


class MorphologyWarning(ScaffoldWarning):
    pass


class ConnectivityWarning(ScaffoldWarning):
    pass


class PackageRequirementWarning(ScaffoldWarning):
    pass


__all__ = [
    "AdapterError",
    "AllenApiError",
    "AttributeMissingError",
    "BootError",
    "CLIError",
    "CastConfigurationError",
    "CastError",
    "CfgReferenceError",
    "ChunkError",
    "CircularMorphologyError",
    "ClassError",
    "ClassMapMissingError",
    "CodeImportError",
    "CommandError",
    "CompartmentError",
    "CompilationError",
    "ConfigTemplateNotFoundError",
    "ConfigurationError",
    "ConfigurationFormatError",
    "ConfigurationWarning",
    "ConnectivityError",
    "ConnectivityWarning",
    "ContinuityError",
    "DataNotFoundError",
    "DataNotProvidedError",
    "DatasetExistsError",
    "DatasetNotFoundError",
    "DependencyError",
    "DistributionCastError",
    "DistributorError",
    "DryrunError",
    "DynamicClassError",
    "DynamicClassInheritanceError",
    "DynamicObjectNotFoundError",
    "EmptyBranchError",
    "EmptySelectionError",
    "EmptyVoxelSetError",
    "ExternalSourceError",
    "FileImportError",
    "FileReferenceError",
    "GatewayError",
    "IncompleteExternalMapError",
    "IncompleteMorphologyError",
    "IndicatorError",
    "InputError",
    "IntersectionDataNotFoundError",
    "InvalidReferenceError",
    "JobCancelledError",
    "JobPoolContextError",
    "JobPoolError",
    "JobSchedulingError",
    "LayoutError",
    "MissingActiveConfigError",
    "MissingMorphologyError",
    "MissingSourceError",
    "MorphologyDataError",
    "MorphologyError",
    "MorphologyRepositoryError",
    "MorphologyWarning",
    "NoReferenceAttributeSignal",
    "NodeNotFoundError",
    "NoneReferenceError",
    "OptionError",
    "PackageRequirementWarning",
    "PackingError",
    "PackingWarning",
    "ParameterError",
    "ParserError",
    "PlacementError",
    "PlacementRelationError",
    "PlacementWarning",
    "PluginError",
    "ReadOnlyOptionError",
    "RedoError",
    "ReificationError",
    "RequirementError",
    "ScaffoldError",
    "ScaffoldWarning",
    "SelectorError",
    "SimulationError",
    "SourceQualityError",
    "StorageError",
    "TopologyError",
    "TreeError",
    "TypeHandlingError",
    "UnfitClassCastError",
    "UnknownConfigAttrError",
    "UnknownGIDError",
    "UnknownStorageEngineError",
    "UnmanagedPartitionError",
    "UnresolvedClassCastError",
    "VoxelSetError",
]
