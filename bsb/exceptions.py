from errr import make_tree as _t, exception as _e

_t(
    globals(),
    ScaffoldError=_e(
        SpatialDimensionError=_e(),
        CLIError=_e(
            CommandError=_e(),
            ConfigTemplateNotFoundError=_e("template", "path"),
            InputError=_e(),
            DryrunError=_e(),
        ),
        ConfigurationError=_e(
            ConfigurationFormatError=_e(),
            DynamicClassError=_e(
                DynamicClassNotFoundError=_e(),
                DynamicClassInheritanceError=_e(),
                ClassMapMissingError=_e(),
            ),
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
            ReferenceError=_e(
                NoReferenceAttributeSignal=_e(),
            ),
            UnknownConfigAttrError=_e("attributes"),
        ),
        CompilationError=_e(
            ConnectivityError=_e(
                "connection_strategy",
                FiberTransformError=_e(
                    QuiverFieldError=_e(),
                ),
            ),
            PlacementError=_e(
                "cell_type", PlacementRelationError=_e("cell_type", "relation")
            ),
            DistributorError=_e("property", "strategy"),
            RedoError=_e(),
        ),
        DependencyError=_e(),
        GatewayError=_e(
            AllenApiError=_e(),
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
        AdapterError=_e(
            NeuronError=_e(
                DeviceConnectionError=_e(),
                TransmitterError=_e(),
                RelayError=_e(),
            ),
            NestError=_e(
                NestKernelError=_e(
                    NestModuleError=_e(),
                ),
                NestModelError=_e(),
                KernelLockedError=_e(),
                SuffixTakenError=_e(),
                ReceptorSpecificationError=_e(),
            ),
            ParallelIntegrityError=_e("rank"),
            ArborError=_e(),
        ),
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
            ContinuityError=_e(),
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
        ),
        DataNotProvidedError=_e(),
        PluginError=_e("plugin"),
        ParserError=_e(
            JsonParseError=_e(
                JsonReferenceError=_e(),
                JsonImportError=_e(),
            ),
        ),
        OrderError=_e(),
        ClassError=_e(),
    ),
)


## Warnings


class ScaffoldWarning(UserWarning):
    pass


class ConfigurationWarning(ScaffoldWarning):
    pass


class UserUserDeprecationWarning(ScaffoldWarning):
    pass


class PlacementWarning(ScaffoldWarning):
    pass


class MorphologyWarning(ScaffoldWarning):
    pass


class ConnectivityWarning(ScaffoldWarning):
    pass


class QuiverFieldWarning(ScaffoldWarning):
    pass


class RepositoryWarning(ScaffoldWarning):
    pass


class SimulationWarning(ScaffoldWarning):
    pass


class KernelWarning(SimulationWarning):
    pass


class CriticalDataWarning(ScaffoldWarning):
    pass
