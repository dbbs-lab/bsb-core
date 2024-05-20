"""
`bsb-core` is the backbone package contain the essential code of the BSB: A component
framework for multiscale bottom-up neural modelling.

`bsb-core` needs to be installed alongside a bundle of desired bsb plugins, some of
which are essential for `bsb-core` to function. First time users are recommended to
install the `bsb` package instead.
"""

__version__ = "4.0.1"

import functools
import importlib
import sys
import typing
import warnings

import bsb.exceptions as _exc

# Patch functools on 3.8
try:
    functools.cache
except AttributeError:
    functools.cache = functools.lru_cache

    # Patch the 'register' method of `singledispatchmethod` pre python 3.10
    def _register(self, cls, method=None):  # pragma: nocover
        if hasattr(cls, "__func__"):
            setattr(cls, "__annotations__", cls.__func__.__annotations__)
        return self.dispatcher.register(cls, func=method)

    functools.singledispatchmethod.register = _register


# Always show all scaffold warnings
for e in _exc.__dict__.values():
    if isinstance(e, type) and issubclass(e, Warning):
        warnings.simplefilter("always", e)

try:
    from .options import profiling as _pr
except Exception:
    pass
else:
    if _pr:
        from .profiling import activate_session

        activate_session()


def _get_annotation_submodule(name: str):
    annotation = __annotations__.get(name, None)
    if annotation:
        type_ = typing.get_args(annotation)
        if type_:
            # typing.Type["bsb.submodule.name"]
            annotation = type_[0].__forward_arg__
        return annotation[4 : -len(name) - 1]


@functools.cache
def __getattr__(name):
    if name == "config":
        return object.__getattribute__(sys.modules[__name__], name)
    module = _get_annotation_submodule(name)
    if module is None:
        return object.__getattribute__(sys.modules[__name__], name)
    else:
        return getattr(importlib.import_module("." + module, package="bsb"), name)


@functools.cache
def __dir__():
    return [*__annotations__.keys()]


# Do not modify: autogenerated public API type annotations of the `bsb` module
# fmt: off
# isort: off
if typing.TYPE_CHECKING:
  import bsb.cell_types
  import bsb.cli
  import bsb.cli.commands
  import bsb.config
  import bsb.config.parsers
  import bsb.config.refs
  import bsb.config.types
  import bsb.connectivity.detailed.shared
  import bsb.connectivity.detailed.voxel_intersection
  import bsb.connectivity.general
  import bsb.connectivity.import_
  import bsb.connectivity.strategy
  import bsb.core
  import bsb.exceptions
  import bsb.mixins
  import bsb.morphologies
  import bsb.morphologies.parsers
  import bsb.morphologies.parsers.parser
  import bsb.morphologies.selector
  import bsb.option
  import bsb.options
  import bsb.placement.arrays
  import bsb.placement.distributor
  import bsb.placement.import_
  import bsb.placement.indicator
  import bsb.placement.random
  import bsb.placement.strategy
  import bsb.plugins
  import bsb.postprocessing
  import bsb.profiling
  import bsb.reporting
  import bsb.services
  import bsb.simulation
  import bsb.simulation.adapter
  import bsb.simulation.cell
  import bsb.simulation.component
  import bsb.simulation.connection
  import bsb.simulation.device
  import bsb.simulation.parameter
  import bsb.simulation.results
  import bsb.simulation.simulation
  import bsb.simulation.targetting
  import bsb.storage
  import bsb.storage._chunks
  import bsb.storage._files
  import bsb.storage.decorators
  import bsb.storage.interfaces
  import bsb.topology
  import bsb.topology.partition
  import bsb.topology.region
  import bsb.trees
  import bsb.voxels

AdapterError: typing.Type["bsb.exceptions.AdapterError"]
AdapterProgress: typing.Type["bsb.simulation.adapter.AdapterProgress"]
AfterConnectivityHook: typing.Type["bsb.postprocessing.AfterConnectivityHook"]
AfterPlacementHook: typing.Type["bsb.postprocessing.AfterPlacementHook"]
AllToAll: typing.Type["bsb.connectivity.general.AllToAll"]
AllenApiError: typing.Type["bsb.exceptions.AllenApiError"]
AllenStructure: typing.Type["bsb.topology.partition.AllenStructure"]
AttributeMissingError: typing.Type["bsb.exceptions.AttributeMissingError"]
BaseCommand: typing.Type["bsb.cli.commands.BaseCommand"]
BidirectionalContact: typing.Type["bsb.postprocessing.BidirectionalContact"]
BootError: typing.Type["bsb.exceptions.BootError"]
BoxTree: typing.Type["bsb.trees.BoxTree"]
BoxTreeInterface: typing.Type["bsb.trees.BoxTreeInterface"]
Branch: typing.Type["bsb.morphologies.Branch"]
BranchLocTargetting: typing.Type["bsb.simulation.targetting.BranchLocTargetting"]
BsbCommand: typing.Type["bsb.cli.commands.BsbCommand"]
BsbOption: typing.Type["bsb.option.BsbOption"]
BsbParser: typing.Type["bsb.morphologies.parsers.parser.BsbParser"]
ByIdTargetting: typing.Type["bsb.simulation.targetting.ByIdTargetting"]
ByLabelTargetting: typing.Type["bsb.simulation.targetting.ByLabelTargetting"]
CLIError: typing.Type["bsb.exceptions.CLIError"]
CLIOptionDescriptor: typing.Type["bsb.option.CLIOptionDescriptor"]
CastConfigurationError: typing.Type["bsb.exceptions.CastConfigurationError"]
CastError: typing.Type["bsb.exceptions.CastError"]
CellModel: typing.Type["bsb.simulation.cell.CellModel"]
CellModelFilter: typing.Type["bsb.simulation.targetting.CellModelFilter"]
CellModelTargetting: typing.Type["bsb.simulation.targetting.CellModelTargetting"]
CellTargetting: typing.Type["bsb.simulation.targetting.CellTargetting"]
CellType: typing.Type["bsb.cell_types.CellType"]
CfgReferenceError: typing.Type["bsb.exceptions.CfgReferenceError"]
Chunk: typing.Type["bsb.storage._chunks.Chunk"]
ChunkError: typing.Type["bsb.exceptions.ChunkError"]
CircularMorphologyError: typing.Type["bsb.exceptions.CircularMorphologyError"]
ClassError: typing.Type["bsb.exceptions.ClassError"]
ClassMapMissingError: typing.Type["bsb.exceptions.ClassMapMissingError"]
CodeDependencyNode: typing.Type["bsb.storage._files.CodeDependencyNode"]
CodeImportError: typing.Type["bsb.exceptions.CodeImportError"]
CommandError: typing.Type["bsb.exceptions.CommandError"]
CompartmentError: typing.Type["bsb.exceptions.CompartmentError"]
CompilationError: typing.Type["bsb.exceptions.CompilationError"]
ConfigTemplateNotFoundError: typing.Type["bsb.exceptions.ConfigTemplateNotFoundError"]
Configuration: typing.Type["bsb.config.Configuration"]
ConfigurationAttribute: typing.Type["bsb.config.ConfigurationAttribute"]
ConfigurationError: typing.Type["bsb.exceptions.ConfigurationError"]
ConfigurationFormatError: typing.Type["bsb.exceptions.ConfigurationFormatError"]
ConfigurationParser: typing.Type["bsb.config.parsers.ConfigurationParser"]
ConfigurationWarning: typing.Type["bsb.exceptions.ConfigurationWarning"]
ConnectionModel: typing.Type["bsb.simulation.connection.ConnectionModel"]
ConnectionStrategy: typing.Type["bsb.connectivity.strategy.ConnectionStrategy"]
ConnectionTargetting: typing.Type["bsb.simulation.targetting.ConnectionTargetting"]
ConnectivityError: typing.Type["bsb.exceptions.ConnectivityError"]
ConnectivityIterator: typing.Type["bsb.storage.interfaces.ConnectivityIterator"]
ConnectivitySet: typing.Type["bsb.storage.interfaces.ConnectivitySet"]
ConnectivityWarning: typing.Type["bsb.exceptions.ConnectivityWarning"]
ContinuityError: typing.Type["bsb.exceptions.ContinuityError"]
Convergence: typing.Type["bsb.connectivity.general.Convergence"]
CsvImportConnectivity: typing.Type["bsb.connectivity.import_.CsvImportConnectivity"]
CsvImportPlacement: typing.Type["bsb.placement.import_.CsvImportPlacement"]
CylindricalTargetting: typing.Type["bsb.simulation.targetting.CylindricalTargetting"]
DataNotFoundError: typing.Type["bsb.exceptions.DataNotFoundError"]
DataNotProvidedError: typing.Type["bsb.exceptions.DataNotProvidedError"]
DatasetExistsError: typing.Type["bsb.exceptions.DatasetExistsError"]
DatasetNotFoundError: typing.Type["bsb.exceptions.DatasetNotFoundError"]
DependencyError: typing.Type["bsb.exceptions.DependencyError"]
DeviceModel: typing.Type["bsb.simulation.device.DeviceModel"]
Distribution: typing.Type["bsb.config.Distribution"]
DistributionCastError: typing.Type["bsb.exceptions.DistributionCastError"]
DistributionContext: typing.Type["bsb.placement.distributor.DistributionContext"]
Distributor: typing.Type["bsb.placement.distributor.Distributor"]
DistributorError: typing.Type["bsb.exceptions.DistributorError"]
DistributorsNode: typing.Type["bsb.placement.distributor.DistributorsNode"]
DryrunError: typing.Type["bsb.exceptions.DryrunError"]
DynamicClassError: typing.Type["bsb.exceptions.DynamicClassError"]
DynamicClassInheritanceError: typing.Type["bsb.exceptions.DynamicClassInheritanceError"]
DynamicObjectNotFoundError: typing.Type["bsb.exceptions.DynamicObjectNotFoundError"]
EmptyBranchError: typing.Type["bsb.exceptions.EmptyBranchError"]
EmptySelectionError: typing.Type["bsb.exceptions.EmptySelectionError"]
EmptyVoxelSetError: typing.Type["bsb.exceptions.EmptyVoxelSetError"]
Engine: typing.Type["bsb.storage.interfaces.Engine"]
Entities: typing.Type["bsb.placement.strategy.Entities"]
EnvOptionDescriptor: typing.Type["bsb.option.EnvOptionDescriptor"]
ExplicitNoRotations: typing.Type["bsb.placement.distributor.ExplicitNoRotations"]
ExternalSourceError: typing.Type["bsb.exceptions.ExternalSourceError"]
FileDependency: typing.Type["bsb.storage._files.FileDependency"]
FileDependencyNode: typing.Type["bsb.storage._files.FileDependencyNode"]
FileImportError: typing.Type["bsb.exceptions.FileImportError"]
FileReferenceError: typing.Type["bsb.exceptions.FileReferenceError"]
FileScheme: typing.Type["bsb.storage._files.FileScheme"]
FileStore: typing.Type["bsb.storage.interfaces.FileStore"]
FixedIndegree: typing.Type["bsb.connectivity.general.FixedIndegree"]
FixedPositions: typing.Type["bsb.placement.strategy.FixedPositions"]
FractionFilter: typing.Type["bsb.simulation.targetting.FractionFilter"]
GatewayError: typing.Type["bsb.exceptions.GatewayError"]
GeneratedMorphology: typing.Type["bsb.storage.interfaces.GeneratedMorphology"]
HasDependencies: typing.Type["bsb.mixins.HasDependencies"]
Hemitype: typing.Type["bsb.connectivity.strategy.Hemitype"]
HemitypeCollection: typing.Type["bsb.connectivity.strategy.HemitypeCollection"]
Implicit: typing.Type["bsb.placement.distributor.Implicit"]
ImplicitNoRotations: typing.Type["bsb.placement.distributor.ImplicitNoRotations"]
ImportConnectivity: typing.Type["bsb.connectivity.import_.ImportConnectivity"]
ImportPlacement: typing.Type["bsb.placement.import_.ImportPlacement"]
IncompleteExternalMapError: typing.Type["bsb.exceptions.IncompleteExternalMapError"]
IncompleteMorphologyError: typing.Type["bsb.exceptions.IncompleteMorphologyError"]
IndicatorError: typing.Type["bsb.exceptions.IndicatorError"]
InputError: typing.Type["bsb.exceptions.InputError"]
Interface: typing.Type["bsb.storage.interfaces.Interface"]
IntersectionDataNotFoundError: typing.Type["bsb.exceptions.IntersectionDataNotFoundError"]
Intersectional: typing.Type["bsb.connectivity.detailed.shared.Intersectional"]
InvalidReferenceError: typing.Type["bsb.exceptions.InvalidReferenceError"]
InvertedRoI: typing.Type["bsb.mixins.InvertedRoI"]
JobCancelledError: typing.Type["bsb.exceptions.JobCancelledError"]
JobPool: typing.Type["bsb.services.JobPool"]
JobPoolContextError: typing.Type["bsb.exceptions.JobPoolContextError"]
JobPoolError: typing.Type["bsb.exceptions.JobPoolError"]
JobSchedulingError: typing.Type["bsb.exceptions.JobSchedulingError"]
LabelTargetting: typing.Type["bsb.simulation.targetting.LabelTargetting"]
Layer: typing.Type["bsb.topology.partition.Layer"]
LayoutError: typing.Type["bsb.exceptions.LayoutError"]
LocationTargetting: typing.Type["bsb.simulation.targetting.LocationTargetting"]
MPI: typing.Type["bsb.services.MPI"]
MPILock: typing.Type["bsb.services.MPILock"]
Meter: typing.Type["bsb.profiling.Meter"]
MissingActiveConfigError: typing.Type["bsb.exceptions.MissingActiveConfigError"]
MissingMorphologyError: typing.Type["bsb.exceptions.MissingMorphologyError"]
MissingSourceError: typing.Type["bsb.exceptions.MissingSourceError"]
MorphIOParser: typing.Type["bsb.morphologies.parsers.parser.MorphIOParser"]
Morphology: typing.Type["bsb.morphologies.Morphology"]
MorphologyDataError: typing.Type["bsb.exceptions.MorphologyDataError"]
MorphologyDependencyNode: typing.Type["bsb.storage._files.MorphologyDependencyNode"]
MorphologyDistributor: typing.Type["bsb.placement.distributor.MorphologyDistributor"]
MorphologyError: typing.Type["bsb.exceptions.MorphologyError"]
MorphologyGenerator: typing.Type["bsb.placement.distributor.MorphologyGenerator"]
MorphologyOperation: typing.Type["bsb.storage._files.MorphologyOperation"]
MorphologyParser: typing.Type["bsb.morphologies.parsers.parser.MorphologyParser"]
MorphologyRepository: typing.Type["bsb.storage.interfaces.MorphologyRepository"]
MorphologyRepositoryError: typing.Type["bsb.exceptions.MorphologyRepositoryError"]
MorphologySelector: typing.Type["bsb.morphologies.selector.MorphologySelector"]
MorphologySet: typing.Type["bsb.morphologies.MorphologySet"]
MorphologyWarning: typing.Type["bsb.exceptions.MorphologyWarning"]
NameSelector: typing.Type["bsb.morphologies.selector.NameSelector"]
NetworkDescription: typing.Type["bsb.storage.interfaces.NetworkDescription"]
NeuroMorphoScheme: typing.Type["bsb.storage._files.NeuroMorphoScheme"]
NeuroMorphoSelector: typing.Type["bsb.morphologies.selector.NeuroMorphoSelector"]
NoReferenceAttributeSignal: typing.Type["bsb.exceptions.NoReferenceAttributeSignal"]
NodeNotFoundError: typing.Type["bsb.exceptions.NodeNotFoundError"]
NoneReferenceError: typing.Type["bsb.exceptions.NoneReferenceError"]
NoopLock: typing.Type["bsb.storage.interfaces.NoopLock"]
NotParallel: typing.Type["bsb.mixins.NotParallel"]
NotSupported: typing.Type["bsb.storage.NotSupported"]
NrrdDependencyNode: typing.Type["bsb.storage._files.NrrdDependencyNode"]
NrrdVoxels: typing.Type["bsb.topology.partition.NrrdVoxels"]
Operation: typing.Type["bsb.storage._files.Operation"]
OptionDescriptor: typing.Type["bsb.option.OptionDescriptor"]
OptionError: typing.Type["bsb.exceptions.OptionError"]
PackageRequirement: typing.Type["bsb.config.types.PackageRequirement"]
PackageRequirementWarning: typing.Type["bsb.exceptions.PackageRequirementWarning"]
PackingError: typing.Type["bsb.exceptions.PackingError"]
PackingWarning: typing.Type["bsb.exceptions.PackingWarning"]
ParallelArrayPlacement: typing.Type["bsb.placement.arrays.ParallelArrayPlacement"]
Parameter: typing.Type["bsb.simulation.parameter.Parameter"]
ParameterError: typing.Type["bsb.exceptions.ParameterError"]
ParameterValue: typing.Type["bsb.simulation.parameter.ParameterValue"]
ParserError: typing.Type["bsb.exceptions.ParserError"]
Partition: typing.Type["bsb.topology.partition.Partition"]
PlacementError: typing.Type["bsb.exceptions.PlacementError"]
PlacementIndications: typing.Type["bsb.cell_types.PlacementIndications"]
PlacementIndicator: typing.Type["bsb.placement.indicator.PlacementIndicator"]
PlacementRelationError: typing.Type["bsb.exceptions.PlacementRelationError"]
PlacementSet: typing.Type["bsb.storage.interfaces.PlacementSet"]
PlacementStrategy: typing.Type["bsb.placement.strategy.PlacementStrategy"]
PlacementWarning: typing.Type["bsb.exceptions.PlacementWarning"]
Plotting: typing.Type["bsb.cell_types.Plotting"]
PluginError: typing.Type["bsb.exceptions.PluginError"]
ProfilingSession: typing.Type["bsb.profiling.ProfilingSession"]
ProgressEvent: typing.Type["bsb.simulation.simulation.ProgressEvent"]
ProjectOptionDescriptor: typing.Type["bsb.option.ProjectOptionDescriptor"]
RandomMorphologies: typing.Type["bsb.placement.distributor.RandomMorphologies"]
RandomPlacement: typing.Type["bsb.placement.random.RandomPlacement"]
RandomRotations: typing.Type["bsb.placement.distributor.RandomRotations"]
ReadOnlyManager: typing.Type["bsb.storage.interfaces.ReadOnlyManager"]
ReadOnlyOptionError: typing.Type["bsb.exceptions.ReadOnlyOptionError"]
RedoError: typing.Type["bsb.exceptions.RedoError"]
Reference: typing.Type["bsb.config.refs.Reference"]
Region: typing.Type["bsb.topology.region.Region"]
RegionGroup: typing.Type["bsb.topology.region.RegionGroup"]
ReificationError: typing.Type["bsb.exceptions.ReificationError"]
Relay: typing.Type["bsb.postprocessing.Relay"]
ReportListener: typing.Type["bsb.core.ReportListener"]
RepresentativesTargetting: typing.Type["bsb.simulation.targetting.RepresentativesTargetting"]
RequirementError: typing.Type["bsb.exceptions.RequirementError"]
Rhomboid: typing.Type["bsb.topology.partition.Rhomboid"]
RootCommand: typing.Type["bsb.cli.commands.RootCommand"]
RotationDistributor: typing.Type["bsb.placement.distributor.RotationDistributor"]
RotationSet: typing.Type["bsb.morphologies.RotationSet"]
RoundRobinMorphologies: typing.Type["bsb.placement.distributor.RoundRobinMorphologies"]
Scaffold: typing.Type["bsb.core.Scaffold"]
ScaffoldError: typing.Type["bsb.exceptions.ScaffoldError"]
ScaffoldWarning: typing.Type["bsb.exceptions.ScaffoldWarning"]
ScriptOptionDescriptor: typing.Type["bsb.option.ScriptOptionDescriptor"]
SelectorError: typing.Type["bsb.exceptions.SelectorError"]
Simulation: typing.Type["bsb.simulation.simulation.Simulation"]
SimulationBackendPlugin: typing.Type["bsb.simulation.SimulationBackendPlugin"]
SimulationComponent: typing.Type["bsb.simulation.component.SimulationComponent"]
SimulationData: typing.Type["bsb.simulation.adapter.SimulationData"]
SimulationError: typing.Type["bsb.exceptions.SimulationError"]
SimulationRecorder: typing.Type["bsb.simulation.results.SimulationRecorder"]
SimulationResult: typing.Type["bsb.simulation.results.SimulationResult"]
SimulatorAdapter: typing.Type["bsb.simulation.adapter.SimulatorAdapter"]
SomaTargetting: typing.Type["bsb.simulation.targetting.SomaTargetting"]
SourceQualityError: typing.Type["bsb.exceptions.SourceQualityError"]
SphericalTargetting: typing.Type["bsb.simulation.targetting.SphericalTargetting"]
SpoofDetails: typing.Type["bsb.postprocessing.SpoofDetails"]
Stack: typing.Type["bsb.topology.region.Stack"]
Storage: typing.Type["bsb.storage.Storage"]
StorageError: typing.Type["bsb.exceptions.StorageError"]
StorageNode: typing.Type["bsb.storage.interfaces.StorageNode"]
StoredFile: typing.Type["bsb.storage.interfaces.StoredFile"]
StoredMorphology: typing.Type["bsb.storage.interfaces.StoredMorphology"]
SubTree: typing.Type["bsb.morphologies.SubTree"]
Targetting: typing.Type["bsb.simulation.targetting.Targetting"]
TopologyError: typing.Type["bsb.exceptions.TopologyError"]
TreeError: typing.Type["bsb.exceptions.TreeError"]
TypeHandler: typing.Type["bsb.config.types.TypeHandler"]
TypeHandlingError: typing.Type["bsb.exceptions.TypeHandlingError"]
UnfitClassCastError: typing.Type["bsb.exceptions.UnfitClassCastError"]
UnknownConfigAttrError: typing.Type["bsb.exceptions.UnknownConfigAttrError"]
UnknownGIDError: typing.Type["bsb.exceptions.UnknownGIDError"]
UnknownStorageEngineError: typing.Type["bsb.exceptions.UnknownStorageEngineError"]
UnmanagedPartitionError: typing.Type["bsb.exceptions.UnmanagedPartitionError"]
UnresolvedClassCastError: typing.Type["bsb.exceptions.UnresolvedClassCastError"]
UriScheme: typing.Type["bsb.storage._files.UriScheme"]
UrlScheme: typing.Type["bsb.storage._files.UrlScheme"]
VolumetricRotations: typing.Type["bsb.placement.distributor.VolumetricRotations"]
VoxelData: typing.Type["bsb.voxels.VoxelData"]
VoxelIntersection: typing.Type["bsb.connectivity.detailed.voxel_intersection.VoxelIntersection"]
VoxelSet: typing.Type["bsb.voxels.VoxelSet"]
VoxelSetError: typing.Type["bsb.exceptions.VoxelSetError"]
Voxels: typing.Type["bsb.topology.partition.Voxels"]
WeakInverter: typing.Type["bsb.config.types.WeakInverter"]
WorkflowError: typing.Type["bsb.services.WorkflowError"]
activate_session: "bsb.profiling.activate_session"
box_layout: "bsb.topology.box_layout"
branch_iter: "bsb.morphologies.branch_iter"
chunklist: "bsb.storage._chunks.chunklist"
compose_nodes: "bsb.config.compose_nodes"
copy_configuration_template: "bsb.config.copy_configuration_template"
create_engine: "bsb.storage.create_engine"
create_topology: "bsb.topology.create_topology"
discover: "bsb.plugins.discover"
discover_engines: "bsb.storage.discover_engines"
format_configuration_content: "bsb.config.format_configuration_content"
from_storage: "bsb.core.from_storage"
get_active_session: "bsb.profiling.get_active_session"
get_config_attributes: "bsb.config.get_config_attributes"
get_config_path: "bsb.config.get_config_path"
get_configuration_parser: "bsb.config.parsers.get_configuration_parser"
get_configuration_parser_classes: "bsb.config.parsers.get_configuration_parser_classes"
get_engine_node: "bsb.storage.get_engine_node"
get_engines: "bsb.storage.get_engines"
get_module_option: "bsb.options.get_module_option"
get_option: "bsb.options.get_option"
get_option_classes: "bsb.options.get_option_classes"
get_option_descriptor: "bsb.options.get_option_descriptor"
get_option_descriptors: "bsb.options.get_option_descriptors"
get_partitions: "bsb.topology.get_partitions"
get_project_option: "bsb.options.get_project_option"
get_root_regions: "bsb.topology.get_root_regions"
get_simulation_adapter: "bsb.simulation.get_simulation_adapter"
handle_cli: "bsb.cli.handle_cli"
handle_command: "bsb.cli.handle_command"
is_module_option_set: "bsb.options.is_module_option_set"
is_partition: "bsb.topology.is_partition"
is_region: "bsb.topology.is_region"
load_root_command: "bsb.cli.commands.load_root_command"
make_config_diagram: "bsb.config.make_config_diagram"
meter: "bsb.profiling.meter"
node_meter: "bsb.profiling.node_meter"
on_main: "bsb.storage.decorators.on_main"
on_main_until: "bsb.storage.decorators.on_main_until"
open_storage: "bsb.storage.open_storage"
parse_configuration_content: "bsb.config.parse_configuration_content"
parse_configuration_file: "bsb.config.parse_configuration_file"
parse_morphology_content: "bsb.morphologies.parsers.parse_morphology_content"
parse_morphology_file: "bsb.morphologies.parsers.parse_morphology_file"
read_option: "bsb.options.read_option"
refs: "bsb.config.refs"
register_option: "bsb.options.register_option"
register_service: "bsb.services.register_service"
report: "bsb.reporting.report"
reset_module_option: "bsb.options.reset_module_option"
set_module_option: "bsb.options.set_module_option"
store_option: "bsb.options.store_option"
types: "bsb.config.types"
unregister_option: "bsb.options.unregister_option"
view_profile: "bsb.profiling.view_profile"
view_support: "bsb.storage.view_support"
walk_node_attributes: "bsb.config.walk_node_attributes"
walk_nodes: "bsb.config.walk_nodes"
warn: "bsb.reporting.warn"
