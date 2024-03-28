import abc
import uuid
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from .. import config
from .._util import rotation_matrix_from_vectors
from ..config.types import ndarray
from ..exceptions import EmptySelectionError
from ..morphologies import MorphologySet, RotationSet
from ..profiling import node_meter
from ..storage._files import NrrdDependencyNode
from ..topology.partition import Partition
from .indicator import PlacementIndications


@dataclass
class DistributionContext:
    indicator: PlacementIndications
    partitions: List[Partition]


@config.dynamic(attr_name="strategy", required=True)
class Distributor(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super(cls, cls).__init_subclass__(**kwargs)
        # Decorate subclasses to measure performance
        node_meter("distribute")(cls)

    @abc.abstractmethod
    def distribute(self, positions, context):
        """
        Is called to distribute cell properties.

        :param partitions: The partitions the cells were placed in.
        :type context: Additional context information such as the placement indications
          and partitions.
        :type context: ~bsb.placement.distributor.DistributionContext
        :returns: An array with the property data
        :rtype: numpy.ndarray
        """
        pass


@config.dynamic(
    attr_name="strategy", required=False, default="random", auto_classmap=True
)
class MorphologyDistributor(Distributor):
    may_be_empty = config.attr(type=bool, default=False)

    @abc.abstractmethod
    def distribute(self, positions, morphologies, context):
        """
        Is called to distribute cell morphologies and optionally rotations.

        :param positions: Placed positions under consideration
        :type positions: numpy.ndarray
        :param morphologies: The template morphology loaders. You can decide to use them
          and/or generate new ones in the MorphologySet that you produce. If you produce
          any new morphologies, don't forget to encapsulate them in a
          :class:`~bsb.storage.interfaces.StoredMorphology` loader, or better yet, use
          the :class:`~bsb.placement.distributor.MorphologyGenerator`.
        :param context: The placement indicator and partitions.
        :type context: ~bsb.placement.distributor.DistributionContext
        :returns: A MorphologySet with assigned morphologies, and optionally a RotationSet
        :rtype: Union[~bsb.morphologies.MorphologySet, Tuple[
          ~bsb.morphologies.MorphologySet, ~bsb.morphologies.RotationSet]]
        """
        pass


@config.node
class RandomMorphologies(MorphologyDistributor, classmap_entry="random"):
    """
    Distributes selected morphologies randomly without rotating them.

    .. code-block:: json

      { "placement": { "place_XY": {
        "distribute": {
            "morphologies": {"strategy": "random"}
        }
      }}}
    """

    may_be_empty = config.provide(False)

    def distribute(self, positions, morphologies, context):
        """
        Uses the morphology selection indicators to select morphologies and
        returns a MorphologySet of randomly assigned morphologies
        """
        return np.random.default_rng().integers(len(morphologies), size=len(positions))


@config.node
class RoundRobinMorphologies(MorphologyDistributor, classmap_entry="roundrobin"):
    """
    Distributes selected morphologies round robin, values are looped and assigned one by
    one in order.

    .. code-block:: json

      { "placement": { "place_XY": {
        "distribute": {
            "morphologies": {"strategy": "roundrobin"}
        }
      }}}
    """

    may_be_empty = config.provide(False)

    def distribute(self, positions, morphologies, context):
        ll = len(morphologies)
        lp = len(positions)
        return np.tile(np.arange(ll), lp // ll + 1)[:lp]


@config.node
class MorphologyGenerator(MorphologyDistributor, classmap_entry=None):
    """
    Special case of the morphology distributor that provides extra convenience when
    generating new morphologies.
    """

    may_be_empty = config.attr(type=bool, default=True)

    def __init_subclass__(cls, **kwargs):
        super(cls, cls).__init_subclass__(**kwargs)
        # Decorate subclasses to measure performance
        node_meter("generate")(cls)

    def distribute(self, positions, morphologies, context):
        pass

    @abc.abstractmethod
    def generate(self, positions, morphologies, context):
        pass


@config.dynamic(attr_name="strategy", required=False, default="none", auto_classmap=True)
class RotationDistributor(Distributor):
    """
    Rotates everything by nothing!
    """

    @abc.abstractmethod
    def distribute(self, positions, context):
        pass


@config.node
class ExplicitNoRotations(RotationDistributor, classmap_entry="explicitly_none"):
    def distribute(self, positions, context):
        return np.zeros((len(positions), 3))


# Sentinel mixin class to tag RotationDistributor as being overridable by morphology
# distributor.
class Implicit:
    pass


@config.node
class ImplicitNoRotations(ExplicitNoRotations, Implicit, classmap_entry="none"):
    pass


@config.node
class RandomRotations(RotationDistributor, classmap_entry="random"):
    def distribute(self, positions, context):
        return np.random.rand(len(positions), 3) * 360


@config.node
class VolumetricRotations(RotationDistributor, classmap_entry="orientation_field"):
    orientation_path = config.attr(required=True, type=NrrdDependencyNode)
    """Path to the nrrd file containing the volumetric orientation field. It provides a rotation 
    for each voxel considered. Its shape should be (3, L, W, D) where L, W and D are the sizes of 
    the field."""
    orientation_resolution = config.attr(required=False, default=25.0, type=float)
    """Voxel size resolution of the orientation field.
    """
    default_vector = config.attr(
        required=False,
        default=lambda: np.array([0.0, -1.0, 0.0]),
        call_default=True,
        type=ndarray(),
    )
    """Default orientation vector of each position.
    """
    space_origin = config.attr(
        required=False,
        default=lambda: np.array([0.0, 0.0, 0.0]),
        call_default=True,
        type=ndarray(),
    )
    """Origin point for the orientation field.
    """

    def distribute(self, positions, context):
        """
        Rotates according to a volumetric orientation field of specific resolution.
        For each position, find the equivalent voxel in the volumetric orientation field and apply
        the rotation from the default_vector to the corresponding orientation vector.
        Positions outside the orientation field will not be rotated.

        :param positions: Placed positions under consideration. Its shape is (N, 3) where N is the
            number of positions.
        :param context: The placement indicator and partitions.
        :type context: ~bsb.placement.distributor.DistributionContext
        :returns: A RotationSet object containing the 3D Euler angles in degrees for the rotation
            of each position.
        :rtype: RotationSet
        """

        orientation_field = self.orientation_path.load_object()
        voxel_pos = np.asarray(
            np.floor((positions - self.space_origin) / self.orientation_resolution),
            dtype=int,
        )

        # filter for positions inside the orientation field.
        filter_inside = (
            np.all(voxel_pos >= 0, axis=1)
            * (voxel_pos[:, 0] < orientation_field.shape[1])
            * (voxel_pos[:, 1] < orientation_field.shape[2])
            * (voxel_pos[:, 2] < orientation_field.shape[3])
        )

        # By default, positions outside the field should not rotate.
        # So their target orientation vector will be set to the default_vector,
        # from which the rotation is processed.
        orientations = np.full((positions.shape[0], 3), self.default_vector, dtype=float)
        # Expected orientation_field shape is (3, L, W, D) where L, W and D are the sizes
        # of the field. Here we want to filter on the space dimensions, so we move the axes.
        if filter_inside.any():
            orientations[filter_inside] = np.moveaxis(orientation_field, 0, -1)[
                voxel_pos[filter_inside, 0],
                voxel_pos[filter_inside, 1],
                voxel_pos[filter_inside, 2],
            ]
            orientations[
                np.isnan(orientations).any(axis=1) + ~orientations.any(axis=1)
            ] = self.default_vector

        return RotationSet(
            Rotation.from_matrix(
                rotation_matrix_from_vectors(self.default_vector, v)
            ).as_euler("xyz", degrees=True)
            for v in orientations
        )


@config.node
class DistributorsNode:
    morphologies: MorphologyDistributor = config.attr(
        type=MorphologyDistributor, default=dict, call_default=True
    )
    rotations: RotationDistributor = config.attr(
        type=RotationDistributor, default=dict, call_default=True
    )
    properties: dict[Distributor] = config.catch_all(type=Distributor)

    def __call__(self, key, partitions, indicator, positions, loaders=None):
        context = DistributionContext(indicator, partitions)
        if key == "morphologies":
            distributor = getattr(self, key)
            if hasattr(distributor, "generate"):
                distribute = distributor.generate
            else:
                distribute = distributor.distribute
            values = distribute(positions, loaders, context)
            if isinstance(values, tuple):
                # Check for accidental tuple return values. If you return a 2 sized
                # tuple you're still fucked, but the Gods must truly hate you.
                try:
                    morphologies, rotations = values
                except TypeError:
                    raise ValueError(
                        "Morphology distributors may only return tuples when they are"
                        + " to be unpacked as (morphologies, rotations)"
                    ) from None
            else:
                values = (values, None)
            return values
        elif key == "rotations":
            distribute = getattr(self, key).distribute
        else:
            distribute = self.properties[key].distribute
        return distribute(positions, context)

    def _curry(self, partitions, indicator, positions, loaders=None):
        def curried(key):
            return self(key, partitions, indicator, positions, loaders)

        return curried

    def _specials(self, partitions, indicator, positions):
        sel = indicator.assert_indication("morphologies")
        loaders = self.scaffold.storage.morphologies.select(*sel)
        if not loaders and not self.morphologies.may_be_empty:
            raise EmptySelectionError(
                f"Given {len(sel)} selectors: did not find any suitable morphologies", sel
            )
        distr = self._curry(partitions, indicator, positions, loaders)
        morphologies, rotations = distr("morphologies")
        if morphologies is not None and (
            rotations is None or not isinstance(self.rotations, Implicit)
        ):
            # If a RotationDistributor is not explicitly marked as `Implicit`, it
            # overrides the MorphologyDistributor's implicit rotations.
            rotations = distr("rotations")
        if hasattr(self.morphologies, "generate"):
            prefix = self._config_parent.name
            generated = {}
            indices = []
            # Get all the unique morphology objects from the return value and map to them
            for m in morphologies:
                idx = generated.setdefault(m, len(generated))
                indices.append(idx)
            mr = self.scaffold.morphologies
            uid = uuid.uuid4()
            loaders = []
            all_meta = {}
            # Save morphologies one by one
            for gen_morpho, i in generated.items():
                name = f"{prefix}-{uid}-{i}"
                # `update_metadata=False` so morphology is incompletely stored.
                saved = mr.save(name, gen_morpho, update_meta=False)
                all_meta[name] = saved.get_meta()
                loaders.append(saved)
            # Finish morphologies by bulk saving their collective metadata
            mr.update_all_meta(all_meta)
            morphologies = MorphologySet(loaders, indices)
        if not isinstance(morphologies, MorphologySet) and morphologies is not None:
            morphologies = MorphologySet(loaders, morphologies)

        return morphologies, rotations

    def _has_mdistr(self):
        # This function checks if this distributor node has specified a morpho distributor
        return self.__class__.morphologies.is_dirty(self)

    def _has_rdistr(self):
        # This function checks if this distributor node has specified a rotation distributor
        return self.__class__.rotations.is_dirty(self)


__all__ = [
    "DistributionContext",
    "Distributor",
    "DistributorsNode",
    "ExplicitNoRotations",
    "Implicit",
    "ImplicitNoRotations",
    "MorphologyDistributor",
    "MorphologyGenerator",
    "RandomMorphologies",
    "RandomRotations",
    "RotationDistributor",
    "RoundRobinMorphologies",
    "VolumetricRotations",
]
