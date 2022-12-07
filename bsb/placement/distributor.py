from .. import config
from ..topology.partition import Partition
from ..exceptions import EmptySelectionError
from ..profiling import node_meter
from ..morphologies import MorphologySet
from .indicator import PlacementIndications
from dataclasses import dataclass
import numpy as np
import abc
import uuid
from typing import List


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
class DistributorsNode:
    morphologies = config.attr(
        type=MorphologyDistributor, default=dict, call_default=True
    )
    rotations = config.attr(type=RotationDistributor, default=dict, call_default=True)
    properties = config.catch_all(type=Distributor)

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
                f"Given {len(sel)} selectors: did not find any suitable morphologies",
                sel,
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
            for gen_morpho, i in generated.items():
                name = f"{prefix}-{uid}-{i}"
                saved = mr.save(name, gen_morpho)
                loaders.append(saved)
            morphologies = MorphologySet(loaders, indices)
        if not isinstance(morphologies, MorphologySet) and morphologies is not None:
            morphologies = MorphologySet(loaders, morphologies)

        return morphologies, rotations

    def _has_mdistr(self):
        # This function checks if this distributor node has specified a morpho distributor
        return self.__class__.morphologies.is_dirty(self)
