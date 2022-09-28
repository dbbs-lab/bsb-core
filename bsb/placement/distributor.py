from .. import config
from ..exceptions import EmptySelectionError
from ..morphologies import MorphologySet
import numpy as np
import abc


@config.dynamic(attr_name="strategy", required=True)
class Distributor(abc.ABC):
    @abc.abstractmethod
    def distribute(self, partitions, indicator, positions):
        """
        Is called to distribute cell properties.

        :param partitions: The partitions the cells were placed in.
        :type partitions: List[~bsb.topology.partition.Partition]
        :param indicator: The indicator of the cell type whose properties are being
          distributed.
        :param positions: Positions of the cells.
        :type positions: numpy.ndarray
        :returns: An array with the property data
        :rtype: numpy.ndarray
        """
        pass


@config.dynamic(
    attr_name="strategy", required=False, default="random", auto_classmap=True
)
class MorphologyDistributor(Distributor):
    @abc.abstractmethod
    def distribute(self, partitions, indicator, positions):
        """
        Is called to distribute cell morphologies and optionally rotations.

        :param partitions: The partitions the morphologies need to be distributed in.
        :type partitions: List[~bsb.topology.partition.Partition]
        :param indicator: The indicator of the cell type whose morphologies are being
          distributed.
        :param positions: Placed positions under consideration
        :type positions: numpy.ndarray
        :returns: A MorphologySet with assigned morphologies, and optionally a RotationSet
        :rtype: Union[~bsb.morphologies.MorphologySet, Tuple[
          ~bsb.morphologies.MorphologySet, ~bsb.morphologies.RotationSet]]
        """
        pass


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

    def distribute(self, partitions, indicator, positions):
        """
        Uses the morphology selection indicators to select morphologies and
        returns a MorphologySet of randomly assigned morphologies
        """
        sel = indicator.assert_indication("morphologies")
        loaders = self.scaffold.storage.morphologies.select(*sel)
        if not loaders:
            raise EmptySelectionError(
                f"Given {len(sel)} selectors: did not find any suitable morphologies",
                sel,
            )
        else:
            ids = np.random.default_rng().integers(len(loaders), size=len(positions))
        return MorphologySet(loaders, ids)


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

    def distribute(self, partitions, indicator, positions):
        sel = indicator.assert_indication("morphologies")
        loaders = self.scaffold.storage.morphologies.select(*sel)
        if not loaders:
            raise EmptySelectionError(
                f"Given {len(sel)} selectors: did not find any suitable morphologies",
                sel,
            )
        else:
            ll = len(loaders)
            lp = len(positions)
            ids = np.tile(np.arange(ll), lp // ll + 1)[:lp]
        return MorphologySet(loaders, ids)


@config.dynamic(attr_name="strategy", required=False, default="none", auto_classmap=True)
class RotationDistributor(Distributor):
    """
    Rotates everything by nothing!
    """

    @abc.abstractmethod
    def distribute(self, partitions, indicator, positions):
        pass


class ExplicitNoRotations(RotationDistributor, classmap_entry="explicitly_none"):
    def distribute(self, partitions, indicator, positions):
        return np.zeros((len(positions), 3))


# Sentinel mixin class to tag RotationDistributor as being overridable by morphology
# distributor.
class Implicit:
    pass


class ImplicitNoRotations(ExplicitNoRotations, Implicit, classmap_entry="none"):
    pass


class RandomRotations(RotationDistributor, classmap_entry="random"):
    def distribute(self, partitions, indicator, positions):
        return np.random.rand(len(positions), 3) * 360


@config.node
class DistributorsNode:
    morphologies = config.attr(
        type=MorphologyDistributor, default=dict, call_default=True
    )
    rotations = config.attr(type=RotationDistributor, default=dict, call_default=True)
    properties = config.catch_all(type=Distributor)

    def _curry(self, partitions, indicator, positions):
        def curried(key):
            return self(key, partitions, indicator, positions)

        return curried

    def __call__(self, key, partitions, indicator, positions):
        if key in ("morphologies", "rotations"):
            distributor = getattr(self, key)
        else:
            distributor = self.properties[key]
        return distributor.distribute(partitions, indicator, positions)
