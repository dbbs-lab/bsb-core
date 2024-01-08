import typing

import errr
import numpy as np
import scipy.stats.distributions as _distributions

from .. import config
from ..exceptions import DistributionCastError
from . import types

if typing.TYPE_CHECKING:
    from ..core import Scaffold

# Scan the scipy distributions module for all distribution names. Ignore `_gen` which are
# the factory functions for the distribution classes. `rvs` is a duck type check.
_available_distributions = [
    d
    for d, v in _distributions.__dict__.items()
    if hasattr(v, "rvs") and not d.endswith("_gen")
]
_available_distributions.append("constant")


@config.node
class Distribution:
    scaffold: "Scaffold"
    distribution: str = config.attr(
        type=types.in_(_available_distributions), required=True
    )
    parameters: dict[str, typing.Any] = config.catch_all(type=types.any_())

    def __init__(self, **kwargs):
        if self.distribution == "constant":
            self._distr = _ConstantDistribution(self.parameters["constant"])
            return

        try:
            self._distr = getattr(_distributions, self.distribution)(**self.parameters)
        except Exception as e:
            errr.wrap(
                DistributionCastError, e, prepend=f"Can't cast to '{self.distribution}': "
            )

    def draw(self, n):
        return self._distr.rvs(size=n)

    def __getattr__(self, attr):
        if "_distr" not in self.__dict__:
            raise AttributeError("No underlying _distr found for distribution node.")
        return getattr(self._distr, attr)


class _ConstantDistribution:
    def __init__(self, const):
        self.const = const

    def rvs(self, size):
        return np.full(size, self.const, dtype=type(self.const))
