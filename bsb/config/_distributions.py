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
    """Name of the scipy.stats distribution function"""
    parameters: dict[str, typing.Any] = config.catch_all(type=types.any_())
    """Parameters to pass to the distribution"""

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
        """Draw n random samples from the distribution"""
        return self._distr.rvs(size=n)

    def definition_interval(self, epsilon=0):
        """
        Returns the `epsilon` and 1 - `epsilon` values of
        the distribution Percent point function.

        :param float epsilon: ratio of the interval to ignore
        """
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon must be between 0 and 1")
        return self._distr.ppf(epsilon), self._distr.ppf(1 - epsilon)

    def cdf(self, value):
        """
        Returns the result of the cumulative distribution function for `value`

        :param float value: value to evaluate
        """
        return self._distr.cdf(value)

    def sf(self, value):
        """
        Returns the result of the Survival function for `value`

        :param float value: value to evaluate
        """
        return self._distr.sf(value)

    def __getattr__(self, attr):
        if "_distr" not in self.__dict__:
            raise AttributeError("No underlying _distr found for distribution node.")
        return getattr(self._distr, attr)


class _ConstantDistribution:
    def __init__(self, const):
        self.const = const

    def rvs(self, size):
        return np.full(size, self.const, dtype=type(self.const))
