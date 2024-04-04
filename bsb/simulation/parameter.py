from .. import config


@config.dynamic(attr_name="type", auto_classmap=True, required=False)
class ParameterValue:
    def __init__(self, value=None, /, **kwargs):
        self._constant = value


@config.dynamic(attr_name="type", auto_classmap=True, required=False)
class Parameter:
    value: ParameterValue = config.attr(type=ParameterValue)


__all__ = ["Parameter", "ParameterValue"]
