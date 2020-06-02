"""
    An attrs-inspired class annotation system, but my A stands for amateuristic.
"""

from ._make import wrap_init, make_get_node_name, make_cast
from inspect import signature
from ..exceptions import CastError


def root(root_cls):
    """
        Decorate a class as a configuration root node.
    """
    root_cls.attr_name = root_cls.node_name = r"{root}"
    node(root_cls, root=True)

    return root_cls


def node(node_cls, root=False, dynamic=False):
    """
        Decorate a class as a configuration node.
    """
    attrs = {
        k: v
        for k, v in node_cls.__dict__.items()
        if isinstance(v, ConfigurationAttribute)
    }
    # Give the attributes the name they were assigned in the class
    for name, attr in attrs.items():
        attr.attr_name = name

    if hasattr(node_cls, "_config_attrs"):
        node_cls._config_attrs.update(attrs)
    else:
        node_cls._config_attrs = attrs
    wrap_init(node_cls)
    make_get_node_name(node_cls, root=root)
    make_cast(node_cls, dynamic=dynamic, root=root)

    return node_cls


def dynamic(node_cls):
    class_attr = ConfigurationAttribute(type=str, required=True)
    setattr(node_cls, "class", class_attr)
    return node(node_cls, dynamic=True)


def attr(**kwargs):
    return ConfigurationAttribute(**kwargs)


def ref(reference, **kwargs):
    return ConfigurationReferenceAttribute(reference, **kwargs)


_list = list
_dict = dict
_type = type


def list(**kwargs):
    return ConfigurationListAttribute(**kwargs)


def dict(**kwargs):
    return ConfigurationDictAttribute(**kwargs)


def _setattr(instance, name, value):
    instance.__dict__["_" + name] = value


def _getattr(instance, name):
    return instance.__dict__["_" + name]


class ConfigurationAttribute:
    def __init__(
        self, type=None, default=None, call_default=False, required=False, key=False
    ):
        self.required = required
        self.key = key
        self.default = default
        self.call_default = call_default
        self.type = self._get_type(type)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return _getattr(instance, self.attr_name)

    def __set__(self, instance, value):
        if value is None:
            # Don't cast None to a value of the attribute type.
            return _setattr(instance, self.attr_name, None)
        if self.type.__casting__:
            value = self.type(value, parent=instance)
        else:
            try:
                value = self.type(value, parent=instance)
            except:
                raise CastError(
                    "Couldn't cast {} from '{}' into a {}".format(
                        self.get_node_name(instance), value, self.type.__name__
                    )
                )
        _setattr(instance, self.attr_name, value)

    def _get_type(self, type):
        cast_name = None
        casting = False
        # Determine type of the attribute
        if not type and self.default:
            t = _type(self.default)
        else:
            t = type or str
        cast_name = t.__name__
        if hasattr(t, "__cast__"):
            t = t.__cast__
            casting = True
        # Inspect the signature and wrap the typecast in a wrapper that will accept and
        # strip the missing 'key' kwarg
        try:
            sig = signature(t)
            params = sig.parameters
        except:
            params = []
        if "key" not in params:
            o = t

            def _t(*args, **kwargs):
                if "key" in kwargs:
                    del kwargs["key"]
                return o(*args, **kwargs)

            t = _t
        if "parent" not in params:
            o2 = t

            def _t2(value, parent, *args, **kwargs):
                return o2(value, *args, **kwargs)

            t = _t2
        t.__name__ = cast_name
        t.__casting__ = casting
        return t

    def get_node_name(self, instance):
        return instance.get_node_name() + "." + self.attr_name


class ConfigurationListAttribute(ConfigurationAttribute):
    def __set__(self, instance, value):
        # Trigger a TypeError outside of the CastError block
        _iter = iter(value)
        try:
            for i, elem in enumerate(_iter):
                value[i] = self.type(elem, key=i)
        except:
            raise CastError(
                "Couldn't cast {}[{}] from '{}' into a {}".format(
                    self.get_node_name(), i, value, self.type.__name__
                )
            )
        _setattr(instance, self.attr_name, value)


class cfgdict(_dict):
    def __getattr__(self, name):
        if name not in self:
            raise KeyError(name)
        return self.get(name)

    def get_node_name(self):
        return self._config_parent.get_node_name() + "." + self._attr_name


class ConfigurationDictAttribute(ConfigurationAttribute):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        print("SETTING DICT", value)
        _setattr(instance, self.attr_name, self.__cast__(value, parent=instance))

    def __cast__(self, value, parent, key=None):
        _cfgdict = cfgdict(value or _dict())
        _cfgdict._config_parent = parent
        _cfgdict._attr_name = self.attr_name
        try:
            for key, value in _cfgdict.items():
                _cfgdict[key] = self.child_type(value, parent=_cfgdict, key=key)
        except:
            if self.child_type.__casting__:
                raise
            raise CastError(
                "Couldn't cast {}.{} from '{}' into a {}".format(
                    self.get_node_name(parent), key, value, self.child_type.__name__
                )
            )
        return _cfgdict

    def _get_type(self, type):
        self.child_type = super()._get_type(type)
        return self.__cast__


class ConfigurationReferenceAttribute(ConfigurationAttribute):
    def __init__(self, reference, **kwargs):
        self.ref_lambda = reference
        # No need to cast to any types: the reference we fetch will already have been cast
        if "type" in kwargs:
            del kwargs["type"]
        super().__init__(**kwargs)

    def __set__(self, instance, value):
        if value is None:
            _setattr(instance, self.attr_name, None)
        if isinstance(value, str):
            setattr(instance, self.attr_name + "_reference", value)
        else:
            _setattr(instance, self.attr_name, value)

    def fetch_reference(self, instance, root):
        reference_parent = self.ref_lambda(root, instance)
        return reference_parent[getattr(instance, self.attr_name)]
