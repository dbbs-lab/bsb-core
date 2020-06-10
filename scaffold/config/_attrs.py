"""
    An attrs-inspired class annotation system, but my A stands for amateuristic.
"""

from ._make import wrap_init, make_get_node_name, make_cast
from inspect import signature
from ..exceptions import *


def root(root_cls):
    """
        Decorate a class as a configuration root node.
    """
    root_cls.attr_name = root_cls.node_name = r"{root}"
    node(root_cls, root=True)

    return root_cls


def node(node_cls, root=False, dynamic=False, pluggable=False):
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
        # If _config_attrs is already present on the class it's possible that we inherited
        # it from our parent. If so we shouldn't update the parent's dictionary but create
        # a new one and update it with our parent's and then ours.
        n_attrs = {}
        n_attrs.update(node_cls._config_attrs)
        n_attrs.update(attrs)
        node_cls._config_attrs = n_attrs
    else:
        node_cls._config_attrs = attrs
    wrap_init(node_cls)
    make_get_node_name(node_cls, root=root)
    make_cast(node_cls, dynamic=dynamic, pluggable=pluggable, root=root)

    return node_cls


def dynamic(node_cls):
    class_attr = ConfigurationAttribute(type=str, required=True)
    setattr(node_cls, "class", class_attr)
    return node(node_cls, dynamic=True)


def pluggable(key, plugin_name=None, unpack=None):
    def inner_decorator(node_cls):
        node_cls._config_plugin_name = plugin_name
        node_cls._config_plugin_key = key
        node_cls._config_plugin_unpack = unpack
        class_attr = ConfigurationAttribute(type=str, required=True)
        setattr(node_cls, key, class_attr)
        return node(node_cls, pluggable=True)

    return inner_decorator


def attr(**kwargs):
    return ConfigurationAttribute(**kwargs)


def ref(reference, **kwargs):
    return ConfigurationReferenceAttribute(reference, **kwargs)


def slot(**kwargs):
    return ConfigurationAttributeSlot(**kwargs)


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
        self,
        type=None,
        default=None,
        call_default=False,
        required=False,
        key=False,
        validation=None,
    ):
        self.required = required
        self.key = key
        self.default = default
        self.call_default = call_default
        self.type = self._get_type(type)
        self.early_validator = validation

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return _getattr(instance, self.attr_name)

    def __set__(self, instance, value, key=None):
        if value is None:
            # Don't cast None to a value of the attribute type.
            return _setattr(instance, self.attr_name, None)
        if self.type.__casting__:
            value = self.type(value, parent=instance, key=key)
        else:
            try:
                value = self.type(value, parent=instance, key=key)
            except:
                raise CastError(
                    "Couldn't cast {} from '{}' into {}".format(
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


class cfglist(_list):
    def get_node_name(self):
        return self._config_parent.get_node_name() + "." + self._attr.attr_name


class ConfigurationListAttribute(ConfigurationAttribute):
    def __init__(self, *args, size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size

    def __set__(self, instance, value, key=None):
        _setattr(instance, self.attr_name, self.__cast__(value, parent=instance))

    def __cast__(self, value, parent, key=None):
        _cfglist = cfglist(value or _list())
        if value is None:
            return _cfglist
        if self.size is not None and len(_cfglist) != self.size:
            raise CastError(
                "Couldn't cast {} in {} into a {}-element list.".format(
                    value, self.get_node_name(parent), self.size
                )
            )
        try:
            for i, elem in enumerate(_cfglist):
                _cfglist[i] = self.child_type(elem, parent=_cfglist, key=i)
                _cfglist[i]._index = i
        except:
            raise CastError(
                "Couldn't cast {}[{}] from '{}' into a {}".format(
                    self.get_node_name(parent), i, elem, self.type.__name__
                )
            )
        return _cfglist

    def _get_type(self, type):
        self.child_type = super()._get_type(type)
        return self.__cast__


class cfgdict(_dict):
    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(name)
        return self.get(name)

    def get_node_name(self):
        return self._config_parent.get_node_name() + "." + self._attr.attr_name


class ConfigurationDictAttribute(ConfigurationAttribute):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __set__(self, instance, value, key=None):
        _setattr(instance, self.attr_name, self.__cast__(value, parent=instance))

    def __cast__(self, value, parent, key=None):
        _cfgdict = cfgdict(value or _dict())
        _cfgdict._config_parent = parent
        _cfgdict._attr = self
        try:
            for ckey, value in _cfgdict.items():
                _cfgdict[ckey] = self.child_type(value, parent=_cfgdict, key=ckey)
        except:
            if self.child_type.__casting__:
                raise
            raise CastError(
                "Couldn't cast {}.{} from '{}' into a {}".format(
                    self.get_node_name(parent), ckey, value, self.child_type.__name__
                )
            )
        return _cfgdict

    def _get_type(self, type):
        self.child_type = super()._get_type(type)
        return self.__cast__


class ConfigurationReferenceAttribute(ConfigurationAttribute):
    def __init__(self, reference, key=None, **kwargs):
        self.ref_lambda = reference
        self.ref_key = key
        # No need to cast to any types: the reference we fetch will already have been cast
        if "type" in kwargs:
            del kwargs["type"]
        super().__init__(**kwargs)

    def get_ref_key(self):
        return self.ref_key or (self.attr_name + "_reference")

    def __set__(self, instance, value, key=None):
        if value is None:
            _setattr(instance, self.attr_name, None)
        if isinstance(value, str):
            setattr(instance, self.get_ref_key(), value)
        else:
            _setattr(instance, self.attr_name, value)

    def __ref__(self, instance, root):
        reference_parent = self.ref_lambda(root, instance)
        reference_attr = self.get_ref_key()
        if not hasattr(instance, reference_attr):
            return None
        reference_key = getattr(instance, reference_attr)
        if reference_key not in reference_parent:
            raise ReferenceError(
                "Reference '{}' of {} does not exist in {}".format(
                    reference_key,
                    self.get_node_name(instance),
                    reference_parent.get_node_name(),
                )
            )
        return reference_parent[getattr(instance, reference_attr)]


class ConfigurationAttributeSlot(ConfigurationAttribute):
    def __set__(self, instance, value):
        raise NotImplementedError(
            "Configuration slot {} of {} is empty. The {} plugin provided by '{}' should fill it with a configuration attribute.".format(
                self.attr_name,
                instance.get_node_name(),
                instance.__class__._scaffold_plugin.module_name,
                instance.__class__._scaffold_plugin.dist,
            )
        )
