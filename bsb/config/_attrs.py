"""
    An attrs-inspired class annotation system, but my A stands for amateuristic.
"""

from ._make import wrap_init, make_get_node_name, make_cast, make_dictable
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
    make_dictable(node_cls)

    return node_cls


def dynamic(
    node_cls=None, attr_name="class", classmap=None, auto_classmap=False, **kwargs
):
    """
        Decorate a class to be castable to a dynamically configurable class using
        a class configuration attribute

        Example
        -------

        Register a required string attribute 'class':

        .. code-block:: python

            @dynamic
            class Example:
                pass

        Register a string attribute 'type' with a default value 'pkg.DefaultClass':

        .. code-block:: python

            @dynamic(attr_name='type', required=False, default='pkg.DefaultClass')
            class Example:
                pass

        :param attr_name: Name under which to register the class attribute in the node.
        :type attr_name: str
        :param kwargs: All keyword arguments are passed to the constructor of the
          :class:`attribute <config._attrs.ConfigurationAttribute>`.
    """
    if "required" not in kwargs:
        kwargs["required"] = True
    if "type" not in kwargs:
        kwargs["type"] = str
    class_attr = ConfigurationAttribute(**kwargs)
    dynamic_config = DynamicNodeConfiguration(classmap, auto_classmap)
    if node_cls is None:
        # If node_cls is None, it means that no positional argument was given, which most
        # likely means that the @dynamic(...) syntax was used instead of the @dynamic.
        # This means we have to return an inner decorator instead of the decorated class
        def decorator(node_cls):
            return _dynamic(node_cls, class_attr, attr_name, dynamic_config)

        return decorator
    # Regular @dynamic syntax used, return decorated class
    return _dynamic(node_cls, class_attr, attr_name, dynamic_config)


class DynamicNodeConfiguration:
    def __init__(self, classmap=None, auto_classmap=False):
        self.classmap = classmap
        self.auto_classmap = auto_classmap


def _dynamic(node_cls, class_attr, attr_name, config):
    setattr(node_cls, attr_name, class_attr)
    node_cls._config_dynamic_attr = attr_name
    return node(node_cls, dynamic=config)


def pluggable(key, plugin_name=None, unpack=None):
    """
        Create a node whose configuration is defined by a plugin.

        Example
        -------

        If you want to use the :guilabel:`attr` to chose from all the installed
        `dbbs_scaffold.my_plugin` plugins:

        .. code-block:: python

            @pluggable('attr', 'my_plugin')
            class PluginNode:
                pass

        This will then read :guilabel:`attr`, load the plugin and configure the node from
        the node class specified by the plugin.

        :param plugin_name: The name of the category of the plugin endpoint
        :type plugin_name: str
        :param unpack: Optional callable to get the desired node out of the plugin, by
          default the plugin should be the node class itself.
        :type unpack: callable
    """

    def inner_decorator(node_cls):
        node_cls._config_plugin_name = plugin_name
        node_cls._config_plugin_key = key
        node_cls._config_plugin_unpack = unpack
        class_attr = ConfigurationAttribute(type=str, required=True)
        setattr(node_cls, key, class_attr)
        return node(node_cls, pluggable=True)

    return inner_decorator


def attr(**kwargs):
    """
        Create a configuration attribute.

        Only works when used inside of a class decorated with the :func:`node
        <.config.node>`, :func:`dynamic <.config.dynamic>`,  :func:`root <.config.root>`
        or  :func:`pluggable <.config.pluggable>` decorators.

        :param type: Type of the attribute's value.
        :type type: callable
        :param required: Should an error be thrown if the attribute is not present?
        :type required: bool
        :param default: Default value.
        :type default: any
        :param call_default: Should the default value be used (False) or called (True).
          Useful for default values that should not be shared among objects.
        :type call_default: bool
        :param key: If True the key under which the parent of this attribute appears in
          its parent is stored on this attribute. Useful to store for example the name of
          a node appearing in a dict
    """
    return ConfigurationAttribute(**kwargs)


def ref(reference, **kwargs):
    """
        Create a configuration reference.

        Configuration references are attributes that transform their value into the value
        of another node or value in the document::

          {
            "keys": {
                "a": 3,
                "b": 5
            },
            "simple_ref": "a"
          }

        With ``simple_ref = config.ref(lambda root, here: here["keys"])`` the value ``a``
        will be looked up in the configuration object (after all values have been cast) at
        the location specified by the callable first argument.
    """
    return ConfigurationReferenceAttribute(reference, **kwargs)


def slot(**kwargs):
    """
        Create an attribute slot that is required to be overriden by child or plugin
        classes.
    """
    return ConfigurationAttributeSlot(**kwargs)


_list = list
_dict = dict
_type = type


def list(**kwargs):
    """
        Create a configuration attribute that holds a list of configuration values.
        Best used only for configuration nodes. Use an :func:`attr` in combination with a
        :func:`types.list <.config.types.list>` type for simple values.
    """
    return ConfigurationListAttribute(**kwargs)


def dict(**kwargs):
    """
        Create a configuration attribute that holds a key value pairs of configuration
        values. Best used only for configuration nodes. Use an :func:`attr` in combination
        with a :func:`types.dict <.config.types.dict>` type for simple values.
    """
    return ConfigurationDictAttribute(**kwargs)


def catch_all(**kwargs):
    """
        Catches any unknown key with a value that can be cast to the given type and
        collects them under the attribute name.
    """
    return ConfigurationAttributeCatcher(**kwargs)


def _setattr(instance, name, value):
    instance.__dict__["_" + name] = value


def _getattr(instance, name):
    return instance.__dict__["_" + name]


class ConfigurationAttribute:
    def __init__(
        self, type=None, default=None, call_default=False, required=False, key=False,
    ):
        if not callable(required):
            self.required = lambda s: required
        else:
            self.required = required
        self.key = key
        self.default = default
        self.call_default = call_default
        self.type = self._get_type(type)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return _getattr(instance, self.attr_name)

    def __set__(self, instance, value, key=None):
        if value is None:
            # Don't cast None to a value of the attribute type.
            return _setattr(instance, self.attr_name, None)
        if self.type.__casting__:
            # If the `__casting__` flag is set, the type casting method wants to be
            # responsible for exception handling and we should not handle them here.
            value = self.type(value, parent=instance, key=key)
        else:
            # The `__casting__` flag is not set so we're responsible to catch exceptions.
            try:
                value = self.type(value, parent=instance, key=key)
            except:
                raise CastError(
                    "Couldn't cast {} from '{}' into {}".format(
                        self.get_node_name(instance), value, self.type.__name__
                    )
                )
        # The value was cast to its intented type and the new value can be set.
        _setattr(instance, self.attr_name, value)

    def _get_type(self, type):
        # Determine type of the attribute
        if not type and self.default:
            t = _type(self.default)
        else:
            t = type or str
        return _wrap_handler_pk(t)

    def get_node_name(self, instance):
        return instance.get_node_name() + "." + self.attr_name


def _wrap_handler_pk(t):
    cast_name = t.__name__
    casting = hasattr(t, "__casting__") and t.__casting__
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
        _cfglist._config_parent = parent
        _cfglist._attr = self
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
                try:
                    _cfglist[i]._index = i
                except:
                    pass
        except:
            if self.child_type.__casting__:
                raise
            raise CastError(
                "Couldn't cast {}[{}] from '{}' into a {}".format(
                    self.get_node_name(parent), i, elem, self.child_type.__name__
                )
            )
        return _cfglist

    def _get_type(self, type):
        self.child_type = super()._get_type(type)
        return self.__cast__


class cfgdict(_dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                self.get_node_name() + " object has no attribute '{}'".format(name)
            )

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
    def __init__(self, reference, key=None, ref_type=None, populate=None, **kwargs):
        self.ref_lambda = reference
        self.ref_key = key
        self.ref_type = ref_type
        self.populate = populate
        self.resolve_on_set = False
        self.root = None
        # No need to cast to any types: the reference we fetch will already have been cast
        if "type" in kwargs:  # pragma: nocover
            del kwargs["type"]
        super().__init__(**kwargs)

    def get_ref_key(self):
        return self.ref_key or (self.attr_name + "_reference")

    def __set__(self, instance, value, key=None):
        if self.is_reference_value(value):
            _setattr(instance, self.attr_name, value)
        else:
            setattr(instance, self.get_ref_key(), value)
            if self.resolve_on_set:
                if not self.root:  # pragma: nocover
                    raise ReferenceError(
                        "Can't autoresolve references without a config root."
                    )
                _setattr(instance, self.attr_name, self.__ref__(instance, self.root))

    def is_reference_value(self, value):
        if value is None:
            return True
        if self.ref_type is not None:
            return isinstance(value, self.ref_type)
        elif hasattr(self.ref_lambda, "is_ref"):
            return self.ref_lambda.is_ref(value)
        else:
            return not isinstance(value, str)

    def __ref__(self, instance, root):
        self.root = root
        self.resolve_on_set = True
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
        reference = reference_parent[reference_key]
        if self.populate:
            if hasattr(reference, self.populate):
                getattr(reference, self.populate).append(instance)
            else:
                setattr(reference, self.populate, [instance])
        return reference


class ConfigurationAttributeSlot(ConfigurationAttribute):
    def __set__(self, instance, value):  # pragma: nocover
        raise NotImplementedError(
            "Configuration slot '{}' of {} is empty. The {} plugin provided by '{}' should fill the slot with a configuration attribute.".format(
                self.attr_name,
                instance.get_node_name(),
                instance.__class__._scaffold_plugin.module_name,
                instance.__class__._scaffold_plugin.dist,
            )
        )


def _collect_kv(n, d, k, v):
    d[k] = v


class ConfigurationAttributeCatcher(ConfigurationAttribute):
    def __init__(self, *args, type=str, initial=_dict, catch=_collect_kv, **kwargs):
        super().__init__(*args, type=type, default=initial, call_default=True, **kwargs)
        self.caught = catch

    def __set__(self, instance, value):
        _setattr(instance, self.attr_name, value)

    def __catch__(self, node, key, value):
        # Try to cast to our type, if it fails it will be caught by whoever is asking us
        # to catch this and know we don't catch this value.
        value = self.type(value, parent=node, key=key)
        # If succesfully cast, catch this value by executing our catch callback.
        self.caught(node, _getattr(node, self.attr_name), key, value)
