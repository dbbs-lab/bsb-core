"""
    An attrs-inspired class annotation system, but my A stands for amateuristic.
"""

from ._make import compile_init, make_get_node_name, make_cast, make_dictable, make_tree
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
        # it from our parent. If so we shouldn't update the parent's dictionary but copy
        # it and update it with ours.
        n_attrs = node_cls._config_attrs.copy()
        n_attrs.update(attrs)
        node_cls._config_attrs = n_attrs
    else:
        node_cls._config_attrs = attrs
    node_cls.__init__ = compile_init(node_cls, root=root)
    make_get_node_name(node_cls, root=root)
    make_cast(node_cls, dynamic=dynamic, pluggable=pluggable, root=root)
    make_tree(node_cls)
    make_dictable(node_cls)

    return node_cls


def dynamic(node_cls=None, attr_name="cls", classmap=None, auto_classmap=False, **kwargs):
    """
    Decorate a class to be castable to a dynamically configurable class using
    a class configuration attribute.

    *Example*: Register a required string attribute ``class`` (this is the default):

    .. code-block:: python

        @dynamic
        class Example:
            pass

    *Example*: Register a string attribute ``type`` with a default value
    'pkg.DefaultClass' as dynamic attribute:

    .. code-block:: python

        @dynamic(attr_name='type', required=False, default='pkg.DefaultClass')
        class Example:
            pass

    :param attr_name: Name under which to register the class attribute in the node.
    :type attr_name: str
    :param kwargs: All keyword arguments are passed to the constructor of the
      :func:`attribute <.config.attr>`.
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

    *Example*: If you want to use the :guilabel:`attr` to chose from all the installed
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


def reflist(reference, **kwargs):
    """
    Create a configuration reference list.
    """
    if "default" not in kwargs:
        kwargs["default"] = _list
        kwargs["call_default"] = True
    return ConfigurationReferenceListAttribute(reference, **kwargs)


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


def _hasattr(instance, name):
    return "_" + name in instance.__dict__


class ConfigurationAttribute:
    def __init__(
        self,
        type=None,
        default=None,
        call_default=None,
        required=False,
        key=False,
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
        try:
            value = self.type(value, _parent=instance, _key=key)
        except CastError:
            raise
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
            if self.should_call_default():
                t = _type(self.default())
            else:
                t = _type(self.default)
        else:
            t = type or str
        return _wrap_handler_pk(t)

    def get_node_name(self, instance):
        return instance.get_node_name() + "." + self.attr_name

    def tree(self, instance):
        val = _getattr(instance, self.attr_name)
        if hasattr(val, "__tree__"):
            val = val.__tree__()
        return val

    def should_call_default(self):
        cdf = self.call_default
        return cdf or (cdf is None and callable(self.default))


def _wrap_handler_pk(t):
    # Inspect the signature and wrap the typecast in a wrapper that will accept and
    # strip the missing 'key' kwarg
    try:
        sig = signature(t)
        params = sig.parameters
    except:
        params = []

    pass_key = "_key" in params
    pass_parent = "_parent" in params
    if pass_key and pass_parent:
        return t
    header = "def type_handler(value, *args, _parent=None, _key=None, **kwargs):\n"
    keypass = ", _key=_key" if pass_key else ""
    parentpass = ", _parent=_parent" if pass_parent else ""
    wrap = f" return orig(value, *args{parentpass}{keypass}, **kwargs)"
    exec(header + wrap, {"orig": t}, bait := locals())
    type_handler = bait["type_handler"]
    type_handler.__name__ = t.__name__
    return type_handler


class cfglist(_list):
    def get_node_name(self):
        return self._config_parent.get_node_name() + "." + self._config_attr_name

    @property
    def _config_attr_name(self):
        return self._config_attr.attr_name


class ConfigurationListAttribute(ConfigurationAttribute):
    def __init__(self, *args, size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size

    def __set__(self, instance, value, key=None):
        _setattr(instance, self.attr_name, self.__cast__(value, _parent=instance))

    def __cast__(self, value, parent, key=None):
        _cfglist = cfglist(value or _list())
        _cfglist._config_parent = parent
        _cfglist._config_attr = self
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
                _cfglist[i] = self.child_type(elem, _parent=_cfglist, __key=i)
                try:
                    _cfglist[i]._config_index = i
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

    def tree(self, instance):
        val = _getattr(instance, self.attr_name)
        return [e if not hasattr(e, "__tree__") else e.__tree__() for e in val]


class cfgdict(_dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                self.get_node_name() + " object has no attribute '{}'".format(name)
            )

    @property
    def _config_attr_name(self):
        return self._config_attr.attr_name

    def get_node_name(self):
        return self._config_parent.get_node_name() + "." + self._config_attr_name


class ConfigurationDictAttribute(ConfigurationAttribute):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __set__(self, instance, value, _key=None):
        _setattr(instance, self.attr_name, self.fill(value, _parent=instance))

    def fill(self, value, _parent, _key=None):
        _cfgdict = cfgdict(value or _dict())
        _cfgdict._config_parent = _parent
        _cfgdict._config_attr = self
        try:
            for ckey, value in _cfgdict.items():
                _cfgdict[ckey] = self.child_type(value, _parent=_cfgdict, _key=ckey)
        except CastError:
            raise
        except:
            raise CastError(
                "Couldn't cast {}.{} from '{}' into a {}".format(
                    self.get_node_name(_parent), ckey, value, self.child_type.__name__
                )
            )
        return _cfgdict

    def _get_type(self, type):
        self.child_type = super()._get_type(type)
        return self.fill

    def tree(self, instance):
        val = _getattr(instance, self.attr_name).items()
        return {k: v if not hasattr(v, "__tree__") else v.__tree__() for k, v in val}


class ConfigurationReferenceAttribute(ConfigurationAttribute):
    def __init__(
        self, reference, key=None, ref_type=None, populate=None, pop_unique=True, **kwargs
    ):
        self.ref_lambda = reference
        self.ref_key = key
        self.ref_type = ref_type
        self.populate = populate
        self.pop_unique = pop_unique
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
            if self.should_resolve_on_set(instance):
                if hasattr(instance, "_config_root"):  # pragma: nocover
                    raise ReferenceError(
                        "Can't autoresolve references without a config root."
                    )
                _setattr(
                    instance,
                    self.attr_name,
                    self.__ref__(instance, instance._config_root),
                )

    def is_reference_value(self, value):
        if value is None:
            return True
        if self.ref_type is not None:
            return isinstance(value, self.ref_type)
        elif hasattr(self.ref_lambda, "is_ref"):
            return self.ref_lambda.is_ref(value)
        else:
            return not isinstance(value, str)

    def should_resolve_on_set(self, instance):
        return (
            hasattr(instance, "_config_resolve_on_set")
            and instance._config_resolve_on_set
        )

    def __ref__(self, instance, root):
        try:
            remote, remote_key = self._prepare_self(instance, root)
        except NoReferenceAttributeSignal:
            return None
        return self.resolve_reference(instance, remote, remote_key)

    def _prepare_self(self, instance, root):
        instance._config_root = root
        instance._config_resolve_on_set = True
        remote = self.ref_lambda(root, instance)
        local_attr = self.get_ref_key()
        if not hasattr(instance, local_attr):
            raise NoReferenceAttributeSignal()
        return remote, getattr(instance, local_attr)

    def resolve_reference(self, instance, remote, key):
        if key not in remote:
            raise ReferenceError(
                "Reference '{}' of {} does not exist in {}".format(
                    key,
                    self.get_node_name(instance),
                    remote.get_node_name(),
                )
            )
        value = remote[key]
        if self.populate:
            self.populate_reference(instance, value)
        return value

    def populate_reference(self, instance, reference):
        # Remote descriptors can ask to handle populating itself by implementing a
        # __populate__ method. Here we check if the method exists and if so defer to it.
        if hasattr(reference.__class__, self.populate):
            pop_attr = getattr(reference.__class__, self.populate)
            if hasattr(pop_attr, "__populate__"):
                return pop_attr.__populate__(
                    reference, instance, unique_list=self.pop_unique
                )

        if (
            hasattr(reference, self.populate)
            and (population := getattr(reference, self.populate)) is not None
        ):
            if not self.pop_unique or instance not in population:
                population.append(instance)
        else:
            setattr(reference, self.populate, [instance])

    def tree(self, instance):
        val = getattr(instance, self.get_ref_key(), None)
        if self.is_reference_value(val) and hasattr(val, "_config_key"):
            val = val._config_key
        return val


class ConfigurationReferenceListAttribute(ConfigurationReferenceAttribute):
    def __set__(self, instance, value, key=None):
        if value is None:
            setattr(instance, self.get_ref_key(), [])
            _setattr(instance, self.attr_name, [])
            return
        try:
            remote_keys = _list(iter(value))
        except TypeError:
            raise ReferenceError(
                "Reference list '{}' of {} is not iterable.".format(
                    value, self.get_node_name(instance)
                )
            )
        # Store the referring values to the references key.
        setattr(instance, self.get_ref_key(), remote_keys)
        if self.should_resolve_on_set(instance):
            remote = self.ref_lambda(instance._config_root, instance)
            refs = self.resolve_reference_list(instance, remote, remote_keys)
            _setattr(instance, self.attr_name, refs)

    def get_ref_key(self):
        return self.ref_key or (self.attr_name + "_references")

    def __ref__(self, instance, root):
        try:
            remote, remote_keys = self._prepare_self(instance, root)
        except NoReferenceAttributeSignal:  # pragma: nocover
            return None
        if _hasattr(instance, self.attr_name):
            remote_keys.extend(_getattr(instance, self.attr_name))
            remote_keys = _list(set(remote_keys))
        return self.resolve_reference_list(instance, remote, remote_keys)

    def resolve_reference_list(self, instance, remote, remote_keys):
        refs = []
        for remote_key in remote_keys:
            if not self.is_reference_value(remote_key):
                reference = self.resolve_reference(instance, remote, remote_key)
            else:
                reference = remote_key
                # Usually resolve_reference also populates, but since we have our ref
                # already we skip it and should call populate_reference ourselves.
                if self.populate:
                    self.populate_reference(instance, reference)
            refs.append(reference)
        return refs

    def __populate__(self, instance, value, unique_list=False):
        has_refs = hasattr(instance, self.get_ref_key())
        has_pop = hasattr(instance, self.attr_name)
        if has_pop:
            population = getattr(instance, self.attr_name)
        if has_refs:
            references = getattr(instance, self.get_ref_key())
        is_new = (not has_pop or value not in population) and (
            not has_refs or value not in references
        )
        should_pop = has_pop and (not unique_list or is_new)
        should_ref = should_pop or not has_pop
        if should_pop:
            population.append(value)
        if should_ref:
            if not has_refs:
                setattr(instance, self.get_ref_key(), [value])
            else:
                references.append(value)

    def tree(self, instance):
        val = getattr(instance, self.get_ref_key(), [])
        val = [v._config_key if self._tree_should_unreference(v) else v for v in val]
        return val

    def _tree_should_unreference(self, value):
        return self.is_reference_value(value) and hasattr(value, "_config_key")


class ConfigurationAttributeSlot(ConfigurationAttribute):
    def __set__(self, instance, value):  # pragma: nocover
        raise NotImplementedError(
            "Configuration slot '{}' of {} is empty. The {} plugin provided by '{}' should fill the slot with a configuration attribute.".format(
                self.attr_name,
                instance.get_node_name(),
                instance.__class__._bsb_entry_point.module_name,
                instance.__class__._bsb_entry_point.dist,
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
        value = self.type(value, _parent=node, _key=key)
        # If succesfully cast, catch this value by executing our catch callback.
        self.caught(node, _getattr(node, self.attr_name), key, value)

    def tree(self, instance):
        val = _getattr(instance, self.attr_name).items()
        return {k: v if not hasattr(v, "__tree__") else v.__tree__() for k, v in val}
