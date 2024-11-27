"""
    An attrs-inspired class annotation system, but my A stands for amateuristic.
"""

import builtins
from functools import wraps

import errr

from ..exceptions import (
    BootError,
    CastError,
    CfgReferenceError,
    NoReferenceAttributeSignal,
    RequirementError,
)
from ._compile import _wrap_reserved
from ._hooks import run_hook
from ._make import (
    MISSING,
    _resolve_references,
    compile_class,
    compile_isc,
    compile_new,
    compile_postnew,
    make_copyable,
    make_dictable,
    make_get_node_name,
    make_tree,
    walk_nodes,
    wrap_root_postnew,
)


def root(root_cls):
    """
    Decorate a class as a configuration root node.
    """
    root_cls.attr_name = root_cls.node_name = r"{root}"
    root_cls._config_isroot = True
    return node(root_cls, root=True)


def node(node_cls, root=False, dynamic=False, pluggable=False):
    """
    Decorate a class as a configuration node.
    """
    # Recreate the class to set its metaclass a posteriori
    node_cls = compile_class(node_cls)
    node_cls._config_unset = []
    # Inherit the parent's attributes, if any exist on the class already
    attrs = getattr(node_cls, "_config_attrs", {}).copy()
    for k, v in builtins.dict(node_cls.__dict__).items():
        # Add our attributes
        if isinstance(v, ConfigurationAttribute):
            if v.unset:
                attrs.pop(k, None)
                delattr(node_cls, k)
                # Keep track of what this class wants to unset, in case of MRO traversal.
                node_cls._config_unset.append(k)
            else:
                attrs[k] = v
    node_cls._config_attrs = attrs
    node_cls.__post_new__ = compile_postnew(node_cls)
    node_cls._config_isroot = root
    if root:
        node_cls.__post_new__ = wrap_root_postnew(node_cls.__post_new__)
        node_cls._config_isbooted = False
    node_cls.__new__ = compile_new(
        node_cls, dynamic=dynamic, pluggable=pluggable, root=root
    )
    if dynamic:
        node_cls.__init_subclass__ = compile_isc(node_cls, dynamic)
    make_get_node_name(node_cls, root=root)
    make_tree(node_cls)
    make_dictable(node_cls)
    make_copyable(node_cls)

    return node_cls


def dynamic(
    node_cls=None,
    attr_name="cls",
    classmap=None,
    auto_classmap=False,
    classmap_entry=None,
    **kwargs,
):
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
    if "required" not in kwargs and "default" not in kwargs:
        kwargs["required"] = True
    if "type" not in kwargs:
        kwargs["type"] = str
    class_attr = ConfigurationAttribute(**kwargs)
    dynamic_config = DynamicNodeConfiguration(classmap, auto_classmap, classmap_entry)
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
    def __init__(self, classmap=None, auto_classmap=False, entry=None):
        self.classmap = classmap
        self.auto_classmap = auto_classmap
        self.entry = entry


def _dynamic(node_cls, class_attr, attr_name, config):
    # Set the dynamic attribute
    setattr(node_cls, attr_name, class_attr)
    node_cls._config_dynamic_attr = attr_name
    # Other than that compile the dynamic class like a regular node class
    node_cls = node(node_cls, dynamic=config)

    if config.auto_classmap or config.classmap:
        node_cls._config_dynamic_classmap = config.classmap or {}
    # This adds the parent class to its own classmap, which for subclasses happens in init
    # subclass
    if config.entry is not None:
        if not hasattr(node_cls, "_config_dynamic_classmap"):
            raise ValueError(
                f"Calling `@config.dynamic` with `entry='{config.entry}'`"
                + " requires `classmap` or `auto_classmap` to be set as well"
                + f" on '{node_cls.__name__}'."
            )
        node_cls._config_dynamic_classmap[config.entry] = node_cls
    # Mark the class as its own dynamic root, (grand)child classes will all need to
    # inherit from this as an interface contract.
    node_cls._config_dynamic_root = node_cls
    return node_cls


def pluggable(key, plugin_name=None):
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
    """

    def inner_decorator(node_cls):
        node_cls._config_plugin_name = plugin_name
        node_cls._config_plugin_key = key
        class_attr = ConfigurationAttribute(type=str, required=True)
        setattr(node_cls, key, class_attr)
        return node(node_cls, pluggable=True)

    return inner_decorator


def attr(**kwargs):
    """
    Create a configuration attribute.

    Only works when used inside a class decorated with the :func:`node
    <.config.node>`, :func:`dynamic <.config.dynamic>`,  :func:`root <.config.root>`
    or  :func:`pluggable <.config.pluggable>` decorators.

    :param type: Type of the attribute's value.
    :type type: Callable
    :param required: Should an error be thrown if the attribute is not present?
    :type required: bool
    :param default: Default value.
    :type default: Any
    :param call_default: Should the default value be used (False) or called (True).
      Use this to prevent mutable default values.
    :type call_default: bool
    :param key: If set, the key of the parent is stored on this attribute.
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
        kwargs["default"] = builtins.list
        kwargs["call_default"] = True
    return ConfigurationReferenceListAttribute(reference, **kwargs)


def slot(**kwargs):
    """
    Create an attribute slot that is required to be overriden by child or plugin
    classes.
    """
    return ConfigurationAttributeSlot(**kwargs)


def property(val=None, /, type=None, **kwargs):
    """
    Create a configuration property attribute. You may provide a value or a callable. Call
    `setter` on the return value as you would with a regular property.
    """
    if type is None:
        type = lambda v: v

    def decorator(val):
        prop = val if callable(val) else lambda s: val
        return ConfigurationProperty(prop, type=type, **kwargs)

    if val is None:
        return decorator
    else:
        return decorator(val)


def provide(value):
    """
    Provide a value for a parent class' attribute. Can be a value or a callable, a
    readonly configuration property will be created from it either way.
    """
    prop = property(value)

    def provided(self, instance, value):
        raise AttributeError(f"Can't set attribute, class provides the value '{value}'.")

    # Create a callable object that invokes `provided` when called, and whose `bool()`
    # returns `False`. Later in `_is_settable_attr`, we use this to trick the short
    # circuiting logic, so that this setter doesn't make the internal logic set and error
    # out on this attr.
    prop.setter(
        type("provision", (), {"__call__": provided, "__bool__": lambda s: False})()
    )

    return prop


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


def unset():
    """
    Override and unset an inherited configuration attribute.
    """
    return ConfigurationAttribute(unset=True)


def file(**kwargs):
    """
    Create a file dependency attribute.
    """
    from ..storage._files import FileDependencyNode

    kwargs.setdefault("type", FileDependencyNode)
    return attr(**kwargs)


def _setattr(instance, name, value):
    instance.__dict__["_" + name] = value


def _getattr(instance, name):
    try:
        return instance.__dict__["_" + name]
    except KeyError as e:
        instance.__getattribute__(e.args[0])


def _hasattr(instance, name):
    return "_" + name in instance.__dict__


def _get_root(obj):
    parent = obj
    while hasattr(parent, "_config_parent") and parent._config_parent is not None:
        parent = parent._config_parent
    return parent


def _strict_root(obj):
    root = _get_root(obj)
    return root if getattr(root, "_config_isroot", False) else None


def _booted_root(obj):
    root = _strict_root(obj)
    return root if root and root._config_isfinished and root._config_isbooted else None


def _is_booted(obj):
    return obj and obj._config_isbooted


def _root_is_booted(obj):
    root = _strict_root(obj)
    return _is_booted(root)


def _boot_nodes(top_node, scaffold):
    for node in walk_nodes(top_node):
        node.scaffold = scaffold
        # Boot attributes
        for attr in getattr(node, "_config_attrs", {}).values():
            booted = {None}
            for cls in type(node).__mro__:
                cls_attr = getattr(cls, attr.attr_name, None)
                if (boot := getattr(cls_attr, "__boot__", None)) and boot not in booted:
                    boot(node, scaffold)
                    booted.add(boot)
        # Boot node hook
        try:
            run_hook(node, "boot")
        except Exception as e:
            errr.wrap(BootError, e, prepend=f"Failed to boot {node}:")
    # fixme: why is this here? Will deadlock in case of BootError on specific node only.
    scaffold._comm.barrier()


def _unset_nodes(top_node):
    for node in walk_nodes(top_node):
        try:
            del node.scaffold
        except Exception:
            pass
        node._config_parent = None
        node._config_key = None
        if hasattr(node, "_config_index"):
            node._config_index = None
        run_hook(node, "unboot")


class ConfigurationAttribute:
    """
    Base implementation of all the different configuration attributes. Call the factory
    function :func:`.attr` instead.
    """

    def __init__(
        self,
        type=None,
        default=None,
        call_default=None,
        required=False,
        key=False,
        unset=False,
        hint=MISSING,
    ):
        if not callable(required):
            self.required = lambda s: required
        else:
            self.required = required
        self.key = key
        self.default = default
        self.call_default = call_default
        self.type = self._set_type(type, key)
        self.unset = unset
        self.hint = hint

    def __set_name__(self, owner, name):
        self.attr_name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return _getattr(instance, self.attr_name)

    def fset(self, instance, value):
        return _setattr(instance, self.attr_name, value)

    def __set__(self, instance, value):
        if _hasattr(instance, self.attr_name):
            ex_value = _getattr(instance, self.attr_name)
            _unset_nodes(ex_value)
        if value is None:
            # Don't try to cast None to a value of the attribute type.
            return self.fset(instance, None)
        try:
            value = self.type(value, _parent=instance, _key=self.attr_name)
        except ValueError:
            # This value error should only arise when users are manually setting
            # attributes in an already bootstrapped config tree.
            raise CastError(
                f"'{value}' is not convertible to {self.type.__name__},"
                f" for attribute '{self.attr_name}' of {instance}."
            ) from None
        except (RequirementError, CastError) as e:
            if not hasattr(e, "node") or not e.node:
                e.node, e.attr = instance, self.attr_name
            raise
        except Exception as e:
            raise CastError(
                f"Couldn't cast '{value}' into {self.type.__name__}: {e}",
                instance,
                self.attr_name,
            ) from e
        self.flag_dirty(instance)
        # The value was cast to its intended type and the new value can be set.
        self.fset(instance, value)
        root = _strict_root(instance)
        if _is_booted(root):
            _boot_nodes(value, root.scaffold)

    def _set_type(self, type, key):
        self._config_type = type
        # Determine type of the attribute
        if not type and self.default is not None:
            if self.should_call_default():
                t = builtins.type(self.default())
            else:
                t = builtins.type(self.default)
        else:
            from . import types

            t = type or (key and types.key()) or types.str()
        # This call wraps the type handler so that it accepts all reserved keyword args
        # like `_parent` and `_key`
        t = _wrap_reserved(t)
        return t

    def get_type(self):
        return self._config_type

    def get_hint(self):
        if self.hint is not MISSING:
            return self.hint
        if hasattr(self.type, "__hint__"):
            return self.type.__hint__()
        return MISSING

    def get_node_name(self, instance):
        return instance.get_node_name() + "." + self.attr_name

    def is_node_type(self):
        return hasattr(self._config_type, "_config_attrs")

    def tree(self, instance):
        val = _getattr(instance, self.attr_name)
        return self.tree_of(val)

    def tree_of(self, value):
        # Allow subnodes and other class values to convert themselves to their tree
        # representation
        if hasattr(value, "__tree__"):
            value = value.__tree__()
        # Check if the type handler specifies any inversion function to convert tree
        # values back to how they were found in the document.
        if hasattr(self.type, "__inv__") and value is not None:
            value = self.type.__inv__(value)
        return value

    def flag_dirty(self, instance):
        instance._config_state[self.attr_name] = False
        if self.attr_name not in instance._config_attr_order:
            instance._config_attr_order.append(self.attr_name)

    def is_dirty(self, instance):
        return not instance._config_state.get(self.attr_name, True)

    def flag_pristine(self, instance):
        instance._config_state[self.attr_name] = True

    def get_default(self):
        return self.default() if self.should_call_default() else self.default

    def should_call_default(self):
        cdf = self.call_default
        return cdf or (cdf is None and callable(self.default))


class cfglist(builtins.list):
    """
    Extension of the builtin list to manipulate lists of configuration nodes.
    """

    def get_node_name(self):
        return self._config_parent.get_node_name() + "." + self._config_attr_name

    def append(self, item):
        item = self._preset(len(self), item)
        super().append(item)
        self._postset((item,))
        return self[-1]

    def insert(self, index, item):
        item = self._preset(index, item)
        super().insert(index, item)
        self._postset((item,))
        self._reindex(index)

    def pop(self, index=-1):
        ex_item = super().pop(index)
        _unset_nodes(ex_item)
        self._reindex(index)
        return ex_item

    def clear(self):
        for node in self:
            _unset_nodes(node)
        super().clear()

    def sort(self, **kwargs):
        super().sort(**kwargs)
        self._reindex(0)

    def reverse(self):
        super().reverse()
        self._reindex(0)

    def extend(self, items):
        items = self._fromiter(len(self), items)
        super().extend(items)
        self._postset(items)

    def __setitem__(self, index, item):
        if isinstance(index, int):
            ex_items = [self[index]]
            item = self._preset(index, item)
            items = [item]
            reindex_from = None
        else:
            ex_items = self[index]
            reindex_from = index.indices(len(self))[0]
            items = item = self._fromiter(reindex_from, item)
        for ex_item in ex_items:
            _unset_nodes(ex_item)
        # Don't be fooled, item can be a single value or a list, depending on the index.
        super().__setitem__(index, item)
        if reindex_from is not None:
            self._reindex(reindex_from)
        self._postset(items)

    def _reindex(self, start):
        for i in range(start, len(self)):
            self[i]._config_key = i
            self[i]._config_index = i

    def _fromiter(self, start, items):
        return tuple(self._preset(start + i, item) for i, item in enumerate(items))

    def _preset(self, index, item):
        try:
            item = self._elem_type(item, _parent=self, _key=index)
            try:
                item._config_index = index
            except Exception as e:
                pass
            return item
        except (RequirementError, CastError) as e:
            e.args = (
                f"Couldn't cast element {index} from '{item}'"
                + f" into a {self._elem_type.__name__}. "
                + e.msg,
                *e.args,
            )
            if not e.node:
                e.node, e.attr = self, index
            raise
        except Exception as e:
            raise CastError(
                f"Couldn't cast element {index} from '{item}'"
                + f" into a {self._elem_type.__name__}: {e}"
            )

    def _postset(self, items):
        root = _strict_root(self)
        if root is not None:
            for item in items:
                _resolve_references(root, item)
        if _is_booted(root):
            for item in items:
                _boot_nodes(item, root.scaffold)

    @builtins.property
    def _config_attr_name(self):
        return self._config_attr.attr_name


class ConfigurationListAttribute(ConfigurationAttribute):
    def __init__(self, *args, size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size

    def __set__(self, instance, value, _key=None):
        _setattr(instance, self.attr_name, self.fill(value, _parent=instance))
        self.flag_dirty(instance)

    def __populate__(self, instance, value, unique_list=False):
        cfglist = _getattr(instance, self.attr_name)
        if not unique_list or value not in cfglist:
            builtins.list.append(cfglist, value)

    def fill(self, value, _parent, _key=None):
        _cfglist = cfglist()
        _cfglist._config_parent = _parent
        _cfglist._config_attr = self
        _cfglist._elem_type = self.child_type
        if isinstance(value, builtins.dict):
            raise CastError(f"Dictionary `{value}` given where list is expected.")
        _cfglist.extend(value or builtins.list())
        if self.size is not None and len(_cfglist) != self.size:
            raise CastError(
                f"Couldn't cast {value} into a {self.size}-element list,"
                + f" obtained {len(_cfglist)} elements"
            )
        return _cfglist

    def _set_type(self, type, key=None):
        self.child_type = super()._set_type(type, key=False)

        @wraps(self.fill)
        def wrapper(*args, **kwargs):
            return self.fill(*args, **kwargs)

        # Expose children __inv__ function if it exists
        if hasattr(self.child_type, "__inv__"):
            setattr(wrapper, "__inv__", self.child_type.__inv__)
        return wrapper

    def tree(self, instance):
        val = _getattr(instance, self.attr_name)
        return [self.tree_of(e) for e in val]

    def get_hint(self):
        if self.hint is not MISSING:
            return self.hint
        if hasattr(self.child_type, "__hint__"):
            return [self.child_type.__hint__(), self.child_type.__hint__()]
        return MISSING


class cfgdict(builtins.dict):
    """
    Extension of the builtin dictionary to manipulate dicts of configuration nodes.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                self.get_node_name() + " object has no attribute '{}'".format(name)
            )

    def __setitem__(self, key, value):
        if key in self:
            _unset_nodes(self[key])
        try:
            value = self._elem_type(value, _parent=self, _key=key)
        except (RequirementError, CastError) as e:
            if not (hasattr(e, "node") and e.node):
                e.node, e.attr = self, key
            raise
        except Exception:
            import traceback

            raise CastError(
                "Couldn't cast {}.{} from '{}' into a {}".format(
                    self.get_node_name(), key, value, self._elem_type.__name__
                )
                + "\n"
                + traceback.format_exc()
            )
        else:
            super().__setitem__(key, value)
            root = _strict_root(value)
            if root is not None:
                _resolve_references(root, value)
            if _is_booted(root):
                _boot_nodes(value, root.scaffold)

    def add(self, key, *args, **kwargs):
        if key in self:
            raise KeyError(
                f"{self.get_node_name()} already contains '{key}'."
                + " Use `node[key] = value` if you want to overwrite it."
            )
        self._config_attr.flag_dirty(self._config_parent)
        self[key] = value = self._elem_type(*args, _parent=self, _key=key, **kwargs)
        return value

    def clear(self):
        for node in self.values():
            _unset_nodes(node)
        super().clear()

    def pop(self, key):
        item = super().pop(key)
        _unset_nodes(item)
        return item

    def popitem(self):
        key, value = super().popitem()
        _unset_nodes(value)
        return key, value

    def setdefault(self, key, value):
        if key in self:
            return self[key]
        else:
            self[key] = value
            return self[key]

    def update(self, other):
        for ckey, value in other.items():
            self[ckey] = value

    def __ior__(self, other):
        ex_values = tuple(self.values())
        try:
            merge_f = super().__ior__
        except AttributeError:
            # Patch for 3.8
            merge_f = super().update
        merge_f(other)
        new_values = tuple(self.values())
        for removed_node in (e for e in ex_values if e not in new_values):
            _unset_nodes(removed_node)
        for added_node in (a for a in new_values if a not in ex_values):
            _boot_nodes(added_node, self.scaffold)
        return self

    def copy(self):
        return cfgdictcopy(self)

    @builtins.property
    def _config_attr_name(self):
        return self._config_attr.attr_name

    def get_node_name(self):
        return self._config_parent.get_node_name() + "." + self._config_attr_name


class cfgdictcopy(builtins.dict):
    def __init__(self, other):
        super().__init__(other)
        self._elem_type = other._elem_type
        self._copied_from = other

    @builtins.property
    def _config_attr_name(self):
        return self._copied_from._config_attr_name

    def get_node_name(self):
        return self._copied_from.get_node_name()


class ConfigurationDictAttribute(ConfigurationAttribute):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __set__(self, instance, value, _key=None):
        _setattr(
            instance,
            self.attr_name,
            self.fill(value, _parent=instance, _key=_key or self.attr_name),
        )

    def fill(self, value, _parent, _key=None):
        _cfgdict = cfgdict()
        _cfgdict._config_parent = _parent
        _cfgdict._config_key = _key
        _cfgdict._config_attr = self
        _cfgdict._elem_type = self.child_type
        _cfgdict.update(value or builtins.dict())
        return _cfgdict

    def _set_type(self, type, key=None):
        self.child_type = super()._set_type(type, key=False)

        @wraps(self.fill)
        def wrapper(*args, **kwargs):
            return self.fill(*args, **kwargs)

        # Expose children __inv__ function if it exists
        if hasattr(self.child_type, "__inv__"):
            setattr(wrapper, "__inv__", self.child_type.__inv__)
        return wrapper

    def tree(self, instance):
        val = _getattr(instance, self.attr_name).items()
        return {k: self.tree_of(v) for k, v in val}

    def get_hint(self):
        if self.hint is not MISSING:
            return self.hint
        if hasattr(self.child_type, "__hint__"):
            return {
                "key1": self.child_type.__hint__(),
                "key2": self.child_type.__hint__(),
            }
        return MISSING


class ConfigurationReferenceAttribute(ConfigurationAttribute):
    def __init__(
        self,
        reference,
        key=None,
        ref_type=None,
        populate=None,
        backref=None,
        pop_unique=True,
        **kwargs,
    ):
        self.ref_lambda = reference
        self.ref_key = key
        self.ref_type = ref_type
        self.populate = populate
        self.backref = backref
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
                    raise CfgReferenceError(
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
            raise CfgReferenceError(
                "Reference '{}' of {} does not exist in {}".format(
                    key,
                    self.get_node_name(instance),
                    remote.get_node_name(),
                )
            )
        value = remote[key]
        if self.populate:
            self.populate_reference(instance, value)
        if self.backref:
            self.back_reference(instance, value)

        return value

    def populate_reference(self, instance, reference):
        # Remote descriptors can ask to handle populating itself by implementing a
        # __populate__ method. Here we check if the method exists and if so defer to it.
        if (pop_attr := getattr(reference.__class__, self.populate, None)) and (
            pop_func := getattr(pop_attr, "__populate__", None)
        ):
            pop_func(reference, instance, unique_list=self.pop_unique)
        elif (population := getattr(reference, self.populate, None)) is not None:
            if not self.pop_unique or instance not in population:
                population.append(instance)
        else:
            setattr(reference, self.populate, [instance])

    def back_reference(self, instance, reference):
        if (ref_attr := getattr(reference.__class__, self.backref, None)) and (
            ref_func := getattr(ref_attr, "__backref__", None)
        ):
            ref_func(reference, instance)
        else:
            setattr(reference, self.backref, instance)

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
            remote_keys = builtins.list(iter(value))
        except TypeError:
            raise CfgReferenceError(
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

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self.should_resolve_on_set(instance):
            return super().__get__(instance, owner)
        else:
            return getattr(instance, self.get_ref_key())

    def get_ref_key(self):
        return self.ref_key or (self.attr_name + "_references")

    def __ref__(self, instance, root):
        try:
            remote, remote_keys = self._prepare_self(instance, root)
        except NoReferenceAttributeSignal:  # pragma: nocover
            return None
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
            f"Configuration slot '{self.attr_name}' of {instance.get_node_name()} is"
            f" empty. The {instance.__class__._bsb_entry_point.module_name} plugin"
            f" provided by '{instance.__class__._bsb_entry_point.dist}' should fill the"
            " slot with a configuration attribute."
        )


class ConfigurationProperty(ConfigurationAttribute):
    def __init__(self, fget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fget = fget
        self.fset = None

    def setter(self, f):
        self.fset = f
        return self

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            e = AttributeError(f"Can't set attribute '{self.attr_name}'")
            e.node = self
            raise e
        else:
            return super().__set__(instance, value)


def _collect_kv(n, d, k, v):
    d[k] = v


class ConfigurationAttributeCatcher(ConfigurationAttribute):
    def __init__(
        self,
        *args,
        type=str,
        initial=builtins.dict,
        catch=_collect_kv,
        contains=None,
        tree_cb=None,
        **kwargs,
    ):
        super().__init__(*args, type=type, default=initial, call_default=True, **kwargs)
        self.catch_callback = catch
        if contains is not None:
            self.contains = contains
        if tree_cb is not None:
            self.tree_callback = tree_cb

    def __set__(self, instance, value):
        _setattr(instance, self.attr_name, value)

    def get_caught(self, instance):
        if not hasattr(instance, f"_{self.attr_name}_caught"):
            setattr(instance, f"_{self.attr_name}_caught", {})
        return getattr(instance, f"_{self.attr_name}_caught")

    def __catch__(self, node, key, value):
        # Try to cast to our type, if it fails it will be caught by whoever is asking us
        # to catch this and we don't catch this value.
        cast = self.type(value, _parent=node, _key=key)
        # If succesfully cast, catch this value by executing our catch callback.
        self.catch_callback(node, _getattr(node, self.attr_name), key, cast)
        self.get_caught(node)[key] = cast

    def tree(self, instance):
        # The default attr catcher collects what it catches in a dict. When we want to
        # build the config tree again these values should be placed back in their
        # original keys. We don't want to store our caught values in the config file. To
        # do so we use the `tree_callback` instead.
        return None

    def contains(self, instance, key):
        return key in self.get_caught(instance)

    def tree_callback(self, instance, key):
        # When building the config tree the values that were caught can't be found in the
        # attrs and the tree builder will check all catch-attr's `contains` methods and
        # calls the right tree_callback to fetch the value.
        value = _getattr(instance, self.attr_name)[key]
        if hasattr(value, "__tree__"):
            value = value.__tree__()
        return value
