import functools
import importlib
import inspect
import os
import sys
import types
import warnings
from collections import defaultdict
from re import sub

import errr

from .._package_spec import warn_missing_packages
from .._util import get_qualified_class_name
from ..exceptions import (
    BootError,
    CastError,
    ConfigurationError,
    DynamicClassError,
    DynamicClassInheritanceError,
    DynamicObjectNotFoundError,
    PluginError,
    RequirementError,
    UnfitClassCastError,
    UnresolvedClassCastError,
)
from ..reporting import warn
from ._hooks import overrides


def _has_own_init(meta_subject, kwargs):
    try:
        determined_class = meta_subject.__new__.class_determinant(meta_subject, kwargs)
        return overrides(determined_class, "__init__", mro=True)
    except Exception:
        return overrides(meta_subject, "__init__", mro=True)


def make_metaclass(cls):
    # We make a `NodeMeta` class for each decorated node class, in compliance with any
    # metaclasses they might already have (to prevent metaclass confusion).
    # The purpose of the metaclass is to rewrite `__new__` and `__init__` arguments,
    # and to always call `__new__` and `__init__` in the same manner.
    # The metaclass makes it so that there are 3 overloaded constructor forms:
    #
    # MyNode({ <config dict values> })
    # MyNode(example="attr", values="here")
    # ParentNode(me=MyNode(...))
    #
    # The third makes it that type handling and other types of casting opt out early
    # and keep the object reference that the user gives them
    class ConfigArgRewrite:
        def __call__(meta_subject, *args, _parent=None, _key=None, **kwargs):
            has_own_init = _has_own_init(meta_subject, kwargs)
            # Rewrite the arguments
            primer = args[0] if args else None
            if isinstance(primer, meta_subject):
                _set_pk(primer, _parent, _key)
                return primer
            elif isinstance(primer, dict):
                args = args[1:]
                primed = primer.copy()
                primed.update(kwargs)
                kwargs = primed
            elif primer is not None and not has_own_init:
                # If we're dealing with a typical config node, the primer should be a dict
                # or already precast node. If it is not, we consider it invalid input,
                # unless the user has specified its own `__init__` function and will deal
                # with the input arguments there.
                raise ValueError(f"Unexpected positional argument '{primer}'")
            # Call the base class's new with internal arguments
            instance = meta_subject.__new__(
                meta_subject, *args, _parent=_parent, _key=_key, **kwargs
            )
            instance._config_pos_init = getattr(instance, "_config_pos_init", False)
            # Call the end user's __init__ with the rewritten arguments, if one is defined
            if has_own_init:
                sig = inspect.signature(instance.__init__)
                try:
                    # Check whether the arguments match the signature. We use `sig.bind`
                    # so that the function isn't actually called, as this could mask
                    # `TypeErrors` that occur inside the function.
                    sig.bind(*args, **kwargs)
                except TypeError as e:
                    # Since the user might not know where all these additional arguments
                    # are coming from, inform them that config nodes get passed their
                    # config attrs, and how to correctly override __init__.
                    Param = inspect.Parameter
                    help_params = {"self": Param("self", Param.POSITIONAL_OR_KEYWORD)}
                    help_params.update(sig._parameters)
                    help_params["kwargs"] = Param("kwargs", Param.VAR_KEYWORD)
                    sig._parameters = help_params
                    raise TypeError(
                        f"`{instance.__init__.__module__}.__init__` {e}."
                        + " When overriding `__init__` on config nodes, do not define"
                        + " any positional arguments, and catch any additional"
                        + " configuration attributes that are passed as keyword arguments"
                        + f": e.g. 'def __init__{sig}'"
                    ) from None
                else:
                    instance._config_pos_init = bool(len(args))
                    instance.__init__(*args, **kwargs)
            return instance

    # Avoid metaclass conflicts by prepending our rewrite class to existing metaclass MRO
    class NodeMeta(ConfigArgRewrite, *cls.__class__.__mro__):
        def __new__(cls, *args, **kwargs):
            rcls = super().__new__(cls, *args, **kwargs)
            # `__init_subclass__` refused to be called with correct subclass, so call
            # it ourselves.
            if hasattr(rcls.__bases__[0], "_cfgnode_replaced_ics"):
                rcls.__bases__[0]._cfgnode_replaced_ics(rcls, **kwargs)
            return rcls

    return NodeMeta


class NodeKwargs(dict):
    def __init__(self, instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_shortform = getattr(instance, "_config_pos_init", False)


def compose_nodes(*node_classes):
    """
    Create a composite mixin class of the given classes. Inherit from the returned class
    to inherit from more than one node class.

    """
    meta = type("ComposedMetaclass", tuple(type(cls) for cls in node_classes), {})
    return meta("CompositionMixin", node_classes, {})


def compile_class(cls):
    cls_dict = dict(cls.__dict__)
    if "__dict__" in cls_dict:
        del cls_dict["__dict__"]
    if "__weakref__" in cls_dict:
        del cls_dict["__weakref__"]
    ncls = make_metaclass(cls)(cls.__name__, cls.__bases__, cls_dict)
    for method in ncls.__dict__.values():
        _replace_closure_cells(method, cls, ncls)

    # Shitty hack, for some reason I couldn't find a way to override the first argument
    # of `__init_subclass__` methods, that would otherwise work on other classmethods,
    # so we noop the actual `__init_subclass__` and we call `__init_subclass__` ourselves
    # from the metaclass' `__new__` method, where the argument replacement works as usual.
    if (
        hasattr(ncls, "__init_subclass__")
        and "__init_subclass__" in ncls.__dict__
        and not isinstance(ncls.__init_subclass__, types.BuiltinFunctionType)
    ):
        ncls._cfgnode_replaced_ics = ncls.__init_subclass__.__func__
        ncls.__init_subclass__ = lambda *args, **kwargs: None
    classmap = getattr(ncls, "_config_dynamic_classmap", None)
    if classmap is not None:
        # Replace the reference to the old class with the new class.
        # The auto classmap entry is added in `__init_subclass__`, which happens before
        # we replace the class.
        for k, v in classmap.items():
            if v is cls:
                classmap[k] = ncls
    return ncls


def _replace_closure_cells(method, old, new):
    cl = getattr(method, "__closure__", None) or []
    for cell in cl:
        if cell.cell_contents is old:
            cell.cell_contents = new
        elif inspect.isfunction(cell.cell_contents):
            _replace_closure_cells(cell.cell_contents, old, new)


def compile_isc(node_cls, dynamic_config):
    if not dynamic_config or not dynamic_config.auto_classmap:
        return node_cls.__init_subclass__

    from ._hooks import overrides

    def dud(*args, **kwargs):
        pass

    if overrides(node_cls, "__init_subclass__"):
        f = node_cls.__init_subclass__
    else:
        f = dud

    def __init_subclass__(cls, classmap_entry=MISSING, **kwargs):
        super(node_cls, cls).__init_subclass__(**kwargs)
        if classmap_entry is MISSING:
            classmap_entry = _snake_case(cls.__name__)
        if classmap_entry is not None:
            node_cls._config_dynamic_classmap[classmap_entry] = cls
        f(**kwargs)

    return classmethod(__init_subclass__)


def _snake_case(s):
    return "_".join(
        sub("([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))).split()
    ).lower()


def _node_determinant(cls, kwargs):
    return cls


def compile_new(node_cls, dynamic=False, pluggable=False, root=False):
    if pluggable:
        class_determinant = _get_pluggable_class
    elif dynamic:
        class_determinant = _get_dynamic_class
    else:
        class_determinant = _node_determinant

    def __new__(_cls, *args, _parent=None, _key=None, **kwargs):
        ncls = class_determinant(_cls, kwargs)
        instance = object.__new__(ncls)
        instance._config_pos_init = bool(len(args))
        _set_pk(instance, _parent, _key)
        if root:
            instance._config_isfinished = False
        instance.__post_new__(**kwargs)
        if _cls is not ncls:
            instance.__init__(*args, **kwargs)
        return instance

    __new__.class_determinant = class_determinant

    return __new__


def _set_pk(obj, parent, key):
    obj._config_parent = parent
    obj._config_key = key
    if not hasattr(obj, "_config_attr_order"):
        obj._config_attr_order = []
    if not hasattr(obj, "_config_state"):
        obj._config_state = {}
    for a in get_config_attributes(obj.__class__).values():
        if a.key:
            from ._attrs import _setattr

            _setattr(obj, a.attr_name, key)


def _missing_requirements(instance, attr, kwargs):
    # We use `self.__class__`, not `cls`, to get the proper child class.
    cls = instance.__class__
    dynamic_root = getattr(cls, "_config_dynamic_root", None)
    kwargs = NodeKwargs(instance, kwargs)
    if dynamic_root is not None:
        dynamic_attr = dynamic_root._config_dynamic_attr
        # If we are checking the dynamic attribute, but we're already a dynamic subclass,
        # we skip the required check.
        return (
            attr.attr_name == dynamic_attr
            and cls is dynamic_root
            and attr.required(kwargs)
        ) or (attr.attr_name != dynamic_attr and attr.required(kwargs))
    else:
        return attr.required(kwargs)


def compile_postnew(cls):
    def __post_new__(self, _parent=None, _key=None, **kwargs):
        attrs = get_config_attributes(self.__class__)
        self._config_attr_order = list(kwargs.keys())
        catch_attrs = [a for a in attrs.values() if hasattr(a, "__catch__")]
        leftovers = kwargs.copy()
        values = {}
        for attr in attrs.values():
            name = attr.attr_name
            value = values[name] = leftovers.pop(name, None)
            try:
                if _missing_requirements(self, attr, kwargs) and value is None:
                    raise RequirementError(f"Missing required attribute '{name}'")
            except RequirementError as e:
                # Catch both our own and possible `attr.required` RequirementErrors
                # and set the node detail before passing it on
                e.node = self
                raise
        for attr in attrs.values():
            name = attr.attr_name
            if attr.key and attr.attr_name not in kwargs:
                # If this is a "key" attribute, and the user didn't overwrite it,
                # set the attribute to the config key
                setattr(self, name, self._config_key)
                attr.flag_pristine(self)
            elif (value := values[name]) is None:
                if _is_settable_attr(attr):
                    setattr(self, name, attr.get_default())
                attr.flag_pristine(self)
            else:
                setattr(self, name, value)
                attr.flag_dirty(self)
        for key, value in leftovers.items():
            try:
                _try_catch_attrs(self, catch_attrs, key, value)
            except UncaughtAttributeError:
                try:
                    setattr(self, key, value)
                except AttributeError:
                    raise AttributeError(
                        f"Configuration attribute key '{key}' conflicts with"
                        + f" readonly class attribute on `{self.__class__.__module__}"
                        + f".{self.__class__.__name__}`."
                    ) from None
                raise ConfigurationError(f"Unknown attribute:  '{key}'") from None

    return __post_new__


def wrap_root_postnew(post_new):
    def __post_new__(self, *args, _parent=None, _key=None, _store=None, **kwargs):
        if not hasattr(self, "_meta"):
            self._meta = {"path": None, "produced": True}

        try:
            # Root node bootstrapping sequence
            _bootstrap_components(kwargs.get("components", []), file_store=_store)
            warn_missing_packages(kwargs.get("packages", []))
        except Exception as e:
            raise BootError("Failed to bootstrap configuration.") from e

        try:
            with warnings.catch_warnings(record=True) as log:
                try:
                    post_new(self, *args, _parent=None, _key=None, **kwargs)
                except (CastError, RequirementError) as e:
                    _bubble_up_exc(e, self._meta)
                self._config_isfinished = True
                _resolve_references(self)
        finally:
            _bubble_up_warnings(log)

    return __post_new__


def _is_settable_attr(attr):
    return not hasattr(attr, "fget") or attr.fset


def _bubble_up_exc(exc, meta):
    if hasattr(exc, "node") and exc.node is not None:
        node = " in " + exc.node.get_node_name()
    else:
        node = ""
    attr = f".{exc.attr}" if hasattr(exc, "attr") and exc.attr else ""
    errr.wrap(type(exc), exc, append=node + attr)


def _bubble_up_warnings(log):
    for w in log:
        m = w.message
        if hasattr(m, "node"):
            # Unpack the inner Warning that was passed instead of the warning msg
            attr = f".{m.attr.attr_name}" if hasattr(m, "attr") else ""
            warn(str(m) + " in " + m.node.get_node_name() + attr, type(m), stacklevel=4)
        else:
            warn(str(m), w.category, stacklevel=4)


def _bootstrap_components(components, file_store=None):
    from ..storage._files import CodeDependencyNode

    for component in components:
        component_node = CodeDependencyNode(component)
        component_node.file_store = file_store
        component_node.load_object()


def get_config_attributes(cls):
    attrs = {}
    if not isinstance(cls, type):
        cls = cls.__class__
    for p_cls in reversed(cls.__mro__):
        if hasattr(p_cls, "_config_attrs"):
            attrs.update(p_cls._config_attrs)
        else:
            # Scrape for mixin config attributes
            from ._attrs import ConfigurationAttribute

            attrs.update(
                {
                    key: attr
                    for key, attr in p_cls.__dict__.items()
                    if isinstance(attr, ConfigurationAttribute)
                }
            )
        for unset in getattr(p_cls, "_config_unset", []):
            attrs.pop(unset, None)
    return attrs


def _get_node_name(self):
    name = ".<missing>"
    if getattr(self, "attr_name", None) is not None:
        name = "." + str(self.attr_name)
    if getattr(self, "_config_key", None) is not None:
        name = "." + str(self._config_key)
    if hasattr(self, "_config_index"):
        if self._config_index is None:
            return "{removed}"
        else:
            name = "[" + str(self._config_index) + "]"
    if getattr(self, "name", None) is not None:
        name = "." + self.name
    if getattr(self, "_config_parent", None):
        return self._config_parent.get_node_name() + name
    else:
        return "{standalone}" + name


def make_get_node_name(node_cls, root):
    if root:
        node_cls.get_node_name = lambda self: r"{root}"
    else:
        node_cls.get_node_name = _get_node_name


class UncaughtAttributeError(Exception):
    pass


def _try_catch_attrs(node, catchers, key, value):
    # See if any of the attributes in the node can catch the value of an unknown key in
    # the configuration section. If none of them catch the value, raise an
    # `UncaughtAttributeError`
    for attr in catchers:
        try:
            _try_catch(attr.__catch__, node, key, value)
            break
        except UncaughtAttributeError:
            pass
    else:
        raise UncaughtAttributeError()


def _try_catch(catch, node, key, value):
    try:
        return catch(node, key, value)
    except Exception:
        raise UncaughtAttributeError()


def _get_dynamic_class(node_cls, kwargs):
    if node_cls is not node_cls._config_dynamic_root:
        # When the node is already a subclass of its dynamic root, we don't need to cast
        # it anymore.
        return node_cls

    attr_name = node_cls._config_dynamic_attr
    dynamic_attr = getattr(node_cls, attr_name)
    if attr_name in kwargs:
        loaded_cls_name = kwargs[attr_name]
    elif dynamic_attr.required(kwargs):
        raise RequirementError(f"Dynamic node must contain a '{attr_name}' attribute")
    elif dynamic_attr.should_call_default():  # pragma: nocover
        loaded_cls_name = dynamic_attr.default()
    else:
        # Fall back to the default value, or the current class.
        loaded_cls_name = dynamic_attr.default or node_cls.__name__
    module_path = ["__main__", node_cls.__module__]
    classmap = get_classmap(node_cls)
    interface = getattr(node_cls, "_config_dynamic_root")
    try:
        dynamic_cls = _load_class(
            loaded_cls_name, module_path, interface=interface, classmap=classmap
        )
    except DynamicClassInheritanceError:
        mapped_class_msg = _get_mapped_class_msg(loaded_cls_name, classmap)
        raise UnfitClassCastError(
            "'{}'{} is not a valid class as it does not inherit from {}".format(
                loaded_cls_name,
                mapped_class_msg,
                node_cls.__name__,
            )
        ) from None
    except DynamicClassError:
        mapped_class_msg = _get_mapped_class_msg(loaded_cls_name, classmap)
        raise UnresolvedClassCastError(
            f"Could not resolve '{loaded_cls_name}'{mapped_class_msg} to a class"
        ) from None
    return dynamic_cls


def _get_pluggable_class(node_cls, kwargs):
    plugin_label = node_cls._config_plugin_name or node_cls.__name__
    if node_cls._config_plugin_key not in kwargs:
        raise CastError(
            "Pluggable node must contain a '{}' attribute to select a {}".format(
                node_cls._config_plugin_key,
                plugin_label,
            )
        )
    plugin_name = kwargs[node_cls._config_plugin_key]
    plugins = node_cls.__plugins__()
    if plugin_name not in plugins:
        raise PluginError("Unknown {} '{}'".format(plugin_label, plugin_name))
    plugin_cls = plugins[plugin_name]
    # TODO: Enforce class inheritance
    return plugin_cls


def _get_mapped_class_msg(loaded_cls_name, classmap):
    if classmap and loaded_cls_name in classmap:
        return " (mapped to '{}')".format(classmap[loaded_cls_name])
    else:
        return ""


def _load_class(cfg_classname, module_path, interface=None, classmap=None):
    if classmap and cfg_classname in classmap:
        cfg_classname = classmap[cfg_classname]
    if inspect.isclass(cfg_classname):
        class_ref = cfg_classname
        class_name = cfg_classname.__name__
    else:
        class_ref = _load_object(cfg_classname, module_path)
        class_name = class_ref.__name__

    def qualname(cls):
        return cls.__module__ + "." + cls.__name__

    if interface and not issubclass(class_ref, interface):
        raise DynamicClassInheritanceError(
            "Dynamic class '{}' must derive from {}".format(
                class_name, qualname(interface)
            )
        )
    return class_ref


def _load_object(object_path, module_path):
    class_parts = object_path.split(".")
    object_name = class_parts[-1]
    module_name = ".".join(class_parts[:-1])
    if not module_name:
        object_ref = _search_module_path(object_name, module_path, object_path)
    else:
        object_ref = _get_module_object(object_name, module_name, object_path)

    return object_ref


def _search_module_path(class_name, module_path, cfg_classname):
    for module_name in module_path:
        module_dict = sys.modules[module_name].__dict__
        if class_name in module_dict:
            return module_dict[class_name]
    raise DynamicObjectNotFoundError("Class not found: " + cfg_classname)


def _get_module_object(object_name, module_name, object_path):
    sys.path.append(os.getcwd())
    try:
        module_ref = importlib.import_module(module_name)
    finally:
        tmp = list(reversed(sys.path))
        tmp.remove(os.getcwd())
        sys.path = list(reversed(tmp))
    try:
        return getattr(module_ref, object_name)
    except Exception:
        raise DynamicObjectNotFoundError(f"'{object_path}' not found.")


def make_dictable(node_cls):
    def __contains__(self, attr):
        return attr in get_config_attributes(self.__class__)

    def __getitem__(self, attr):
        if attr in get_config_attributes(self.__class__):
            return getattr(self, attr)
        else:
            raise KeyError(attr)

    def __iter__(self):
        return (attr for attr in get_config_attributes(self.__class__))

    node_cls.__contains__ = __contains__
    node_cls.__getitem__ = __getitem__
    node_cls.__iter__ = __iter__


def make_tree(node_cls):
    def get_tree(instance):
        if hasattr(instance, "__inv__") and not getattr(instance, "_config_inv", None):
            instance._config_inv = True
            inv = instance.__inv__()
            instance._config_inv = False
            return inv
        attrs = get_config_attributes(instance.__class__)
        catch_attrs = [a for a in attrs.values() if hasattr(a, "__catch__")]
        tree = {}
        for name in instance._config_attr_order:
            if name in attrs:
                attr = attrs[name]
                if attr.is_dirty(instance):
                    value = attr.tree(instance)
                else:
                    value = None
            else:
                for catcher in catch_attrs:
                    if catcher.contains(instance, name):
                        value = catcher.tree_callback(instance, name)
                        break
                else:
                    value = getattr(instance, name, None)
            if value is not None:
                tree[name] = value
        return tree

    node_cls.__tree__ = get_tree


def make_copyable(node_cls):
    def loc_copy(instance, memo=None):
        return type(instance)(instance.__tree__())

    node_cls.__copy__ = loc_copy
    node_cls.__deepcopy__ = loc_copy


def walk_node_attributes(node):
    """
    Walk over all of the child configuration nodes and attributes of ``node``.

    :returns: attribute, node, parents
    :rtype: Tuple[:class:`~.config.ConfigurationAttribute`, Any, Tuple]
    """
    attrs = get_config_attributes(node)
    if not attrs:
        if hasattr(node, "_config_attr"):
            attrs = _get_walkable_iterator(node)
        else:
            return
    for attr in attrs.values():
        yield node, attr
        # Yield but don't follow references.
        if hasattr(attr, "__ref__"):
            continue
        child = attr.__get__(node, node.__class__)
        yield from walk_node_attributes(child)


def walk_nodes(node):
    """
    Walk over all of the child configuration nodes of ``node``.

    :returns: node generator
    :rtype: Any
    """
    if hasattr(node.__class__, "_config_attrs"):
        attrs = node.__class__._config_attrs
    elif hasattr(node, "_config_attr"):
        attrs = _get_walkable_iterator(node)
    else:
        return
    yield node
    for attr in attrs.values():
        # Yield but don't follow references.
        if hasattr(attr, "__ref__"):
            continue
        child = attr.__get__(node, node.__class__)
        yield from walk_nodes(child)


def walk_node_values(start_node):
    for node, attr in walk_node_attributes(start_node):
        yield node, attr.attr_name, attr.__get__(node, node.__class__)


def _resolve_references(root, start=None, /):
    from ._attrs import _setattr

    if start is None:
        start = root
    if root._config_isfinished:
        for node, attr in walk_node_attributes(root):
            if hasattr(attr, "__ref__"):
                ref = attr.__ref__(node, root)
                _setattr(node, attr.attr_name, ref)


class WalkIterDescriptor:
    def __init__(self, n, v):
        self.attr_name = n
        self.v = v

    def __get__(self, instance, cls):
        return self.v


def _get_walkable_iterator(node):
    if isinstance(node, dict):
        walkiter = {}
        for name, value in node.items():
            walkiter[name] = WalkIterDescriptor(name, value)
        return walkiter
    elif isinstance(node, list):
        walkiter = {}
        for i, value in enumerate(node):
            walkiter[i] = WalkIterDescriptor(i, value)
        return walkiter


_classmap_registry = defaultdict(dict)


@functools.cache
def load_component_plugins():
    from ..plugins import discover

    plugins = discover("components")
    for plugin in plugins.values():
        if isinstance(plugin, dict):
            for class_name, classmap in plugin.items():
                register_classmap(class_name, classmap)

    return plugins


def register_classmap(cls_name, classmap):
    _classmap_registry[cls_name].update(classmap)


def get_classmap(cls):
    load_component_plugins()
    classmap = getattr(cls, "_config_dynamic_classmap", {})
    classmap.update(_classmap_registry[get_qualified_class_name(cls)])
    return classmap


MISSING = object()
