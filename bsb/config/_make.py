from ..exceptions import (
    CastError,
    RequirementError,
    ConfigurationWarning,
    DynamicClassInheritanceError,
    UnfitClassCastError,
    DynamicClassError,
    UnresolvedClassCastError,
    PluginError,
    DynamicClassNotFoundError,
)
from ..reporting import warn
from ._hooks import overrides
import warnings
import errr
import importlib
import inspect
import sys
import os


def make_metaclass(cls):
    # We make a `NodeMeta` class for each decorated node class, in compliance with any
    # metaclasses they might already have (to prevent metaclass confusion).
    # The purpose of the metaclass is to rewrite `__new__` and `__init__` arguments,
    # and to always call `__new__` and `__init__` in the same manner.
    # The metaclass makes it so that there are 3 overloaded constructor forms:
    #
    # MyNode({ <config dict values> })
    # MyNode(config="dict", values="here")
    # ParentNode(me=MyNode(...))
    #
    # The third makes it that type handling and other types of casting opt out early
    # and keep the object reference that the user gives them
    class ConfigArgRewrite:
        def __call__(meta_subject, *args, _parent=None, _key=None, **kwargs):
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
            # Call the base class's new with internal arguments
            instance = meta_subject.__new__(
                meta_subject, _parent=_parent, _key=_key, **kwargs
            )
            # Call the end user's __init__ with the rewritten arguments, if one is defined
            if overrides(meta_subject, "__init__", mro=True):
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
                    instance.__init__(*args, **kwargs)
            return instance

    # Avoid metaclass conflicts by prepending our rewrite class to existing metaclass MRO
    class NodeMeta(ConfigArgRewrite, *cls.__class__.__mro__):
        pass

    return NodeMeta


def compile_class(cls):
    cls_dict = dict(cls.__dict__)
    if "__dict__" in cls_dict:
        del cls_dict["__dict__"]
    if "__weakref__" in cls_dict:
        del cls_dict["__weakref__"]
    ncls = make_metaclass(cls)(cls.__name__, cls.__bases__, cls_dict)
    for method in ncls.__dict__.values():
        cl = getattr(method, "__closure__", None)
        if cl and cl[0].cell_contents is cls:
            cl[0].cell_contents = ncls
    classmap = getattr(ncls, "_config_dynamic_classmap", None)
    if classmap is not None:
        # Replace the reference to the old class with the new class.
        # The auto classmap entry is added in `__init_subclass__`, which happens before
        # we replace the class.
        for k, v in classmap.items():
            if v is cls:
                classmap[k] = ncls
    return ncls


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

    def __init_subclass__(cls, classmap_entry=None, **kwargs):
        super(node_cls, cls).__init_subclass__(**kwargs)
        if classmap_entry is not None:
            node_cls._config_dynamic_classmap[classmap_entry] = cls
        f(**kwargs)

    return classmethod(__init_subclass__)


def _node_determinant(cls, kwargs):
    return cls


def compile_new(node_cls, dynamic=False, pluggable=False, root=False):
    if pluggable:
        class_determinant = _get_pluggable_class
    elif dynamic:
        class_determinant = _get_dynamic_class
    else:
        class_determinant = _node_determinant

    def __new__(_cls, _parent=None, _key=None, **kwargs):
        ncls = class_determinant(_cls, kwargs)
        instance = object.__new__(ncls)
        _set_pk(instance, _parent, _key)
        if root:
            instance._config_isfinished = False
        instance.__post_new__(**kwargs)
        if _cls is not ncls:
            instance.__init__(**kwargs)
        return instance

    return __new__


def _set_pk(obj, parent, key):
    obj._config_parent = parent
    obj._config_key = key
    if not hasattr(obj, "_config_attr_order"):
        obj._config_attr_order = []
    if not hasattr(obj, "_config_state"):
        obj._config_state = {}
    for a in _get_class_config_attrs(obj.__class__).values():
        if a.key:
            from ._attrs import _setattr

            _setattr(obj, a.attr_name, key)


def compile_postnew(cls, root=False):
    def __post_new__(self, _parent=None, _key=None, **kwargs):
        attrs = _get_class_config_attrs(self.__class__)
        self._config_attr_order = list(kwargs.keys())
        catch_attrs = [a for a in attrs.values() if hasattr(a, "__catch__")]
        leftovers = kwargs.copy()
        values = {}
        for attr in attrs.values():
            name = attr.attr_name
            value = values[name] = leftovers.pop(name, None)
            try:
                if value is None and attr.required(kwargs):
                    raise RequirementError(f"Missing required attribute '{name}'")
            except RequirementError as e:
                if name == getattr(self.__class__, "_config_dynamic_attr", None):
                    # If the dynamic attribute errors in `__post_new__` the constructor of
                    # a non dynamic child class was called, and the dynamic attribute is
                    # no longer required, so silence the error and continue.
                    pass
                else:
                    # Catch both our own and possible `attr.required` RequirementErrors
                    # and set the node detail before passing it on
                    e.node = self
                    raise
        for attr in attrs.values():
            name = attr.attr_name
            if attr.key and attr.attr_name not in kwargs:
                setattr(self, name, self._config_key)
                attr.flag_pristine(self)
            elif (value := values[name]) is None:
                if _is_settable_attr(attr):
                    setattr(self, name, attr.get_default())
                attr.flag_pristine(self)
            else:
                setattr(self, name, value)
                attr.flag_dirty(self)
        # # TODO: catch attrs
        for key, value in leftovers.items():
            try:
                _try_catch_attrs(self, catch_attrs, key, value)
            except UncaughtAttributeError:
                warning = ConfigurationWarning(f"Unknown attribute: '{key}'")
                warning.node = self
                warn(warning, ConfigurationWarning)
                try:
                    setattr(self, key, value)
                except AttributeError:
                    raise AttributeError(
                        f"Unknown configuration attribute key '{key}' conflicts with"
                        + f" readonly class attribute on `{self.__class__.__module__}"
                        + f".{self.__class__.__name__}`."
                    ) from None

    return __post_new__


def wrap_root_postnew(post_new):
    def __post_new__(self, *args, _parent=None, _key=None, **kwargs):
        if not hasattr(self, "_meta"):
            self._meta = {"path": None, "produced": True}
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
            warn(str(m) + " in " + m.node.get_node_name() + attr, type(m))
        else:
            warn(str(m), w.category)


def _get_class_config_attrs(cls):
    attrs = {}
    for p_cls in reversed(cls.__mro__):
        if hasattr(p_cls, "_config_attrs"):
            attrs.update(p_cls._config_attrs)
        for unset in getattr(p_cls, "_config_unset", []):
            attrs.pop(unset, None)
    return attrs


def _get_node_name(self):
    name = ".<missing>"
    if hasattr(self, "attr_name"):
        name = "." + str(self.attr_name)
    if hasattr(self, "_config_key"):
        name = "." + str(self._config_key)
    if hasattr(self, "_config_index"):
        if self._config_index is None:
            name = "{removed}"
        else:
            name = "[" + str(self._config_index) + "]"
    return self._config_parent.get_node_name() + name


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
    attr_name = node_cls._config_dynamic_attr
    dynamic_attr = getattr(node_cls, attr_name)
    if attr_name in kwargs:
        loaded_cls_name = kwargs[attr_name]
    elif dynamic_attr.required(kwargs):
        raise RequirementError(f"Dynamic node must contain a '{attr_name}' attribute")
    elif dynamic_attr.should_call_default():  # pragma: nocover
        loaded_cls_name = dynamic_attr.default()
    else:
        loaded_cls_name = dynamic_attr.default
    module_path = ["__main__", node_cls.__module__]
    classmap = getattr(node_cls, "_config_dynamic_classmap", None)
    try:
        dynamic_cls = _load_class(
            loaded_cls_name, module_path, interface=node_cls, classmap=classmap
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
            "Could not resolve '{}'{} to a class.".format(
                loaded_cls_name, mapped_class_msg
            )
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
        class_parts = cfg_classname.split(".")
        class_name = class_parts[-1]
        module_name = ".".join(class_parts[:-1])
        if module_name == "":
            class_ref = _search_module_path(class_name, module_path, cfg_classname)
        else:
            class_ref = _get_module_class(class_name, module_name, cfg_classname)

    def qualname(cls):
        return cls.__module__ + "." + cls.__name__

    if interface and not issubclass(class_ref, interface):
        raise DynamicClassInheritanceError(
            "Dynamic class '{}' must derive from {}".format(
                class_name, qualname(interface)
            )
        )
    return class_ref


def _search_module_path(class_name, module_path, cfg_classname):
    for module_name in module_path:
        module_dict = sys.modules[module_name].__dict__
        if class_name in module_dict:
            return module_dict[class_name]
    raise DynamicClassNotFoundError("Class not found: " + cfg_classname)


def _get_module_class(class_name, module_name, cfg_classname):
    sys.path.append(os.getcwd())
    try:
        module_ref = importlib.import_module(module_name)
    finally:
        tmp = list(reversed(sys.path))
        tmp.remove(os.getcwd())
        sys.path = list(reversed(tmp))
    module_dict = module_ref.__dict__
    if class_name not in module_dict:
        raise DynamicClassNotFoundError("Class not found: " + cfg_classname)
    return module_dict[class_name]


def make_dictable(node_cls):
    def __contains__(self, attr):
        return attr in _get_class_config_attrs(self.__class__)

    def __getitem__(self, attr):
        if attr in _get_class_config_attrs(self.__class__):
            return getattr(self, attr)
        else:
            raise KeyError(attr)

    def __iter__(self):
        return (attr for attr in _get_class_config_attrs(self.__class__))

    node_cls.__contains__ = __contains__
    node_cls.__getitem__ = __getitem__
    node_cls.__iter__ = __iter__


def make_tree(node_cls):
    def get_tree(instance):
        attrs = _get_class_config_attrs(instance.__class__)
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


def walk_node_attributes(node):
    """
    Walk over all of the child configuration nodes and attributes of ``node``.

    :returns: attribute, node, parents
    :rtype: Tuple[:class:`~.config.ConfigurationAttribute`, Any, Tuple]
    """
    if hasattr(node.__class__, "_config_attrs"):
        attrs = node.__class__._config_attrs
    elif hasattr(node, "_config_attr"):
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
