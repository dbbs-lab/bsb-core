from ..exceptions import *
from .. import exceptions
from ..reporting import warn
import inspect, re, sys, itertools, warnings, errr
from functools import wraps
from ._hooks import overrides


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

    def __new__(_cls, *args, _parent=None, _key=None, **kwargs):
        args = list(args)
        primer = args[0] if args else None
        if isinstance(primer, dict):
            args = args[1:]
            (primed := primer.copy()).update(kwargs)
            kwargs = primed
        ncls = class_determinant(_cls, kwargs)
        if isinstance(primer, ncls):
            _set_pk(primer, _parent, _key)
            return primer
        instance = object.__new__(ncls)
        _set_pk(instance, _parent, _key)
        if instance.__class__ is not node_cls:
            instance.__init__(*args, **kwargs)
        return instance

    return __new__


def _set_pk(obj, parent, key):
    obj._config_parent = parent
    obj._config_key = key
    for a in _get_class_config_attrs(obj.__class__).values():
        if a.key:
            a.__set__(obj, key)


def compile_init(cls, root=False):
    if overrides(cls, "__init__"):
        init = cls.__init__
    else:

        def dud(*args, **kwargs):
            pass

        init = dud

    def __init__(self, *args, _parent=None, _key=None, **kwargs):
        attrs = _get_class_config_attrs(self.__class__)
        catch_attrs = [a for a in attrs.values() if hasattr(a, "__catch__")]
        primer = args[0] if args else None
        if isinstance(primer, self.__class__):
            return
        elif isinstance(primer, dict):
            args = args[1:]
            (primed := primer.copy()).update(kwargs)
            kwargs = primed
        leftovers = kwargs.copy()
        values = {}
        missing_requirements = {}
        for attr in attrs.values():
            name = attr.attr_name
            value = values[name] = leftovers.pop(name, None)
            try:
                if value is None and attr.required(kwargs):
                    raise RequirementError(f"Missing required attribute '{name}'.")
            except RequirementError as e:
                # Catch both our own and possible `attr.required` RequirementErrors and
                # set the node detail before passing it on
                e.node = self
                raise
        for attr in attrs.values():
            name = attr.attr_name
            if attr.key and attr.attr_name not in kwargs:
                value = self._config_key
            elif (value := values[name]) is None:
                value = attr.get_default()
            setattr(self, name, value)
        # # TODO: catch attrs
        for key, value in leftovers.items():
            try:
                _try_catch_attrs(self, catch_attrs, key, value)
            except UncaughtAttributeError:
                warning = ConfigurationWarning(f"Unknown attribute: '{key}'")
                warning.node = self
                warn(warning, ConfigurationWarning)
                setattr(self, key, value)

        init(self, *args, **leftovers)

    return __init__


def wrap_root_init(init):
    def __init__(self, *args, _parent=None, _key=None, **kwargs):
        with warnings.catch_warnings(record=True) as log:
            try:
                init(self, *args, _parent=None, _key=None, **kwargs)
            except (CastError, RequirementError) as e:
                _bubble_up_exc(e)
            _resolve_references(self)
        _bubble_up_warnings(log)

    return __init__


def _bubble_up_exc(exc):
    errr.wrap(type(exc), exc, append=" in " + exc.node.get_node_name())


def _bubble_up_warnings(log):
    for w in log:
        m = w.message
        if hasattr(m, "node"):
            # Unpack the inner Warning that was passed instead of the warning msg
            warn(str(m) + " in " + m.node.get_node_name(), type(m))
        else:
            warn(str(m), w)


def _get_class_config_attrs(cls):
    attrs = {}
    for p_cls in reversed(cls.__mro__):
        if hasattr(p_cls, "_config_attrs"):
            attrs.update(p_cls._config_attrs)
    return attrs


def _get_node_name(self):
    name = ".<missing>"
    if hasattr(self, "attr_name"):
        name = "." + str(self.attr_name)
    if hasattr(self, "_config_key"):
        name = "." + str(self._config_key)
    if hasattr(self, "_config_index"):
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
    except:
        raise
        raise UncaughtAttributeError()


def _get_dynamic_class(node_cls, kwargs):
    attr_name = node_cls._config_dynamic_attr
    dynamic_attr = getattr(node_cls, attr_name)
    if attr_name in kwargs:
        loaded_cls_name = kwargs[attr_name]
    elif dynamic_attr.required(kwargs):
        raise RequirementError(f"Dynamic node must contain a '{attr_name}' attribute.")
    elif dynamic_attr.should_call_default():  # pragma: nocover
        loaded_cls_name = dynamic_attr.default()
    else:
        loaded_cls_name = dynamic_attr.default
    module_path = ["__main__", node_cls.__module__]
    if hasattr(node_cls, "_config_dynamic_classmap"):
        classmap = node_cls._config_dynamic_classmap
    else:
        classmap = None
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
            "Pluggable node '{}' must contain a '{}' attribute to select a {}.".format(
                parent.get_node_name() + "." + key,
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
    qualname = lambda cls: cls.__module__ + "." + cls.__name__
    full_class_name = qualname(class_ref)
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
    module_ref = __import__(module_name, globals(), locals(), [class_name], 0)
    module_dict = module_ref.__dict__
    if not class_name in module_dict:
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

    node_cls.__contains__ = __contains__
    node_cls.__getitem__ = __getitem__


def make_tree(node_cls):
    def get_tree(instance):
        return {
            name: tree
            for name, attr in instance.__class__._config_attrs.items()
            if (tree := attr.tree(instance)) is not None
        }

    node_cls.__tree__ = get_tree


def walk_node_attributes(node):
    """
    Walk over all of the child configuration nodes and attributes of ``node``.

    :returns: attribute, node, parents
    :rtype: :class:`ConfigurationAttribute <.config._attrs.ConfigurationAttribute>`,
      any, tuple
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
        for deep_node, deep_attr in walk_node_attributes(child):
            yield deep_node, deep_attr


def walk_nodes(node):
    """
    Walk over all of the child configuration nodes of ``node``.

    :returns: node generator
    :rtype: any
    """
    if hasattr(node.__class__, "_config_attrs"):
        attrs = node.__class__._config_attrs
    elif hasattr(node, "_attr"):
        attrs = _get_walkable_iterator(node)
    else:
        return
    yield node
    for attr in attrs.values():
        # Yield but don't follow references.
        if hasattr(attr, "__ref__"):
            continue

        child = attr.__get__(node, node.__class__)
        for deep_node in walk_nodes(child):
            yield deep_node


def walk_node_values(start_node):
    for node, attr in walk_node_attributes(start_node):
        yield node, attr.attr_name, attr.__get__(node, node.__class__)


def _resolve_references(root):
    from ._attrs import _setattr

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
