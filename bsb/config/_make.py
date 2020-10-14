from ..exceptions import *
from ..reporting import warn
import inspect, re, sys
from functools import wraps


def wrap_init(cls):
    if hasattr(cls.__init__, "wrapped"):
        return
    wrapped_init = _get_class_init_wrapper(cls)

    def __init__(self, parent=None, **kwargs):
        attrs = _get_class_config_attrs(self.__class__)
        self._config_parent = parent
        for attr in attrs.values():
            if attr.call_default:
                v = attr.default()
            else:
                v = attr.default
            attr.__set__(self, v)
        wrapped_init(self, parent, **kwargs)

    __init__.wrapped = True
    cls.__init__ = __init__
    return __init__


def _get_class_config_attrs(cls):
    attrs = {}
    for p_cls in reversed(cls.__mro__):
        if hasattr(p_cls, "_config_attrs"):
            attrs.update(p_cls._config_attrs)
    return attrs


def _get_class_init_wrapper(cls):
    f = cls.__init__
    params = inspect.signature(f).parameters
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    snake_case = lambda name: pattern.sub("_", name).lower()

    @wraps(f)
    def wrapper(self, parent, **kwargs):
        snake_name = snake_case(parent.__class__.__name__)
        # Node constructors can only have 1 positional argument namely the parent, if you
        # want more complex initialization use a factory classmethod.
        if "parent" in params or snake_name in params:
            f(self, parent, **kwargs)
        else:
            f(self, **kwargs)

    return wrapper


def _get_node_name(self):
    name = ".<missing>"
    if hasattr(self, "attr_name"):
        name = "." + str(self.attr_name)
    if hasattr(self, "_key"):
        name = "." + str(self._key)
    if hasattr(self, "_index"):
        name = "[" + str(self._index) + "]"
    return self._config_parent.get_node_name() + name


def make_get_node_name(node_cls, root):
    if root:
        node_cls.get_node_name = lambda self: r"{root}"
    else:
        node_cls.get_node_name = _get_node_name


def make_cast(node_cls, dynamic=False, pluggable=False, root=False):
    """
    Return a function that can cast a raw configuration node as specified by the
    attribute descriptions in the node class.
    """
    __cast__ = _make_cast(node_cls)
    if root:
        __cast__ = wrap_root_cast(__cast__)
    if pluggable:
        make_pluggable_cast(node_cls)
    elif dynamic:
        make_dynamic_cast(node_cls, dynamic)

    node_cls.__cast__ = __cast__
    return __cast__


def wrap_root_cast(f):
    @wraps(f)
    def __cast__(section, parent, key=None):
        instance = f(section, parent, key)
        _resolve_references(instance)
        return instance

    return __cast__


def _cast_attributes(node, section, node_cls, key):
    attrs = _get_class_config_attrs(node_cls)
    catch_attrs = [a for a in attrs.values() if hasattr(a, "__catch__")]
    attr_names = list(attrs.keys())
    if key:
        node.attr_name = key
    # Cast each of this node's attributes.
    for attr in attrs.values():
        if attr.attr_name in section:
            attr.__set__(node, section[attr.attr_name], key=attr.attr_name)
        else:
            try:
                # Call the requirement function: it either returns a boolean meaning we
                # throw the error, or it can throw a more complex RequirementError itself.
                throw = attr.required(section)
                msg = "Missing required attribute '{}'".format(attr.attr_name)
            except RequirementError as e:
                throw = True
                msg = str(e)
            if throw:
                raise RequirementError(msg + " in {}".format(node.get_node_name()))
        if attr.key and key is not None:
            # The attribute's value should be set to this node's key in its parent.
            attr.__set__(node, key)
    # Check for unknown keys in the configuration section
    for key in section:
        if key not in attr_names:
            value = section[key]
            try:
                _try_catch_attrs(node, catch_attrs, key, value)
            except UncaughtAttributeError:
                warn(
                    "Unknown attribute '{}' in {}".format(key, node.get_node_name()),
                    ConfigurationWarning,
                )
                setattr(node, key, value)
    return node


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


def _make_cast(node_cls):
    def __cast__(section, parent, key=None):
        if hasattr(section.__class__, "_config_attrs"):
            # Casting artifacts found on the section's class so it must have been cast
            # before.
            return section
        if hasattr(node_cls, "__dcast__"):
            # Create an instance of the dynamically configured class.
            node = node_cls.__dcast__(section, parent, key)
        else:
            # Create an instance of the static node class
            node = node_cls(parent=parent)
        if key is not None:
            node._key = key
        _cast_attributes(node, section, node.__class__, key)
        return node

    return __cast__


def make_dynamic_cast(node_cls, dynamic_config):
    attr_name = node_cls._config_dynamic_attr
    dynamic_attr = getattr(node_cls, attr_name)
    if dynamic_config.auto_classmap or dynamic_config.classmap:
        node_cls._config_dynamic_classmap = dynamic_config.classmap or {}

    def __dcast__(section, parent, key=None):
        if dynamic_attr.required(section):
            if attr_name not in section:
                raise RequirementError(
                    "Dynamic node '{}' must contain a '{}' attribute.".format(
                        parent.get_node_name() + ("." + key if key is not None else ""),
                        attr_name,
                    )
                )
            else:
                loaded_cls_name = section[attr_name]
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
                "'{}'{} is not a valid class for {}.{} as it does not inherit from {}".format(
                    loaded_cls_name,
                    mapped_class_msg,
                    parent.get_node_name(),
                    attr_name,
                    node_cls.__name__,
                )
            ) from None
        except DynamicClassError:
            mapped_class_msg = _get_mapped_class_msg(loaded_cls_name, classmap)
            raise UnresolvedClassCastError(
                "Could not resolve '{}'{} to a class in '{}.{}'".format(
                    loaded_cls_name, mapped_class_msg, parent.get_node_name(), attr_name
                )
            ) from None
        node = dynamic_cls(parent=parent)
        return node

    node_cls.__dcast__ = __dcast__
    if dynamic_config.auto_classmap:
        _wrap_isc_auto_classmap(node_cls)
    return __dcast__


def _get_mapped_class_msg(loaded_cls_name, classmap):
    if classmap and loaded_cls_name in classmap:
        return " (mapped to '{}')".format(classmap[loaded_cls_name])
    else:
        return ""


def make_pluggable_cast(node_cls):
    plugin_label = node_cls._config_plugin_name or node_cls.__name__

    def __dcast__(section, parent, key=None):
        if node_cls._config_plugin_key not in section:
            raise CastError(
                "Pluggable node '{}' must contain a '{}' attribute to select a {}.".format(
                    parent.get_node_name() + "." + key,
                    node_cls._config_plugin_key,
                    plugin_label,
                )
            )
        plugin_name = section[node_cls._config_plugin_key]
        plugins = node_cls.__plugins__()
        if plugin_name not in plugins:
            raise PluginError(
                "Unknown {} '{}' in {}".format(
                    plugin_label, plugin_name, parent.get_node_name() + "." + key
                )
            )
        plugin_cls = plugins[plugin_name]
        if node_cls._config_plugin_unpack:
            plugin_cls = node_cls._config_plugin_unpack(plugin_cls)
        # TODO: Enforce class inheritance
        node = plugin_cls(parent=parent)
        return node

    node_cls.__dcast__ = __dcast__
    return __dcast__


def _wrap_isc_auto_classmap(node_cls):
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

    node_cls.__init_subclass__ = classmethod(__init_subclass__)


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
        return getattr(self, attr)

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
    if not hasattr(node.__class__, "_config_attrs"):
        if hasattr(node, "_attr"):
            attrs = _get_walkable_iterator(node)
        else:
            return
    else:
        attrs = node.__class__._config_attrs
    nn = node.attr_name if hasattr(node, "attr_name") else node._attr.attr_name
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
    if not hasattr(node.__class__, "_config_attrs"):
        if hasattr(node, "_attr"):
            attrs = _get_walkable_iterator(node)
        else:
            return
    else:
        attrs = node.__class__._config_attrs
    nn = node.attr_name if hasattr(node, "attr_name") else node._attr.attr_name
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
