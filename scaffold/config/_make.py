from ..exceptions import *
from ..reporting import warn
import inspect, re
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
        name = "." + self.attr_name
    if hasattr(self, "_key"):
        name = "." + self._key
    if hasattr(self, "_index"):
        name = "[" + self._index + "]"
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
        make_dynamic_cast(node_cls)

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
    attr_names = list(attrs.keys())
    if key:
        node.attr_name = key
    # Cast each of this node's attributes.
    for attr in attrs.values():
        if attr.attr_name in section:
            attr.__set__(node, section[attr.attr_name], key=attr.attr_name)
        elif attr.required:
            raise CastError(
                "Missing required attribute '{}' in {}".format(
                    attr.attr_name, node.get_node_name()
                )
            )
        if attr.key and key is not None:
            # The attribute's value should be set to this node's key in its parent.
            attr.__set__(node, key)
    # Check for unknown keys in the configuration section
    for key in section:
        if key not in attr_names:
            warn(
                "Unknown attribute '{}' in {}".format(key, node.get_node_name()),
                ConfigurationWarning,
            )
            setattr(node, key, section[key])
    return node


def _make_cast(node_cls):
    def __cast__(section, parent, key=None):
        if hasattr(node_cls, "__dcast__"):
            # Create an instance of the dynamically configured class.
            node = node_cls.__dcast__(section, parent, key)
        else:
            # Create an instance of the static node class
            node = node_cls(parent=parent)
        if key is not None:
            node._key = key
        if section.__class__ is node.__class__:
            # The 'section' is an already cast node: trying to cast it again would error;
            return section
        _cast_attributes(node, section, node.__class__, key)
        return node

    return __cast__


def make_dynamic_cast(node_cls):
    def __dcast__(section, parent, key=None):
        if "class" not in section:
            raise CastError(
                "Dynamic node '{}' must contain a 'class' attribute.".format(
                    parent.get_node_name() + ("." + key if key is not None else "")
                )
            )
        dynamic_cls = _load_class(section["class"], interface=node_cls)
        node = dynamic_cls(parent=parent)
        return node

    node_cls.__dcast__ = __dcast__
    return __dcast__


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


def _load_class(configured_class_name, interface=None):
    if inspect.isclass(configured_class_name):
        class_ref = configured_class_name
        class_name = configured_class_name.__name__
    else:
        class_parts = configured_class_name.split(".")
        class_name = class_parts[-1]
        module_name = ".".join(class_parts[:-1])
        if module_name == "":
            module_dict = globals()
        else:
            module_ref = __import__(module_name, globals(), locals(), [class_name], 0)
            module_dict = module_ref.__dict__
        if not class_name in module_dict:
            raise DynamicClassError("Class not found: " + configured_class_name)
        class_ref = module_dict[class_name]
    qualname = lambda cls: cls.__module__ + "." + cls.__name__
    full_class_name = qualname(class_ref)
    if interface and not issubclass(class_ref, interface):
        raise DynamicClassError(
            "Dynamic class '{}' must derive from {}".format(
                class_name, qualname(interface)
            )
        )
    return class_ref


def walk_nodes(node):
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
        for deep_node, deep_attr in walk_nodes(child):
            yield deep_node, deep_attr


def walk_node_values(start_node):
    for node, attr in walk_nodes(start_node):
        yield node, attr.attr_name, attr.__get__(node, node.__class__)


def _resolve_references(root):
    for node, attr in walk_nodes(root):
        if hasattr(attr, "__ref__"):
            ref = attr.__ref__(node, root)
            attr.__set__(node, ref)


class WalkIterDescriptor:
    def __init__(self, n, v):
        self.attr_name = n
        self.v = v

    def __get__(self, instance, cls):
        return self.v


def _get_walkable_iterator(node):
    # Currently only handle dict
    walkiter = {}
    for name, value in node.items():
        walkiter[name] = WalkIterDescriptor(name, value)
    return walkiter
