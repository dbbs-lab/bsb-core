from ..exceptions import *
from ..reporting import warn
import inspect, re
from functools import wraps


def wrap_init(cls):
    if hasattr(cls.__init__, "wrapped"):
        return
    wrapped_init = _get_class_init_wrapper(cls)

    def __init__(self, parent, *args, **kwargs):
        attrs = _get_class_config_attrs(self.__class__)
        self._config_parent = parent
        for attr in attrs.values():
            if attr.call_default:
                v = attr.default()
            else:
                v = attr.default
            attr.__set__(self, v)
        wrapped_init(self, parent, *args, **kwargs)

    __init__.wrapped = True
    cls.__init__ = __init__
    return __init__


def _get_class_config_attrs(cls):
    attrs = {}
    for p_cls in reversed(cls.__mro__):
        if hasattr(p_cls, "_config_attrs"):
            attrs.update(cls._config_attrs)
    return attrs


def _get_class_init_wrapper(cls):
    f = cls.__init__
    params = inspect.signature(f).parameters
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    snake_case = lambda name: pattern.sub("_", name).lower()

    @wraps(f)
    def wrapper(self, parent, *args, **kwargs):
        if "parent" in params or snake_case(parent.__class__.__name__) in params:
            f(self, parent, *args, **kwargs)
        else:
            f(self, *args, **kwargs)

    return wrapper


def _get_node_name(self):
    return self._config_parent.get_node_name() + "." + self.attr_name


def make_get_node_name(node_cls, root):
    if root:
        node_cls.get_node_name = lambda self: r"{root}"
    else:
        node_cls.get_node_name = _get_node_name


def make_cast(node_cls, dynamic=False, root=False):
    """
        Return a function that can cast a raw configuration node as specified by the
        attribute descriptions in the node class.
    """
    __cast__ = _make_cast(node_cls)
    if root:
        __cast__ = wrap_root_cast(__cast__)
    if dynamic:
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
    attr_names = list(node_cls._config_attrs.keys())
    if key:
        node.attr_name = key
    # Cast each of this node's attributes.
    for attr in node_cls._config_attrs.values():
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
            node = node_cls.__dcast__(section, parent)
        else:
            # Create an instance of the static node class
            node = node_cls(parent)
        _cast_attributes(node, section, node_cls, key)
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
        node = dynamic_cls(parent)
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


def _get_walkable_iterator(node):
    # Currently only handle dict
    walkiter = {}
    for name, value in node.items():

        class WalkIterDescriptor:
            def __init__(self):
                self.attr_name = name

            def __get__(attr, instance, cls):
                return value

        walkiter[name] = WalkIterDescriptor()
    return walkiter
