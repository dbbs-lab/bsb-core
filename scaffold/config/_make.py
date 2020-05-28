from ..exceptions import *
from ..reporting import warn


def wrap_init(cls, attrs):
    f = cls.__init__

    def __init__(self, parent, *args, **kwargs):
        self._config_parent = parent
        for attr in attrs.values():
            if attr.call_default:
                v = attr.default()
            else:
                v = attr.default
            self.__dict__["_" + attr.attr_name] = v
        f(self, *args, **kwargs)

    cls.__init__ = __init__
    return __init__


def _get_node_name(self):
    return self._config_parent.get_node_name() + "." + self.attr_name


def make_get_node_name(node_cls, root):
    if root:
        node_cls.get_node_name = lambda self: r"root"
    else:
        node_cls.get_node_name = _get_node_name


def make_cast(node_cls):
    """
        Return a function that can cast a raw configuration node as specified by the
        attribute descriptions in the node class.
    """
    attr_names = list(node_cls._config_attrs.keys())

    def __cast__(section, parent, key=None):
        # Create an instance of the node class
        node = node_cls(parent=parent)
        if key:
            node.attr_name = key
        # Cast each of this node's attributes.
        for attr in node_cls._config_attrs.values():
            if attr.attr_name in section:
                node.__dict__["_" + attr.attr_name] = attr.type(
                    section[attr.attr_name], node, key=attr.attr_name
                )
            elif attr.required:
                raise CastError(
                    "Missing required attribute {} in {}".format(
                        attr.attr_name, attr.get_node_name(node)
                    )
                )
            if attr.key and key is not None:
                node.__dict__["_" + attr.attr_name] = key
        # Check for unknown keys in the configuration section
        for key in section:
            if key not in attr_names:
                warn(
                    "Unknown attribute '{}' in {}".format(key, attr.get_node_name(node)),
                    ConfigurationWarning,
                )
                node.__dict__[key] = section[key]
        return node

    node_cls.__cast__ = __cast__
    return __cast__
