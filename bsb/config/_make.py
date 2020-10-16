from ..exceptions import *
from .. import exceptions
from ..reporting import warn
import inspect, re, sys, itertools
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


def compile_new(node_cls, dynamic=False, pluggable=False):
    if not dynamic and not pluggable:
        return node_cls.__new__

    if pluggable:
        class_determinant = _get_pluggable_class
    elif dynamic:
        class_determinant = _get_dynamic_class

    def __new__(cls, *args, **kwargs):
        dyn_kwargs = args[0] if args and isinstance(args[0], dict) else kwargs
        instance = object.__new__(class_determinant(cls, dyn_kwargs))
        return instance.__init__(*args, **kwargs)

    return __new__


def compile_init(cls, root=False):
    attrs = _get_class_config_attrs(cls)
    init_globals = _get_init_globals(cls, attrs)
    header = _compile_init_header(cls, attrs)
    body = _compile_init_body(cls, attrs, root=root)
    exec(header + body, init_globals, bait := locals())
    init = _finalize_init(cls, bait["__init__"])
    return init


def _get_init_globals(cls, attrs):
    init = cls.__init__
    required = _make_is_requireds(attrs)
    set_default = _make_defaults(attrs)
    return {
        "init": init,
        **required,
        **set_default,
        **exceptions.__dict__,
        "res_ref": _resolve_references,
    }


def _make_is_requireds(attrs):
    return {f"ir{n}": _make_is_required(attr) for n, attr in enumerate(attrs.values())}


def _make_is_required(attr):
    def is_required(kwargs):
        return attr.required(kwargs)

    return is_required


def _make_defaults(attrs):
    return {f"default{n}": _make_default(attr) for n, attr in enumerate(attrs.values())}


def _make_default(attr):
    if attr.should_call_default():

        def set_default(instance):
            setattr(instance, attr.attr_name, attr.default())

    else:

        def set_default(instance):
            setattr(instance, attr.attr_name, attr.default)

    return set_default


def _compile_init_header(cls, attrs):
    args = ("self", "*args")
    nones = (f"{k}=None" for k in itertools.chain(attrs, ("_parent", "_key")))
    unknown = ("**unknown",)
    argument_list = ", ".join(itertools.chain(args, nones, unknown))
    kwargs_collector = ", ".join(f"'{k}': {k}" for k in attrs)
    header = f"def __init__({argument_list}):\n"
    header += f"    self._config_parent = _parent\n"
    header += f"    self._config_key = _key\n"
    header += f"    if unknown:\n"
    header += f"        plural = 's' if len(unknown) > 1 else ''\n"
    header += f"        raise UnknownConfigAttrError(f\"Unknown configuration attribute{{plural}} \" + ', '.join(f\"'{{a}}'\" for a in unknown), list(unknown.keys()))\n"
    header += f"    argswap = args and isinstance(args[0], dict)\n"
    header += f"    kwargs = args[0] if argswap else {{{kwargs_collector}}}\n"
    header += f"    if argswap:\n"
    for k in attrs:
        header += f"        {k}=kwargs.get('{k}', None)\n"
    return header


def _compile_init_body(cls, attrs, root=False):
    requirements = "\n".join(_make_requirement_check(n, k) for n, k in enumerate(attrs))
    requirements += "\n" if attrs else ""
    set_initials = "\n".join(
        _make_initial_value(n, attr) for n, attr in enumerate(attrs.values())
    )
    set_initials += "\n" if attrs else ""
    body = requirements + set_initials
    if overrides(cls, "__init__"):
        body += "    init(self, *args, **kwargs)"
    if root:
        body += "    res_ref(self)"
    return body


def _make_requirement_check(n, attr_name):
    condition = f"    if {attr_name} is None and ir{n}(kwargs):\n"
    raise_err = (
        f"        raise RequirementError(\"Missing required attribute '{attr_name}'.\")"
    )
    return condition + raise_err


def _make_initial_value(n, attr):
    attr_name = attr.attr_name
    setter = f"    if {attr_name} is None:\n"
    setter += f"        default{n}(self)\n"
    setter += f"    else:\n"
    setter += f"        self.{attr_name} = {attr_name}"
    return setter


def _finalize_init(cls, init):
    return init


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


def _cast_attributes(node, section, node_cls, key):
    attrs = _get_class_config_attrs(node_cls)
    catch_attrs = [a for a in attrs.values() if hasattr(a, "__catch__")]
    attr_names = list(attrs.keys())
    if key:
        node.attr_name = key
    # Cast each of this node's attributes.
    for attr in attrs.values():
        if attr.attr_name in section:
            attr.__set__(node, section[attr.attr_name], _key=attr.attr_name)
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
    def __cast__(section, parent, _key=None):
        if hasattr(section.__class__, "_config_attrs"):
            # Casting artifacts found on the section's class so it must have been cast
            # before.
            return section
        if hasattr(node_cls, "__dcast__"):
            # Create an instance of the dynamically configured class.
            node = node_cls.__dcast__(section, parent, key)
        else:
            # Create an instance of the static node class
            node = node_cls(_parent=parent)
        if key is not None:
            node._config_key = key
        _cast_attributes(node, section, node.__class__, key)
        return node

    return __cast__


def _get_dynamic_class(node_cls, kwargs):
    attr_name = node_cls._config_dynamic_attr
    dynamic_attr = getattr(node_cls, attr_name)
    if attr_name in kwargs:
        loaded_cls_name = kwargs[attr_name]
    elif dynamic_attr.required(kwargs):
        raise RequirementError(
            "Dynamic node '{}' must contain a '{}' attribute.".format(
                parent.get_node_name() + ("." + key if key is not None else ""),
                attr_name,
            )
        )
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
