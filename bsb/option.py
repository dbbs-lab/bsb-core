"""
This module contains the classes required to construct options.
"""
import os
import toml
import pathlib
import functools
import argparse
from .exceptions import OptionError


class OptionDescriptor:
    """
    Base option property descriptor. Can be inherited from to create a cascading property
    such as the default CLI, env & script descriptors.
    """

    def __init_subclass__(cls, slug=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.slug = slug

    def __init__(self, *tags):
        self.tags = tags

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, f"_bsbopt_{self.slug}_value", instance.get_default())

    def __set__(self, instance, value):
        set_value = getattr(instance, "setter", lambda x: x)(value)
        setattr(instance, f"_bsbopt_{self.slug}_value", set_value)

    def __delete__(self, instance):
        try:
            delattr(instance, f"_bsbopt_{self.slug}_value")
        except AttributeError:
            pass

    def is_set(self, instance):
        return hasattr(instance, f"_bsbopt_{self.slug}_value")


class CLIOptionDescriptor(OptionDescriptor, slug="cli"):
    """
    Descriptor that retrieves its value from the given CLI command arguments.
    """

    pass


class EnvOptionDescriptor(OptionDescriptor, slug="env"):
    """
    Descriptor that retrieves its value from the environment variables.
    """

    def __init__(self, *args, flag=False):
        super().__init__(*args)
        self.flag = flag

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Iterate the env for all tags, if none are set this returns `None`
        for tag in self.tags:
            if tag in os.environ:
                return self._parse(os.environ[tag])

    def __set__(self, instance, value):
        for tag in self.tags:
            os.environ[tag] = getattr(instance, "setter", lambda x: x)(value)

    def is_set(self, instance):
        return any(tag in os.environ for tag in self.tags)

    def _parse(self, value):
        if self.flag:
            if value.strip().upper() in ("ON", "TRUE", "1", "YES"):
                return True
            else:
                return False
        else:
            return value


class ScriptOptionDescriptor(OptionDescriptor, slug="script"):
    """
    Descriptor that retrieves and sets its value from/to the :mod:`bsb.options` module.
    """

    # This class uses `self.tags[0]`, because all tags are aliases of eachother in the
    # options module.

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if not self.tags:
            # This option has no options module binding
            return None
        from .options import get_module_option

        return get_module_option(self.tags[0])

    def __set__(self, instance, value):
        if not self.tags:
            # This option has no options module binding
            return None
        from .options import set_module_option

        set_value = getattr(instance, "setter", lambda x: x)(value)
        return set_module_option(self.tags[0], set_value)

    def __delete__(self, instance):
        from .options import reset_module_option

        for tag in self.tags:
            reset_module_option(tag)

    def is_set(self, instance):
        from .options import is_module_option_set

        return any(is_module_option_set(tag) for tag in self.tags)


class ProjectOptionDescriptor(OptionDescriptor, slug="project"):
    """
    Descriptor that retrieves and stores values in the `pyproject.toml` file. Traverses
    up the filesystem tree until one is found.
    """

    def __init__(self, *tags):
        if len(tags) > 1:
            raise OptionError(f"Project option can have only 1 tag, got {tags}.")
        super().__init__(*(tags[0].split(".") if tags else ()))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self.tags:
            _, proj = _pyproject_bsb()
            for tag in self.tags[:-1]:
                proj = proj.get(tag, None)
                if proj is None:
                    return None
            return proj.get(self.tags[-1], None)

    def __set__(self, instance, value):
        if self.tags:
            path, proj = _pyproject_bsb()
            deeper = proj
            for tag in self.tags[:-1]:
                deeper = deeper.setdefault(tag, {})
            deeper[self.tags[-1]] = value
            _save_pyproject_bsb(proj)

    def __delete__(self, instance):
        if self.tags:
            path, proj = _pyproject_bsb()
            for tag in self.tags[:-1]:
                proj = proj.get(tag, None)
                if proj is None:
                    return None
            try:
                del proj[self.tags[-1]]
            except KeyError:
                pass
            else:
                _save_pyproject_bsb(proj)

    def is_set(self, instance):
        if self.tag:
            return self.tag in proj


class BsbOption:
    """
    Base option class. Can be subclassed to create new options.
    """

    def __init__(self, positional=False):
        self.positional = positional

    def __init_subclass__(
        cls,
        name=None,
        env=(),
        project=(),
        cli=(),
        script=(),
        description=None,
        flag=False,
        inverted=False,
        list=False,
        readonly=False,
        action=False,
    ):
        """
        Subclass hook that defines the characteristics of the subclassed option class.

        :param name: Unique name for identification
        :type name: str
        :param cli: Positional arguments for the :class:`.CLIOptionDescriptor` constructor.
        :type cli: iterable
        :param cli: Positional arguments for the :class:`.CLIOptionDescriptor` constructor.
        :type cli: iterable
        :param env: Positional arguments for the :class:`.EnvOptionDescriptor` constructor.
        :type env: iterable
        :param script: Positional arguments for the :class:`.ScriptOptionDescriptor` constructor.
        :type script: iterable
        :param description: Description of the option's purpose for the user.
        :type description: str
        :param flag: Indicates that the option is a flag and should toggle on a default off boolean when given.
        :type flag: boolean
        :param inverted: Used only for flags. Indicates that the flag is default on and is toggled off when given.
        :param list: Indicates that the option takes multiple values.
        :type list: boolean
        :param readonly: Indicates that an option can be accessed but not be altered from the ``bsb.options`` module.
        :type readonly: boolean
        :param action: Indicates that the option should execute its ``action`` method.
        :type action: boolean
        """
        cls.name = name
        cls.env = EnvOptionDescriptor(*env, flag=flag)
        cls.project = ProjectOptionDescriptor(*project)
        cls.cli = CLIOptionDescriptor(*cli)
        cls.script = ScriptOptionDescriptor(*script)
        cls.description = description
        cls.is_flag = flag
        cls.inverted_flag = inverted
        cls.use_extend = list
        cls.readonly = readonly
        cls.use_action = action
        cls.positional = False

    def get(self, prio=None):
        """
        Get the option's value. Cascades the script, cli, env & default descriptors together.

        :returns: option value
        """
        if prio is not None:
            return getattr(self, prio)

        cls = self.__class__
        if cls.script.is_set(self):
            return self.script
        if cls.cli.is_set(self):
            return self.cli
        if cls.env.is_set(self):
            return self.env
        return self.get_default()

    def get_default(self):
        """
        Override to specify the default value of the option.
        """
        return None

    def get_cli_tags(self):
        """
        Return the ``argparse`` positional arguments from the tags.

        :returns: ``-x`` or ``--xxx`` for each CLI tag.
        :rtype: list
        """
        if self.positional:
            longest = ""
            for t in type(self).cli.tags:
                if len(t) >= len(longest):
                    longest = t
            return [longest]
        else:
            return [("--" if len(t) != 1 else "-") + t for t in type(self).cli.tags]

    def add_to_parser(self, parser, level):
        """
        Register this option into an ``argparse`` parser.
        """
        kwargs = {}
        kwargs["help"] = self.description
        kwargs["dest"] = level * "_" + self.name
        kwargs["action"] = "store"
        if self.positional:
            kwargs["nargs"] = "?"
            kwargs["metavar"] = self.get_cli_tags()
            args = []
        else:
            args = self.get_cli_tags()
        if self.is_flag:
            kwargs["action"] += "_false" if self.inverted_flag else "_true"
            kwargs["default"] = argparse.SUPPRESS
        if self.use_extend:
            kwargs["action"] = "extend"
            kwargs["nargs"] = "+"
        if self.use_action:
            kwargs["dest"] = "internal_action_list"
            kwargs["action"] = "append_const"
            kwargs["const"] = self.action

        parser.add_argument(*args, **kwargs)

    @classmethod
    def register(cls):
        """
        Register this option class into the :mod:`bsb.options` module.
        """
        from . import options

        o = cls()
        for tag in cls.script.tags:
            options.register_module_option(tag, o)

    @classmethod
    def _unregister(cls):
        """
        Remove this option class from the :mod:`bsb.options` module, not part of the
        public API as removing options is undefined behavior but useful for testing.
        """
        from . import options

        options._remove_tags(*cls.script.tags)


@functools.cache
def _pyproject_content():
    path = pathlib.Path.cwd()
    while str(path) != path.root:
        proj = path / "pyproject.toml"
        if proj.exists():
            with open(proj, "r") as f:
                return proj.resolve(), toml.load(f)
        path = path.parent
    return None, {}  # pragma: nocover


@functools.cache
def _pyproject_bsb():
    path, content = _pyproject_content()
    return path, content.get("tools", {}).get("bsb", {})


def _save_pyproject_bsb(project):
    path, content = _pyproject_content()
    if path is None:
        raise OptionError(
            "No 'pyproject.toml' in current dir or parents,"
            + " can't set project settings."
        )
    content.setdefault("tools", {})["bsb"] = project
    with open(path, "w") as f:
        toml.dump(content, f)
