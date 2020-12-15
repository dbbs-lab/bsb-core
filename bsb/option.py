"""
This module contains the classes required to construct options.
"""
import os


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
        return getattr(instance, f"_{self.slug}_value", instance.get_default())

    def __set__(self, instance, value):
        setattr(instance, f"_{self.slug}_value", value)

    def is_set(self, instance):
        return hasattr(instance, f"_{self.slug}_value")


class CLIOptionDescriptor(OptionDescriptor, slug="cli"):
    """
    Descriptor that retrieves its value from the given CLI command arguments.
    """

    pass


class EnvOptionDescriptor(OptionDescriptor, slug="env"):
    """
    Descriptor that retrieves its value from the environment variables.
    """

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Iterate the env for all tags, if none are set this returns `None`
        for tag in self.tags:
            if tag in os.environ:
                return os.environ[tag]

    def is_set(self, instance):
        return any(tag in os.environ for tag in self.tags)


class ScriptOptionDescriptor(OptionDescriptor, slug="script"):
    """
    Descriptor that retrieves and sets its value from/to the :mod:`bsb.options` module.
    """

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

        return set_module_option(self.tags[0], value)

    def is_set(self, instance):
        from .options import is_module_option_set

        return any(is_module_option_set(tag) for tag in self.tags)


class BsbOption:
    """
    Base option class. Can be subclassed to create new options.
    """

    def __init_subclass__(
        cls,
        name=None,
        cli=(),
        env=(),
        script=(),
        description=None,
        flag=False,
        inverted=False,
        list=False,
        readonly=False,
        action=False,
        **kwargs,
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
        cls.cli = CLIOptionDescriptor(*cli)
        cls.env = EnvOptionDescriptor(*env)
        cls.script = ScriptOptionDescriptor(*script)
        cls.description = description
        cls.is_flag = flag
        cls.inverted_flag = inverted
        cls.use_extend = list
        cls.readonly = readonly
        cls.use_action = action

    def get(self):
        """
        Get the option's value. Cascades the script, cli, env & default descriptors together.

        :returns: option value
        """
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

    @classmethod
    def get_cli_tags(cls):
        """
        Return the ``argparse`` positional arguments from the tags.

        :returns: ``-x`` or ``--xxx`` for each CLI tag.
        :rtype: list
        """
        return [("--" if len(t) != 1 else "-") + t for t in cls.cli.tags]

    def add_to_parser(self, parser, level):
        """
        Register this option into an ``argparse`` parser.
        """
        kwargs = {}
        kwargs["help"] = self.description
        kwargs["dest"] = level * "_" + self.name
        kwargs["action"] = "store"
        if self.is_flag:
            kwargs["action"] += "_false" if self.inverted_flag else "_true"
        if self.use_extend:
            kwargs["action"] = "extend"
            kwargs["nargs"] = "+"
        if self.use_action:
            kwargs["dest"] = "internal_action_list"
            kwargs["action"] = "append_const"
            kwargs["const"] = self.action

        parser.add_argument(*self.get_cli_tags(), **kwargs)

    @classmethod
    def register(cls):
        """
        Register this option class into the :mod:`bsb.options` module.
        """
        from . import options

        for tag in cls.script.tags:
            options.register_module_option(tag, cls)
