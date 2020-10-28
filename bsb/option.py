import os


class OptionDescriptor:
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
    pass


class EnvOptionDescriptor(OptionDescriptor, slug="env"):
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
    def __init_subclass__(cls, cli=(), env=(), script=(), **kwargs):
        cls.cli = CLIOptionDescriptor(*cli)
        cls.env = EnvOptionDescriptor(*env)
        cls.script = ScriptOptionDescriptor(*script)

    def get(self):
        cls = self.__class__
        if cls.script.is_set(self):
            return self.script
        if cls.cli.is_set(self):
            return self.cli
        if cls.env.is_set(self):
            return self.env
        return self.get_default()

    def get_default(self):
        return None
