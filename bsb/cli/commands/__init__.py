"""
Contains all of the logic required to create commands. It should always suffice to import
just this module for a user to create their own commands.

Inherit from :class:`BaseCommand` for regular CLI style commands, or from
:class:`BsbCommand` if you want more freedom in what exactly constitutes a command to the
BSB.
"""

import argparse

from ...exceptions import CommandError
from ...reporting import report


class BaseParser(argparse.ArgumentParser):
    """
    Inherits from argparse.ArgumentParser and overloads the ``error``
    method so that when an error occurs, instead of exiting and exception
    is thrown.
    """

    def error(self, message):
        """
        Raise message, instead of exiting.

        :param message: Error message
        :type message: str
        """
        raise CommandError(message)


_is_root = True


class BsbCommand:
    def add_to_parser(self):
        raise NotImplementedError("Commands must implement a `add_to_parser` method.")

    def handler(self, context):
        raise NotImplementedError("Commands must implement a `handler` method.")

    def __init_subclass__(cls, parent=None, abstract=False, name=None, **kwargs):
        global _is_root
        if abstract:
            return
        if cls.add_to_parser is BsbCommand.add_to_parser:
            raise NotImplementedError("Commands must implement a `add_to_parser` method.")
        if cls.handler is BsbCommand.handler:
            raise NotImplementedError("Commands must implement a `handler` method.")
        if name is None:
            raise CommandError(f"{cls} must register a name.")
        cls.name = name
        cls._subcommands = []
        # The very first registered command will be the RootCommand for `bsb`
        if _is_root:
            _is_root = False
        else:
            if parent is None:
                parent = RootCommand
            parent._subcommands.append(cls)


class BaseCommand(BsbCommand, abstract=True):
    def add_to_parser(self, parent, context, locals, level):
        locals = locals.copy()
        locals.update(self.get_options())
        parser = parent.add_parser(self.name)
        self.add_parser_arguments(parser)
        self.add_parser_options(parser, context, locals, level)
        parser.set_defaults(handler=self.execute_handler)
        self.add_subparsers(parser, context, self._subcommands, locals, level)
        return parser

    def add_subparsers(self, parser, context, commands, locals, level):
        if len(commands) > 0:
            subparsers = parser.add_subparsers()
            for command in commands:
                c = command()
                c._parent = self
                c.add_to_parser(subparsers, context, locals, level + 1)

    def execute_handler(self, namespace, dryrun=False):
        reduced = {}
        context = namespace._context
        for k, v in namespace.__dict__.items():
            if v is None or k in ["_context", "handler"]:
                continue
            stripped = k.lstrip("_")
            level = len(k) - len(stripped)
            if stripped not in reduced or level > reduced[stripped][0]:
                reduced[stripped] = (level, v)

        namespace.__dict__ = {k: v[1] for k, v in reduced.items()}
        self.add_locals(context)
        context.set_cli_namespace(namespace)
        report(f"Context: {context}", level=4)
        if not dryrun:
            self.handler(context)

    def add_locals(self, context):
        # Merge our options into the context, preserving those in the context as we're
        # going up the tree towards lower priority and less specific options.
        options = self.get_options()
        options.update(context.options)
        context.options = options
        if hasattr(self, "_parent"):
            self._parent.add_locals(context)

    def add_parser_options(self, parser, context, locals, level):
        merged = {}
        merged.update(context.options)
        merged.update(locals)
        for option in merged.values():
            option.add_to_parser(parser, level)

    def get_options(self):
        raise NotImplementedError(
            "BaseCommands must implement a `get_options(self)` method."
        )

    def add_parser_arguments(self, parser):
        raise NotImplementedError(
            "BaseCommands must implement an `add_parser_arguments(self, parser)` method."
        )


class RootCommand(BaseCommand, name="bsb"):
    def handler(self, context):
        pass

    def get_parser(self, context):
        parser = BaseParser()
        parser.set_defaults(_context=context)
        parser.set_defaults(handler=self.execute_handler)
        locals = self.get_options()
        self.add_parser_options(parser, context, locals, 0)
        self.add_subparsers(parser, context, self._subcommands, locals, 0)
        return parser

    def get_options(self):
        return {}


def load_root_command():
    from ...plugins import discover

    # Simply discovering the plugin modules should append them to their parent command
    # class using the `__init_subclass__` function.
    discover("commands")
    return RootCommand()


__all__ = ["BaseCommand", "BsbCommand", "RootCommand", "load_root_command"]
