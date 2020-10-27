import abc
import argparse
from ...exceptions import *


class BaseParser(argparse.ArgumentParser):
    """
    Inherits from argparse.ArgumentParser and overloads the ``error``
    method so that when an error occurs, instead of exiting and exception
    is thrown.
    """

    def error(self, message):
        raise CommandError(message)


_is_root = True


class BsbCommand:
    def add_to_parser(self):
        raise NotImplementedError("Commands must implement a `add_to_parser` method.")

    def handler(self):
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
        if _is_root:
            _is_root = False
        elif parent is None:
            RootCommand._subcommands.append(cls)
        else:
            parent._subcommands.append(cls)


class BaseCommand(BsbCommand, abstract=True):
    def get_subcommands(self):
        if callable(self.subcommands):
            self.subcommands = list(self.subcommands())
        return self.subcommands

    def add_to_parser(self, parent, context):
        parser = parent.add_parser(self.name)
        parser.set_defaults(handler=self.handler)
        self.add_subparsers(parser, context, self._subcommands)
        return parser

    def add_subparsers(self, parser, context, commands):
        if len(commands) > 0:
            subparsers = parser.add_subparsers()
            for command in commands:
                command().add_to_parser(subparsers, context)


class RootCommand(BaseCommand, name="bsb"):
    def handler(self, namespace):
        print("open repl", namespc)

    def get_parser(self, context):
        parser = BaseParser()
        parser.set_defaults(handler=self.handler)
        self.add_subparsers(parser, context, self._subcommands)
        return parser


def load_root_command():
    from ...plugins import discover

    # Simply discovering the plugin modules should append them to their parent command
    # class using the `__init_subclass__` function.
    discover("commands")
    return RootCommand()
