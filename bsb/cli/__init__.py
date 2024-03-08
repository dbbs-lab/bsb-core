import builtins
import inspect
import sys

from .._contexts import get_cli_context, reset_cli_context
from ..exceptions import CommandError, DryrunError
from .commands import load_root_command


def handle_cli():
    handle_command(sys.argv[1:], exit=True)


def handle_command(command, dryrun=False, exit=False):
    reset_cli_context()
    context = get_cli_context()
    root_command = load_root_command()
    parser = root_command.get_parser(context)
    try:
        namespace = parser.parse_args(command)
    except CommandError as e:
        if exit:
            print(e)
            builtins.exit(1)
        else:
            raise
    if not dryrun:
        for action in namespace.internal_action_list or ():
            action(namespace)
    if not dryrun or _can_dryrun(namespace.handler, namespace):
        namespace.handler(namespace, dryrun=dryrun)
    else:  # pragma: nocover
        raise DryrunError(f"`{namespace.handler.__name__}` doesn't support dryruns.")
    return context


def _can_dryrun(handler, namespace):
    try:
        return bool(inspect.signature(handler).bind(namespace, dryrun=True))
    except TypeError:
        return False


__all__ = ["handle_cli", "handle_command"]
