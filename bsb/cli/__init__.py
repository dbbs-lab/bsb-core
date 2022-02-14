from .._contexts import get_cli_context, reset_cli_context
from .commands import load_root_command
from ..exceptions import *
import sys
import inspect


def handle_cli(dryrun=False):
    reset_cli_context()
    context = get_cli_context()
    handle_command(sys.argv[1:], context, dryrun=dryrun)
    return context


def handle_command(command, context, dryrun=False):
    root_command = load_root_command()
    parser = root_command.get_parser(context)
    try:
        namespace = parser.parse_args(command)
    except CommandError as e:
        print(e)
        exit(1)
    if not dryrun:
        for action in namespace.internal_action_list or ():
            action(namespace)
    if not dryrun or _can_dryrun(namespace.handler, namespace):
        namespace.handler(namespace, dryrun=dryrun)
    else:  # pragma: nocover
        raise DryrunError(f"`{namespace.handler.__name__}` doesn't support dryruns.")


def _can_dryrun(handler, namespace):
    try:
        return bool(inspect.signature(handler).bind(namespace, dryrun=True))
    except TypeError:
        return False
