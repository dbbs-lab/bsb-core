from .._contexts import get_cli_context
from .commands import load_root_command
from ..exceptions import *
import sys


def handle_cli():
    context = get_cli_context()
    handle_command(sys.argv[1:], context)


def handle_command(command, context):
    root_command = load_root_command()
    parser = root_command.get_parser(context)
    try:
        namespace = parser.parse_args(command)
    except CommandError as e:
        print(e)
        exit(1)
    for action in namespace.internal_action_list or ():
        action(namespace)
    namespace.handler(namespace)
