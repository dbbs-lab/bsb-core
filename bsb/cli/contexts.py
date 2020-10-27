from .options import load_options


class CommandContext:
    def __init__(self, options):
        self.options = options


class CLIContext(CommandContext):
    pass


def get_cli_context():
    options = {k: v() for k, v in load_options().items()}
    return CLIContext(options)
