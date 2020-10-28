class Context:
    def __init__(self, options):
        self.options = options


class CLIContext(Context):
    pass


def get_cli_context():
    from .options import load_options

    options = [o() for o in load_options()]
    return CLIContext(options)
