class Context:
    def __init__(self, options):
        self.options = options


class CLIContext(Context):
    def set_cli_namespace(self, namespace):
        for option in self.options.values():
            for tag in option.__class__.cli.tags:
                if hasattr(namespace, tag):
                    option.cli = getattr(namespace, tag)
        self.arguments = namespace

    def __getattr__(self, attr):
        for option in self.options.values():
            if option.name == attr:
                return option.get()
        return super().__getattribute__(attr)


def get_cli_context():
    from .options import load_options

    options = {k: o() for k, o in load_options().items()}
    return CLIContext(options)
