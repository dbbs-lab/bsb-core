class Context:
    def __init__(self, options):
        self.options = options

    def __str__(self):
        opt = "; ".join(f"{opt.name}: {opt.get()}" for opt in self.options.values())
        return f"<Context options({opt})>"


class CLIContext(Context):
    def set_cli_namespace(self, namespace):
        for option in self.options.values():
            for tag in _tags_to_namespace(option.__class__.cli.tags):
                if hasattr(namespace, tag):
                    option.cli = getattr(namespace, tag)
        self.arguments = namespace

    def __getattr__(self, attr):
        for option in self.options.values():
            if option.name == attr:
                return option.get()
        return super().__getattribute__(attr)

    def __str__(self):
        base = super().__str__()
        if not hasattr(self, "arguments"):
            return base[:-1] + " without CLI arguments>"
        return base[:-1] + f" {self.arguments}>"


def reset_cli_context():
    from .options import get_option_descriptors

    for opt in get_option_descriptors().values():
        del opt.cli


def get_cli_context():
    from .options import get_option_descriptors

    return CLIContext(get_option_descriptors())


def _tags_to_namespace(tags):
    yield from (tag.replace("-", "_") for tag in tags)
