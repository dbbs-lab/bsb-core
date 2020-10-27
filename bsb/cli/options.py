class BsbOption:
    pass


class BaseOption:
    pass


def load_options():
    from ..plugins import discover

    return discover("options")
