# Create a class whose instances can, unlike a list, be modified by the plugin system
class _obj_list(list):
    pass


def __plugin__():
    return _obj_list(__path__)
