import functools


def from_hdf5(file, missing_ok=False):
    """
    Generate a :class:`.core.Scaffold` from an HDF5 file.

    :param file: Path to the HDF5 file.
    :returns: A scaffold object
    :rtype: :class:`Scaffold`
    """
    from ..storage import Storage

    storage = Storage("hdf5", file, missing_ok=missing_ok)
    return storage.load()


@functools.cache
def __getattr__(attr):
    if attr == "Scaffold":
        from .scaffold import Scaffold

        return Scaffold
    else:
        raise AttributeError()
