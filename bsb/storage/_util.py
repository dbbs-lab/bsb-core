import functools
import pathlib
import typing

import appdirs

from ._files import FileDependency

if typing.TYPE_CHECKING:
    from .interfaces import Storage

_bsb_dirs = appdirs.AppDirs("bsb")
_cache_path = pathlib.Path(_bsb_dirs.user_cache_dir)

cache: "Storage"


def __getattr__(name):
    if name == "cache":
        return _get_cache_storage()
    else:
        raise AttributeError(f"{__name__} has no attribute '{name}.")


@functools.cache
def _get_cache_storage():
    from ..storage import Storage

    return Storage("fs", _cache_path)


def _cached_file(source, cache=True):
    return FileDependency(source, file_store=_get_cache_storage().files, cache=cache)
