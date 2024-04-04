import typing

import numpy as np

from ..exceptions import ChunkError

_iinfo = np.iinfo(np.int16)
Chunklike = typing.Union["Chunk", "numpy.typing.ArrayLike"]


class Chunk(np.ndarray):
    """
    Chunk identifier, consisting of chunk coordinates and size.
    """

    def __new__(cls, chunk, chunk_size):
        if any(c < _iinfo.min or c > _iinfo.max for c in chunk):
            raise ChunkError(
                f"Chunk coordinates must be between {_iinfo.min} and {_iinfo.max}."
            )
        obj = super().__new__(cls, (3,), dtype=np.short)
        obj[:] = chunk
        obj._size = np.array(chunk_size, dtype=float)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._size = getattr(obj, "_size", None)

    def __ne__(self, other):
        return self.id != other.id

    def __eq__(self, other):
        self_id, other_id = _safe_ids(self, other)
        return self_id == other_id

    def __gt__(self, other):
        self_id, other_id = _safe_ids(self, other)
        return self_id > other_id

    def __lt__(self, other):
        self_id, other_id = _safe_ids(self, other)
        return self_id < other_id

    def __ge__(self, other):
        self_id, other_id = _safe_ids(self, other)
        return self_id >= other_id

    def __le__(self, other):
        self_id, other_id = _safe_ids(self, other)
        return self_id <= other_id

    def __hash__(self):
        return int(self.id)

    def __reduce__(self):
        # Pickle ourselves, appending the `_size` attribute to our reduced state
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self._size,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # Unpickle ourselves, grabbing the state we appended for `_size`
        super().__setstate__(state[:-1])
        self._size = state[-1]

    def _safe_id(self):
        return int(self) if self.shape == () else self.id

    @property
    def dimensions(self):
        return self._size

    @dimensions.setter
    def dimensions(self, value):
        self._size = np.array(value, dtype=float)

    @property
    def id(self):
        return sum(int(n) * 2 ** (i * 16) for i, n in enumerate(self.astype(np.uint16)))

    @property
    def box(self):
        return np.array(np.concatenate((self.ldc, self.mdc)), copy=False)

    @property
    def ldc(self):
        return np.array(self._size * self.astype(np.float64), copy=False)

    @property
    def mdc(self):
        return np.array(self._size * (self.astype(np.float64) + 1), copy=False)

    @classmethod
    def from_id(cls, id, size):
        return cls(
            np.uint16(
                [id % 2**16, (id // 2**16) % 2**16, (id // 2**32) % 2**16]
            ).astype(np.int16),
            size,
        )


def chunklist(chunks) -> typing.List[Chunk]:
    """
    Convert an iterable of chunk like objects to a sorted unique chunklist
    """
    return sorted(set(c if isinstance(c, Chunk) else Chunk(c, None) for c in chunks))


def _safe_ids(self, other):
    return (
        np.array(self, copy=False).view(Chunk)._safe_id(),
        np.array(other, copy=False).view(Chunk)._safe_id(),
    )


__all__ = ["Chunk", "chunklist"]
