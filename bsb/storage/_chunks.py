import numpy as np
from ..exceptions import ChunkError

_iinfo = np.iinfo(np.int16)


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
        self_id = np.array(self, copy=False).view(Chunk)._safe_id()
        other_id = np.array(other, copy=False).view(Chunk)._safe_id()
        return self_id == other_id

    def __gt__(self, other):
        return self.id > other.id

    def __lt__(self, other):
        return self.id < other.id

    def __ge__(self, other):
        return self.id >= other.id

    def __le__(self, other):
        return self.id <= other.id

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

    @property
    def id(self):
        return sum(n * 2 ** (i * 16) for i, n in enumerate(self.astype(np.uint16)))

    @property
    def box(self):
        return np.array(np.concatenate((self.ldc, self.mdc)), copy=False)

    @property
    def ldc(self):
        return np.array(self._size * self, copy=False)

    @property
    def mdc(self):
        # self._size * (self + 1) might overflow when this formula will not.
        return np.array(self._size * self + self._size, copy=False)

    @classmethod
    def from_id(cls, id, size):
        raw = [id % 2**17, id // 2**16 % 2**17, id // 2**32 % 2**17]
        unpacked = np.array(raw, dtype=np.uint16).astype(np.int16)
        return cls(unpacked, size)
