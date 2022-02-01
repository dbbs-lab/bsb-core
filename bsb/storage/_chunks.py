import numpy as np


class Chunk(np.ndarray):
    def __new__(cls, chunk, chunk_size):
        obj = super().__new__(cls, (3,), dtype=np.short)
        obj[:] = chunk
        obj._size = np.array(chunk_size, dtype=float)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._size = getattr(obj, "_size", None)

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

    def __eq__(self, other):
        if isinstance(other, int) or len(other.shape) == 0:
            return self.id == other
        else:
            return self.id == other.view(Chunk).id

    @property
    def dimensions(self):
        return self._size

    @property
    def id(self):
        return sum(self[i] * 2 ** (i * 16) for i in range(3))

    @property
    def box(self):
        return np.concatenate((self.ldc, self.mdc))

    @property
    def ldc(self):
        return self._size * self

    @property
    def mdc(self):
        # self._size * (self + 1) might overflow when this formula will not.
        return self._size * self + self._size

    @classmethod
    def from_id(cls, id, size):
        unpacked = np.array([id], dtype=np.dtype("u8")).view("u2")
        if unpacked[-1]:
            raise OverflowError("int too large to be a chunk id")
        return cls(unpacked[:-1], size)
