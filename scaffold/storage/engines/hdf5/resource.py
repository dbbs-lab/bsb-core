import numpy as np


class Resource:
    def __init__(self, engine, path):
        self._engine = engine
        self._path = path

    def create(self, data, *args, **kwargs):
        with self._engine.open("a") as f:
            f().create_dataset(self._path, data=data, *args, **kwargs)

    def remove(self):
        with self._engine.open("a") as f:
            del f()[self._path]

    def get_dataset(self, selector=()):
        with self._engine.open("r") as f:
            return f()[self._path][selector]

    @property
    def attributes(self):
        with self._engine.open("r") as f:
            return dict(f()[self._path].attrs)

    def get_attribute(self, name):
        attrs = self.attributes
        if name not in attrs:
            raise AttributeMissingError(
                "Attribute '{}' not found in '{}'".format(name, self._path)
            )
        return attrs[name]

    def exists(self):
        with self._engine.open("r") as f:
            return self._path in f()

    def unmap(self, selector=(), mapping=lambda m, x: m[x], data=None):
        if data is None:
            data = self.get_dataset(selector)
        map = self.get_attribute("map")
        unmapped = []
        for record in data:
            unmapped.append(mapping(map, record))
        return np.array(unmapped)

    def unmap_one(self, data, mapping=None):
        if mapping is None:
            return self.unmap(data=[data])
        else:
            return self.unmap(data=[data], mapping=mapping)

    def __iter__(self):
        return iter(self.get_dataset())

    @property
    def shape(self):
        with self._engine.open("r") as f:
            return f()[self._path].shape

    def __len__(self):
        return self.shape[0]

    def append(self, new_data, dtype=float):
        if self.exists():
            data = self.get_dataset()
            self.remove()
        else:
            shape = new_data.shape
            shape[0] = 0
            data = np.zeros(shape, dtype=dtype)
        self.create(np.concatenate((data, new_data)), dtype=dtype)
