import itertools

import numpy as np

from ._util import obj_str_insert


class _lset(set):
    def __hash__(self):
        return int.from_bytes(":|\\!#è".join(sorted(self)).encode(), "little")

    def __eq__(self, other):
        return hash(self) == hash(_lset(other))

    def copy(self):
        return self.__class__(self)


class EncodedLabels(np.ndarray):
    def __new__(subtype, *args, labels=None, **kwargs):
        kwargs["dtype"] = int
        array = super().__new__(subtype, *args, **kwargs)
        if labels is None:
            labels = {0: _lset()}
        array.labels = {int(k): _lset(v) for k, v in labels.items()}
        return array

    @obj_str_insert
    def __repr__(self):
        labellist = ", ".join(
            (
                f"{sum(self == k)} labelled {list(ls)}"
                if len(ls)
                else f"{sum(self == k)} unlabelled"
            )
            for k, ls in self.labels.items()
        )
        return f"with {len(self)} elements, of which {labellist}"

    __str__ = __repr__

    def __array_finalize__(self, array):
        if array is not None:
            self.labels = getattr(array, "labels", {0: _lset()})

    def __eq__(self, other):
        try:
            return np.allclose(*EncodedLabels._merged_translate((self, other)))
        except Exception:
            return np.array(self, copy=False) == other

    @property
    def raw(self):
        return np.array(self, copy=False)

    def copy(self, *args, **kwargs):
        cp = super().copy(*args, **kwargs)
        cp.labels = {k: v.copy() for k, v in cp.labels.items()}
        return cp

    def label(self, labels, points):
        if not len(points):
            return
        _transitions = {}
        # A counter that skips existing values.
        counter = (c for c in itertools.count() if c not in self.labels)

        # This local function looks up the new id that a point should transition
        # to when `labels` are added to the labels it already has.
        def transition(point):
            nonlocal _transitions
            # Check if we already know the transition of this value.
            if point in _transitions:
                return _transitions[point]
            else:
                # First time making this transition. Join the existing and new labels
                trans_labels = self.labels[point].copy()
                trans_labels.update(labels)
                # Check if this new combination of labels already is assigned an id.
                for k, v in self.labels.items():
                    if trans_labels == v:
                        # Transition labels already exist, store and return it
                        _transitions[point] = k
                        return k
                else:
                    # Transition labels are a new combination, store them under a new id.
                    transition = next(counter)
                    self.labels[transition] = trans_labels
                    # Cache the result
                    _transitions[point] = transition
                    return transition

        # Replace the label values with the transition values
        self[points] = np.vectorize(transition)(self[points])

    def contains(self, labels):
        return np.any(self.get_mask(labels))

    def index_of(self, labels):
        for i, lset in self.labels.items():
            if lset == labels:
                return i
        else:
            raise IndexError(f"Labelset {labels} does not exist")

    def get_mask(self, labels):
        has_any = [k for k, v in self.labels.items() if any(lbl in v for lbl in labels)]
        return np.isin(self, has_any)

    def walk(self):
        """
        Iterate over the branch, yielding the labels of each point.
        """
        for x in self:
            yield self.labels[x].copy()

    def expand(self, label):
        """
        Translate a label value into its corresponding labelset.
        """
        return self.labels[label].copy()

    @classmethod
    def none(cls, len):
        """
        Create EncodedLabels without any labelsets.
        """
        return cls(len, buffer=np.zeros(len, dtype=int))

    @classmethod
    def from_labelset(cls, len, labelset):
        """
        Create EncodedLabels with all points labelled to the given labelset.
        """
        return cls(len, buffer=np.ones(len), labels={0: _lset(), 1: _lset(labelset)})

    @classmethod
    def concatenate(cls, *label_arrs):
        if not label_arrs:
            return EncodedLabels.none(0)
        lookups = EncodedLabels._get_merged_lookups(label_arrs)
        total = sum(len(len_) for len_ in label_arrs)
        concat = cls(total, labels=lookups[0])
        ptr = 0
        for block in EncodedLabels._merged_translate(label_arrs, lookups):
            nptr = ptr + len(block)
            # Concatenate the translated block
            concat[ptr:nptr] = block
            ptr = nptr
        return concat

    @staticmethod
    def _get_merged_lookups(arrs):
        if not arrs:
            return {0: _lset()}
        merged = {}
        new_labelsets = set()
        to_map_arrs = {}
        for arr in arrs:
            for k, l in arr.labels.items():
                if k not in merged:
                    # The label spot is available, so take it
                    merged[k] = l
                elif merged[k] != l:
                    # The labelset doesn't match, so this array will have to be mapped,
                    # and a new spot found for the conflicting labelset.
                    new_labelsets.add(l)
                    # np ndarray unhashable, for good reason, so use `id()` for quick hash
                    to_map_arrs[id(arr)] = arr
                # else: this labelset matches with the superset's nothing to do

        # Collect new spots for new labelsets
        counter = (c for c in itertools.count() if c not in merged)
        lset_map = {}
        for labelset in new_labelsets:
            key = next(counter)
            merged[key] = labelset
            lset_map[labelset] = key

        return merged, to_map_arrs, lset_map

    def _merged_translate(arrs, lookups=None):
        if lookups is None:
            merged, to_map_arrs, lset_map = EncodedLabels._get_merged_lookups(arrs)
        else:
            merged, to_map_arrs, lset_map = lookups
        for arr in arrs:
            if id(arr) not in to_map_arrs:
                # None of the label array's labelsets need to be mapped, good as is.
                block = arr
            else:
                # Lookup each labelset, if found, map to new value, otherwise, map to
                # original value.
                arrmap = {og: lset_map.get(lset, og) for og, lset in arr.labels.items()}
                block = np.vectorize(arrmap.get)(arr)
            yield block
