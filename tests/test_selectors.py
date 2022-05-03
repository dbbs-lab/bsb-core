import unittest, os, sys, numpy as np, h5py, json, string, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.placement.indicator import NameSelector
from bsb.config import from_json
from bsb.exceptions import *
from bsb.storage.interfaces import StoredMorphology


def spoof(*names):
    return [StoredMorphology(n, None, {"name": n}) for n in names]


def spoof_combo(name, *names):
    return [
        StoredMorphology(n, None, {"name": n})
        for n in [name + x for x in names] + [x + name for x in names]
    ]


class TestSelectors(unittest.TestCase):
    def assertPicked(self, picks, sel, loaders, msg=None):
        self.assertEqual(picks, [l.name for l in loaders if sel.pick(l)], msg=msg)

    def test_strict_empty_name_selector(self):
        ns = NameSelector(names=[])
        all = spoof("A", "B", "*", "", "-", "0", "\\", "|", '"', ".*", "\\.")
        ns.validate(all)
        self.assertPicked([], ns, all, "Empty name selector should pick nothing")
        none = spoof()
        ns.validate(none)
        self.assertPicked([], ns, none, "Empty name selector should pick nothing")

    def test_strict_single_name_selector(self):
        ns = NameSelector(names=["B"])
        all = spoof("A", "B", "*", "", "-", "0", "\\", "|", '"', ".*", "\\.")
        ns.validate(all)
        self.assertPicked(["B"], ns, all, "Exact match to name expected")
        none = spoof()
        with self.assertRaises(MissingMorphologyError):
            ns.validate(none)

    def test_wild_single_name_selector(self):
        ns = NameSelector(names=["B*"])
        all = spoof_combo("B", "A", "B", "*", "-", "0", "\\", "|", '"', ".*", "\\.")
        all += spoof("", "B")
        ns.validate(all)
        exp = ["BA", "BB", "B*", "B-", "B0", "B\\", "B|", 'B"', "B.*", "B\\.", "BB", "B"]
        self.assertPicked(exp, ns, all, "Exact match to name expected")
        none = spoof()
        with self.assertRaises(MissingMorphologyError):
            ns.validate(none)
