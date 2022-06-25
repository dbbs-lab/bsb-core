import unittest, os, sys, numpy as np, h5py, json, string, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.morphologies.selector import NameSelector, NeuroMorphoSelector
from bsb.core import Scaffold
from bsb.cell_types import CellType
from bsb.config import from_json, Configuration
from bsb.morphologies import Morphology, Branch
from bsb.exceptions import *
from bsb.storage.interfaces import StoredMorphology
from test_setup import skip_parallel


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
        ws = NameSelector(names=["*"])
        self.assertEqual(len(all), sum(map(ws.pick, all)), "wildcard should select all")

    # For some reason, under parallel conditions, likely due to the `morphologies.save`,
    # we deadlock.
    @skip_parallel
    def test_cell_type_shorthand(self):
        ct = CellType(spatial=dict(morphologies=[{"names": "*"}]))
        cfg = Configuration.default(
            storage={"root": "selectors_test.hdf5"}, cell_types={"ct": ct}
        )
        s = Scaffold(cfg)
        s.morphologies.save("A", Morphology([Branch([[0, 0, 0]], [1])]), overwrite=True)
        self.assertEqual(1, len(ct.get_morphologies()), "Should select saved morpho")
        ct.spatial.morphologies[0].names = ["B"]
        with self.assertRaises(MissingMorphologyError):
            self.assertEqual(0, len(ct.get_morphologies()), "should select 0 morpho")

    def test_nm_selector(self):
        ct = CellType(
            spatial=dict(
                morphologies=[
                    {
                        "select": "from_neuromorpho",
                        "names": ["H17-03-013-11-08-04_692297214_m"],
                    }
                ]
            )
        )
        cfg = Configuration.default(cell_types={"ct": ct})
        s = Scaffold(cfg)
        self.assertIn("H17-03-013-11-08-04_692297214_m", s.morphologies, "missing NM")
        with self.assertRaises(SelectorError, msg="doesnt exist, should error"):
            # Reinject to trigger `boot`
            ct.spatial.morphologies[0] = {"select": "from_neuromorpho", "names": ["H1_m"]}
