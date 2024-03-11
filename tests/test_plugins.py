import unittest

from bsb_test import spoof_plugin

from bsb import PlacementStrategy


class A(PlacementStrategy):
    def place(self, chunk, indicators):
        pass


class TestComponentPlugins(unittest.TestCase):
    @spoof_plugin(
        "components", "super", {"bsb.placement.strategy.PlacementStrategy": {"__a__": A}}
    )
    def test_classmap_dict(self):
        a = PlacementStrategy(strategy="__a__", cell_types=[], partitions=[])
