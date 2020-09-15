import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb import config
from bsb.config import from_json
from bsb.exceptions import *


def p(file):
    return os.path.join(os.path.dirname(__file__), "parser_tests", file)


def c(f):
    with open(p(f), "r") as fh:
        return fh.read()


class TestConfiguration(unittest.TestCase):
    def test_get_parser(self):
        config.get_parser("json")
        self.assertRaises(PluginError, config.get_parser, "doesntexist")

    def test_parse_empty_doc(self):
        tree, meta = config.get_parser("json").parse(c("doc.json"))
        self.assertEqual({}, tree, "'doc.json' parse should produce empty dict")

    def test_parse_basics(self):
        tree, meta = config.get_parser("json").parse(c("basics.json"))
        print(dict(tree))
        self.assertEqual(3, tree["list"][2], "Incorrectly parsed basic JSON")
        self.assertEqual(
            "just like that",
            tree["nest me hard"]["oh yea"],
            "Incorrectly parsed nested JSON",
        )
