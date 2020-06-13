import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.config import from_json
from scaffold.exceptions import *


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


minimal_config = relative_to_tests_folder("configs/test_minimal.json")
full_config = relative_to_tests_folder("configs/test_full_v4.json")


def as_json(f):
    import json

    with open(f, "r") as fh:
        return json.load(fh)


class TestConfiguration(unittest.TestCase):
    def test_minimal_json_bootstrap(self):
        config = from_json(minimal_config)

    def test_minimal_json_content_bootstrap(self):
        with open(minimal_config, "r") as f:
            content = f.read()
        config = from_json(data=content)

    def test_full_json_bootstrap(self):
        config = from_json(full_config)

    def test_missing_nodes(self):
        self.assertRaises(RequirementError, from_json, data="""{}""")

    def test_unknown_attributes(self):
        data = as_json(minimal_config)
        data["shouldntexistasattr"] = 15
        with self.assertWarns(ConfigurationWarning) as warning:
            config = from_json(data=data)

        self.assertIn("""Unknown attribute 'shouldntexistasattr'""", str(warning.warning))
        self.assertIn("""in {root}""", str(warning.warning))
