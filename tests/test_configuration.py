import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold import config
from scaffold.config import from_json
from scaffold.exceptions import *


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


minimal_config = relative_to_tests_folder("configs/test_minimal.json")
full_config = relative_to_tests_folder("configs/test_full_v4.json")


@config.root
class TestRoot:
    pass


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


class TestConfigAttrs(unittest.TestCase):
    def test_components_on_module(self):
        t = [
            "attr",
            "ref",
            "dict",
            "list",
            "dynamic",
            "node",
            "root",
            "slot",
            "pluggable",
        ]
        for a in t:
            with self.subTest(check=a):
                self.assertTrue(
                    hasattr(config, a), "Missing {} in config module".format(a)
                )

    def test_empty_test_node(self):
        @config.node
        class Test:
            pass

        self.assertTrue(hasattr(Test, "_config_attrs"))
        t = Test()
        t2 = Test.__cast__({}, TestRoot())

    def test_attr(self):
        @config.node
        class Test:
            str = config.attr()

        t = Test()
        t2 = Test.__cast__({}, TestRoot())
        node_name = Test.str.get_node_name(t2)
        self.assertTrue(node_name.endswith(".str"), "str attribute misnomer")

    def test_inheritance(self):
        @config.node
        class Test:
            name = config.attr(type=str, required=True)

        class Child(Test):
            pass

        c = Child.__cast__({"name": "Hello"}, TestRoot())
        self.assertRaises(RequirementError, Child.__cast__, {}, TestRoot())


class TestConfigDict(unittest.TestCase):
    def test_dict_attr(self):
        @config.node
        class Child:
            key = config.attr(key=True)
            name = config.attr(type=str, required=True)

        @config.node
        class Test:
            l = config.dict(type=Child, required=True)

        t = Test.__cast__(
            {"l": {"e": {"name": "hi"}, "ss": {"name": "other"}}}, TestRoot()
        )
        self.assertEqual(len(t.l), 2, "Dict length incorrect")
        self.assertEqual(t.l.e, t.l["e"], "Dict access incorrect")
        self.assertEqual(type(t.l.e), Child, "Dict child class incorrect")
        self.assertEqual(t.l.e.key, "e", "Child key key incorrectly set")


class TestConfigList(unittest.TestCase):
    def test_list_attr(self):
        @config.node
        class Child:
            index = config.attr(key=True)
            name = config.attr(type=str, required=True)

        @config.node
        class Test:
            l = config.list(type=Child, required=True)

        @config.node
        class TestSize:
            l = config.list(type=Child, required=True, size=3)

        test_conf = {"l": [{"name": "hi"}, {"name": "other"}]}
        t = Test.__cast__(test_conf, TestRoot())
        self.assertEqual(len(t.l), 2, "List length incorrect")
        self.assertEqual(type(t.l[0]), Child, "List item class incorrect")
        self.assertEqual(t.l[1].index, 1, "Child index key incorrectly set")
        self.assertTrue(t.l.get_node_name().endswith(".l"), "Dict node name incorrect")
        self.assertRaises(CastError, TestSize.__cast__, test_conf, TestRoot())

        test_conf2 = {"l": [{"name": "hi"}, {}, {"name": "hi"}]}
        self.assertRaises(RequirementError, TestSize.__cast__, test_conf2, TestRoot())


class TestConfigRef(unittest.TestCase):
    def test_referencing(self):
        @config.node
        class Test:
            name = config.attr(required=True)
            name_ref = config.ref(lambda root, here: here, required=True, type=int)

        @config.root
        class Resolver:
            test = config.attr(type=Test, required=True)

        r = Resolver.__cast__({"test": {"name": "Johnny", "name_ref": "name"}}, None)
        self.assertEqual(r.test.name_ref, "Johnny")
        self.assertEqual(r.test.name_ref_reference, "name")
