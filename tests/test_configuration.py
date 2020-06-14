import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold import config
from scaffold.config import from_json
from scaffold.core import Scaffold
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
        Scaffold(config)

    def test_default_bootstrap(self):
        cfg = config.Configuration.default()
        Scaffold(cfg)

    def test_minimal_json_content_bootstrap(self):
        with open(minimal_config, "r") as f:
            content = f.read()
        config = from_json(data=content)
        Scaffold(config)

    def test_full_json_bootstrap(self):
        config = from_json(full_config)
        Scaffold(config)

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
            i = config.attr(type=int)

        t = Test()
        t2 = Test.__cast__({}, TestRoot())
        node_name = Test.str.get_node_name(t2)
        self.assertTrue(node_name.endswith(".str"), "str attribute misnomer")
        self.assertRaises(CastError, Test.__cast__, {"i": {}}, TestRoot())

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

        conf = {"l": {"e": {"name": "hi"}, "ss": {"name": "other"}}}
        t = Test.__cast__(conf, TestRoot())
        self.assertTrue(t.l.get_node_name().endswith(".l"), "Dict node name incorrect")
        self.assertEqual(len(t.l), 2, "Dict length incorrect")
        self.assertEqual(t.l.e, t.l["e"], "Dict access incorrect")
        self.assertEqual(type(t.l.e), Child, "Dict child class incorrect")
        self.assertEqual(t.l.e.key, "e", "Child key key incorrectly set")
        conf2 = {"l": {"e": {}, "ss": {"name": "other"}}}
        self.assertRaises(RequirementError, Test.__cast__, conf2, TestRoot())

        @config.node
        class TestSimple:
            l = config.dict(type=int)

        self.assertRaises(CastError, TestSimple.__cast__, conf2, TestRoot())


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

        @config.node
        class TestNormal:
            l = config.list(type=int, size=3)

        test_conf = {"l": [{"name": "hi"}, {"name": "other"}]}
        t = Test.__cast__(test_conf, TestRoot())
        self.assertEqual(len(t.l), 2, "List length incorrect")
        self.assertEqual(type(t.l[0]), Child, "List item class incorrect")
        self.assertEqual(t.l[1].index, 1, "Child index key incorrectly set")
        self.assertTrue(t.l.get_node_name().endswith(".l"), "Dict node name incorrect")
        self.assertRaises(CastError, TestSize.__cast__, test_conf, TestRoot())

        test_conf2 = {"l": [{"name": "hi"}, {}, {"name": "hi"}]}
        int_test = TestNormal.__cast__({"l": [1, 2, 3]}, TestRoot())
        self.assertEqual(int_test.l[2], 3)
        test_conf3 = {"l": [1, {}, 3]}
        self.assertRaises(CastError, TestNormal.__cast__, test_conf3, TestRoot())
        test_conf4 = {"l": [{"name": "hi"}, {}]}
        self.assertRaises(RequirementError, Test.__cast__, test_conf4, TestRoot())


class TestConfigRef(unittest.TestCase):
    def test_referencing(self):
        @config.node
        class Test:
            name = config.attr(required=True)
            name_ref = config.ref(lambda root, here: here, required=True, type=int)
            type_ref = config.ref(lambda root, here: here, ref_type=str)

        @config.root
        class Resolver:
            test = config.attr(type=Test, required=True)

        r = Resolver.__cast__(
            {"test": {"name": "Johnny", "name_ref": "name", "type_ref": "name"}}, None
        )
        self.assertEqual(r.test.name_ref, "Johnny")
        self.assertEqual(r.test.name_ref_reference, "name")

        self.assertRaises(
            ReferenceError,
            Resolver.__cast__,
            {"test": {"name": "Johnny", "name_ref": "nname"}},
            None,
        )


class TestHooks(unittest.TestCase):
    def test_hooks(self):
        class Exc(Exception):
            pass

        class Exc2(Exception):
            pass

        class Exc3(Exception):
            pass

        class Unhooked(Exception):
            pass

        class to_hook:
            def basic(self):
                raise Exc()

        class inherits_before_hooks(to_hook):
            pass

        def raise_before(self):
            raise Exc2()

        def raise_essential(self):
            raise Exc3()

        config.on("basic", to_hook)(raise_before)
        self.assertRaises(Exc, to_hook().basic)
        config.before("basic", to_hook)(raise_before)
        self.assertRaises(Exc2, to_hook().basic)

        class inherits_after_hooks(to_hook):
            pass

        self.assertRaises(Exc2, inherits_before_hooks().basic)
        self.assertRaises(Exc2, inherits_after_hooks().basic)

        class overwrites_nonessential(to_hook):
            def basic(self):
                raise Unhooked()

        config.before("basic", to_hook, essential=True)(raise_essential)
        self.assertRaises(Unhooked, overwrites_nonessential().basic)
        self.assertRaises(Exc3, config.run_hook, overwrites_nonessential(), "basic")
        self.assertRaises(Exc3, config.run_hook, to_hook(), "basic")

    def test_double_exec(self):
        a = 0

        class to_hook:
            def basic(self):
                nonlocal a
                a += 10

        def hook(self):
            nonlocal a
            a += 10

        config.before("basic", to_hook)(hook)
        config.after("basic", to_hook)(hook)
        to_hook().basic()
        self.assertEqual(a, 30, "If the function and both hooks fired, a should be 30.")

    def test_has_hook(self):
        class test:
            def __hook1__(self):
                pass

            def hook1(self):
                pass

            def hook2(self):
                pass

            def __hook3__(self):
                pass

        self.assertTrue(config.has_hook(test, "hook1"))
        self.assertTrue(config.has_hook(test, "hook2"))
        self.assertTrue(config.has_hook(test, "hook3"))
        self.assertFalse(config.has_hook(test, "hook4"))
        self.assertTrue(config.has_hook(test(), "hook1"))
        self.assertTrue(config.has_hook(test(), "hook2"))
        self.assertTrue(config.has_hook(test(), "hook3"))
        self.assertFalse(config.has_hook(test(), "hook4"))
