import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb import config
from bsb.config import from_json
from bsb.exceptions import *
from bsb.models import Layer, CellType


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

    @unittest.expectedFailure
    def test_no_unknown_attributes(self):
        with self.assertWarns(ConfigurationWarning) as warning:
            config = from_json(minimal_config)

    @unittest.expectedFailure
    def test_full_no_unknown_attributes(self):
        with self.assertWarns(ConfigurationWarning) as warning:
            config = from_json(full_config)

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
            "catch_all",
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

    def test_requirement(self):
        @config.node
        class Test:
            name = config.attr(type=str, required=True)

        def special(value):
            raise RequirementError("special")

        @config.node
        class Test2:
            name = config.attr(type=str, required=special)

        @config.node
        class Test3:
            name = config.attr(type=str, required=lambda x: True)

        self.assertRaises(RequirementError, Test.__cast__, {}, TestRoot())
        self.assertRaisesRegex(
            RequirementError, r"special in ", Test2.__cast__, {}, TestRoot()
        )
        self.assertRaises(RequirementError, Test3.__cast__, {}, TestRoot())

        t = Test.__cast__({"name": "hello"}, TestRoot())
        self.assertEqual(
            t, Test.__cast__(t, None), "Already cast object should not be altered"
        )


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


@config.dynamic
class DynamicBase:
    name = config.attr(type=str, required=True)


@config.dynamic(attr_name="test")
class DynamicAttrBase:
    name = config.attr(type=str, required=True)


@config.dynamic(required=False, default="DynamicBaseDefault")
class DynamicBaseDefault:
    name = config.attr(type=str, required=True)


class NotInherited:
    pass


class TestDynamic(unittest.TestCase):
    def test_dynamic_requirements(self):
        self.assertRaisesRegex(
            RequirementError,
            "must contain a 'class' attribute",
            DynamicBase.__cast__,
            {},
            TestRoot(),
        )
        self.assertRaisesRegex(
            RequirementError,
            "must contain a 'test' attribute",
            DynamicAttrBase.__cast__,
            {},
            TestRoot(),
        )

    def test_dynamic(self):
        self.assertTrue(
            isinstance(
                DynamicBaseDefault.__cast__({"name": "ello"}, TestRoot()),
                DynamicBaseDefault,
            ),
            "Dynamic cast with default 'DynamicBaseDefault' should produce instance of type 'DynamicBaseDefault'",
        )

    def test_dynamic_inheritance(self):
        # Test that inheritance is enforced.
        # The cast should raise an UnfitClassCastError while the direct _load_class call
        # should raise a DynamicClassInheritanceError
        self.assertRaises(
            UnfitClassCastError,
            DynamicBase.__cast__,
            {"name": "ello", "class": "NotInherited"},
            TestRoot(),
        )
        self.assertRaises(
            DynamicClassInheritanceError,
            sys.modules["bsb.config._make"]._load_class,
            NotInherited,
            [],
            interface=DynamicBase,
        )
        # TODO: Test that the error message shows the mapped class name if a classmap exists

    def test_dynamic_missing(self):
        # Test that non existing classes raise the UnresolvedClassCastError.
        self.assertRaises(
            UnresolvedClassCastError,
            DynamicBase.__cast__,
            {"name": "ello", "class": "DoesntExist"},
            TestRoot(),
        )

    def test_dynamic_module_path(self):
        # Test that the module path can help find classes.
        self.assertEqual(
            sys.modules["bsb.config._make"]._load_class(
                "NotInherited", [NotInherited.__module__]
            ),
            NotInherited,
        )
        # Test that without the module path the same class can't be found
        self.assertRaises(
            DynamicClassNotFoundError,
            sys.modules["bsb.config._make"]._load_class,
            "NotInherited",
            [],
        )


@config.dynamic(
    classmap={"a": "ClassmapChildA", "b": "ClassmapChildB", "d": "ClassmapChildD",}
)
class ClassmapParent:
    pass


class ClassmapChildA(ClassmapParent):
    pass


class ClassmapChildB(ClassmapParent):
    pass


class TestClassmaps(unittest.TestCase):
    def test_dynamic_classmap(self):
        a = ClassmapParent.__cast__({"class": "a"}, TestRoot())
        self.assertEqual(ClassmapChildA, a.__class__, "Classmap failed")
        b = ClassmapParent.__cast__({"class": "b"}, TestRoot())
        self.assertEqual(ClassmapChildB, b.__class__, "Classmap failed")

    def test_missing_classmap_entry(self):
        self.assertRaises(
            UnresolvedClassCastError, ClassmapParent.__cast__, {"class": "c"}, TestRoot()
        )

    def test_missing_classmap_class(self):
        self.assertRaisesRegex(
            UnresolvedClassCastError,
            "'d' \(mapped to 'ClassmapChildD'\)",
            ClassmapParent.__cast__,
            {"class": "d"},
            TestRoot(),
        )


@config.dynamic(auto_classmap=True)
class CleanAutoClassmap:
    pass


class AutoClassmapChildA(CleanAutoClassmap, classmap_entry="a"):
    pass


class AutoClassmapChildB(CleanAutoClassmap, classmap_entry="b"):
    pass


class UnregisteredAutoClassmapChildC(CleanAutoClassmap):
    pass


@config.dynamic(auto_classmap=True, classmap={"d": "AutoClassmapChildD"})
class DirtyAutoClassmap:
    pass


class AutoClassmapChildC(DirtyAutoClassmap, classmap_entry="c"):
    pass


class AutoClassmapChildD(DirtyAutoClassmap):
    pass


class TestAutoClassmap(unittest.TestCase):
    def test_dynamic_autoclassmap(self):
        self.assertEqual(
            {"a": AutoClassmapChildA, "b": AutoClassmapChildB},
            CleanAutoClassmap._config_dynamic_classmap,
            "Automatic classmap incorrect",
        )

    def test_combined_autoclassmap(self):
        self.assertEqual(
            {"c": AutoClassmapChildC, "d": "AutoClassmapChildD"},
            DirtyAutoClassmap._config_dynamic_classmap,
            "Automatic classmap with manual entry incorrect",
        )


class TestWalk(unittest.TestCase):
    def test_walk_values(self):
        @config.node
        class Deeper:
            ey = config.list(type=int, required=True)

        @config.node
        class Base:
            att = config.attr()
            deep = config.attr(type=Deeper)

        @config.root
        class Root:
            smth = config.attr(type=Base)

        b = Root.__cast__({"smth": {"att": "hello", "deep": {"ey": [1, 2, 3]}}}, None)
        iter_collected = [*sys.modules["bsb.config._make"].walk_node_values(b)]
        self.assertEqual(len(iter_collected), 7)


from bsb.config import types


class TestTypes(unittest.TestCase):
    def test_in(self):
        @config.node
        class Test:
            c = config.attr(type=types.in_([1, 2, 3]))

        b = Test.__cast__({"c": 3}, TestRoot())
        self.assertEqual(b.c, 3)
        self.assertRaises(CastError, Test.__cast__, {"c": 4}, TestRoot())

    def test_in_inf(self):
        class Fib:
            def __call__(self):
                a, b = 0, 1
                while True:
                    yield a
                    a, b = b, a + b

            def __contains__(self, x):
                m = -1
                f = self()
                while x > m:
                    m = next(f)
                    if x == m:
                        return True
                return False

            def __str__(self):
                return "the fibonacci series"

        @config.node
        class Test:
            c = config.attr(type=types.in_(Fib()))

        b = Test.__cast__({"c": 13}, TestRoot())
        self.assertRaisesRegex(
            CastError, "fibonacci", Test.__cast__, {"c": 14}, TestRoot()
        )

    def test_multiple_types(self):
        @config.node
        class TestS:
            c = config.attr(type=types.or_(int, str))

        @config.node
        class TestF:
            c = config.attr(type=types.or_(int, int))

        b = TestS.__cast__({"c": "h"}, TestRoot())
        self.assertEqual(b.c, "h")
        self.assertRaises(CastError, TestF.__cast__, {"c": "h"}, TestRoot())

    def test_scalar_expand(self):
        @config.node
        class Test:
            c = config.attr(type=types.scalar_expand(int, expand=lambda s: [s, s]))

        b = Test.__cast__({"c": 2}, TestRoot())
        self.assertEqual(b.c, [2, 2])

    def test_list(self):
        @config.node
        class Test:
            c = config.attr(type=types.list(int))
            d = config.attr(type=types.list(int, size=3))

        b = Test.__cast__({"c": [2, 2]}, TestRoot())
        self.assertEqual(b.c, [2, 2])
        b = Test.__cast__({"c": None}, TestRoot())
        self.assertEqual(b.c, None)
        self.assertRaises(CastError, Test.__cast__, {"c": [2, "f"]}, TestRoot())
        self.assertRaises(CastError, Test.__cast__, {"d": [2, 2]}, TestRoot())

    def test_fraction(self):
        @config.node
        class Test:
            c = config.attr(type=types.fraction())

        b = Test.__cast__({"c": 0.1}, TestRoot())
        self.assertEqual(b.c, 0.1)
        self.assertRaises(CastError, Test.__cast__, {"c": -0.1}, TestRoot())

    # def test_constant_distribution(self):
    #     raise NotImplementedError("Luie zak")
    #
    # def test_distribution(self):
    #     raise NotImplementedError("Luie zak")
    #
    # def test_evaluation(self):
    #     raise NotImplementedError("Luie zak")
