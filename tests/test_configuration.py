import unittest
import sys
import numpy as np
import json
from bsb.core import Scaffold
from bsb import config
from bsb.config import from_json, Configuration, _attrs
from bsb.config import types
from bsb.exceptions import (
    CfgReferenceError,
    RequirementError,
    ConfigurationWarning,
    CastError,
    UnfitClassCastError,
    DynamicClassInheritanceError,
    UnresolvedClassCastError,
    DynamicObjectNotFoundError,
    ClassMapMissingError,
)
from bsb.topology.region import RegionGroup
from bsb.unittest import get_config_path

minimal_config = get_config_path("test_minimal.json")
full_config = get_config_path("test_full_v4.json")


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

    def test_no_unknown_attributes(self):
        try:
            with self.assertWarns(ConfigurationWarning) as cm:
                from_json(minimal_config)
            self.fail(f"Unknown configuration attributes detected: {cm.warning}")
        except AssertionError:
            pass

    def test_full_no_unknown_attributes(self):
        try:
            with self.assertWarns(ConfigurationWarning) as cm:
                from_json(full_config)
            self.fail(f"Unknown configuration attributes detected: {cm.warning}")
        except AssertionError:
            pass

    def test_unknown_attributes(self):
        data = as_json(minimal_config)
        data["shouldntexistasattr"] = 15
        with self.assertWarns(ConfigurationWarning) as warning:
            from_json(data=data)

        self.assertIn(
            """Unknown attribute: 'shouldntexistasattr'""", str(warning.warning)
        )


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
        Test()
        Test({}, _parent=TestRoot())

    def test_attr(self):
        @config.node
        class Test:
            str = config.attr()
            i = config.attr(type=int)

        Test()
        t2 = Test({}, _parent=TestRoot())
        node_name = Test.str.get_node_name(t2)
        self.assertTrue(node_name.endswith(".str"), "str attribute misnomer")
        with self.assertRaises(CastError):
            Test({"i": {}}, _parent=TestRoot())

    def test_inheritance(self):
        @config.node
        class Test:
            name = config.attr(type=str, required=True)

        class Child(Test):
            pass

        Child({"name": "Hello"}, _parent=TestRoot())
        with self.assertRaises(RequirementError):
            Child({}, _parent=TestRoot())

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

        with self.assertRaises(RequirementError):
            Test({}, _parent=TestRoot())
        with self.assertRaisesRegex(RequirementError, r"special"):
            Test2({}, _parent=TestRoot())
        with self.assertRaises(RequirementError):
            Test3({}, _parent=TestRoot())
        t = Test({"name": "hello"}, _parent=TestRoot())
        self.assertEqual(t, Test(t), "Already cast object should not be altered")


class TestConfigDict(unittest.TestCase):
    def test_dict_attr(self):
        @config.node
        class Child:
            key = config.attr(key=True)
            name = config.attr(type=str, required=True)

        @config.node
        class Test:
            dictattr = config.dict(type=Child, required=True)

        conf = {"dictattr": {"e": {"name": "hi"}, "ss": {"name": "other"}}}
        t = Test(conf, _parent=TestRoot())
        self.assertTrue(
            t.dictattr.get_node_name().endswith(".dictattr"), "Dict node name incorrect"
        )
        self.assertEqual(len(t.dictattr), 2, "Dict length incorrect")
        self.assertEqual(t.dictattr.e, t.dictattr["e"], "Dict access incorrect")
        self.assertEqual(type(t.dictattr.e), Child, "Dict child class incorrect")
        self.assertEqual(t.dictattr.e._config_key, "e", "Child key key incorrectly set")
        conf2 = {"dictattr": {"e": {}, "ss": {"name": "other"}}}
        with self.assertRaises(RequirementError):
            Test(conf2, _parent=TestRoot())

        @config.node
        class TestSimple:
            dictattr = config.dict(type=int)

        with self.assertRaises(CastError):
            TestSimple(conf2, _parent=TestRoot())


class TestConfigList(unittest.TestCase):
    def test_list_attr(self):
        @config.node
        class Child:
            index = config.attr(key=True)
            name = config.attr(type=str, required=True)

        @config.node
        class Test:
            listattr = config.list(type=Child, required=True)

        @config.node
        class TestSize:
            listattr = config.list(type=Child, required=True, size=3)

        @config.node
        class TestNormal:
            listattr = config.list(type=int, size=3)

        test_conf = {"listattr": [{"name": "hi"}, {"name": "other"}]}
        t = Test(test_conf, _parent=TestRoot())
        self.assertEqual(len(t.listattr), 2, "List length incorrect")
        self.assertEqual(type(t.listattr[0]), Child, "List item class incorrect")
        self.assertEqual(
            t.listattr[1]._config_index, 1, "Child index key incorrectly set"
        )
        self.assertTrue(
            t.listattr.get_node_name().endswith(".listattr"), "Dict node name incorrect"
        )
        with self.assertRaises(CastError):
            TestSize(test_conf, _parent=TestRoot())

        int_test = TestNormal({"listattr": [1, 2, 3]}, _parent=TestRoot())
        self.assertEqual(int_test.listattr[2], 3)
        test_conf3 = {"listattr": [1, {}, 3]}
        with self.assertRaises(CastError):
            TestNormal(test_conf3, _parent=TestRoot())
        test_conf4 = {"listattr": [{"name": "hi"}, {}]}
        with self.assertRaises(RequirementError):
            Test(test_conf4, TestRoot())

    def test_catch_dict(self):
        @config.node
        class TestNormal:
            listattr = config.list(type=int, size=3)

        with self.assertRaises(CastError, msg="Regression of #457"):
            TestNormal(listattr={5: "hey", 6: "boo"})


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

        r = Resolver({"test": {"name": "Johnny", "name_ref": "name", "type_ref": "name"}})
        self.assertEqual(r.test.name_ref, "Johnny")
        self.assertEqual(r.test.name_ref_reference, "name")

        with self.assertRaises(CfgReferenceError):
            Resolver({"test": {"name": "Johnny", "name_ref": "nname"}})


@config.root
class BootRoot:
    empty_list = config.reflist(lambda r, h: h)
    none = config.reflist(lambda r, h: h, default=None)


def _bootstrap(cfg, scaffold):
    for node in config.walk_nodes(cfg):
        node.scaffold = scaffold
        config.run_hook(node, "boot")
    return cfg


class TestConfigRefList(unittest.TestCase):
    def test_reflist_defaults(self):
        root = BootRoot({})
        _bootstrap(root, None)
        self.assertEqual([], root.empty_list)
        self.assertEqual([], root.none)

    def test_non_iterable(self):
        root = BootRoot({})
        with self.assertRaises(CfgReferenceError):
            root.empty_list = 5


class HasRefsReference:
    def __call__(self, r, h):
        return r

    def is_ref(self, value):
        return isinstance(value, HasRefs)


@config.node
class HasLists:
    cfglist = config.list()
    reflist = config.reflist(HasRefsReference())
    list = config.attr(type=list)


@config.node
class HasRefs:
    ref_cfg = config.ref(lambda r, h: r, ref_type=HasLists, populate="cfglist")
    ref = config.ref(lambda r, h: r, ref_type=HasLists, populate="list")
    ref_ref = config.ref(lambda r, h: r, ref_type=HasLists, populate="reflist")
    ref_ref2 = config.ref(lambda r, h: r, ref_type=HasLists, populate="reflist")
    reflist = config.reflist(lambda r, h: r, ref_type=HasLists, populate="list")


@config.root
class PopRoot:
    lists = config.attr(type=HasLists)
    referrers = config.attr(type=HasRefs)
    refs2 = config.attr(type=HasRefs)


class TestPopulate(unittest.TestCase):
    def test_populate(self):
        pop_root = PopRoot(
            {"lists": {}, "referrers": {"ref_cfg": "lists", "ref": "lists"}}
        )
        _bootstrap(pop_root, None)
        self.assertEqual(1, len(pop_root.lists.cfglist), "`populate` config.list failure")
        self.assertEqual(
            pop_root.referrers,
            pop_root.lists.cfglist[0],
            "`populate` config.list failure",
        )
        self.assertEqual(1, len(pop_root.lists.list), "`populate` list failure")
        self.assertEqual(
            pop_root.referrers, pop_root.lists.list[0], "`populate` list failure"
        )

    def test_populate_reflist(self):
        pop_root = PopRoot({"lists": {}, "referrers": {"ref_ref": "lists"}})
        _bootstrap(pop_root, None)
        self.assertEqual(
            1, len(pop_root.lists.reflist), "`populate` config.reflist failure"
        )
        self.assertEqual(
            pop_root.referrers,
            pop_root.lists.reflist[0],
            "`populate` config.reflist failure",
        )

    def test_populate_reflist_unique(self):
        conf = {
            "lists": {"reflist": []},
            "referrers": {"ref_ref": "lists", "ref_ref2": "lists"},
        }
        pop_root = PopRoot(conf)
        _bootstrap(pop_root, None)
        self.assertEqual(1, len(pop_root.lists.reflist))
        self.assertEqual(pop_root.referrers, pop_root.lists.reflist[0])

    def test_populate_reflist_with_refkeys_unique(self):
        # Test that unicity also takes into account existing reference keys.
        pop_root = PopRoot(
            {
                "lists": {"reflist": ["referrers"]},
                "referrers": {"ref_ref": "lists", "ref_ref2": "lists"},
            },
        )
        _bootstrap(pop_root, None)
        self.assertEqual(1, len(pop_root.lists.reflist))
        self.assertEqual(pop_root.referrers, pop_root.lists.reflist[0])

    def test_populate_reflist_not_unique(self):
        HasRefs.ref_ref.pop_unique = False
        pop_root = PopRoot(
            {
                "lists": {"reflist": ["referrers", "refs2"]},
                "referrers": {"ref_ref": "lists"},
                "refs2": {"ref_ref": "lists"},
            }
        )
        _bootstrap(pop_root, None)
        self.assertEqual(4, len(pop_root.lists.reflist))
        self.assertEqual(pop_root.referrers, pop_root.lists.reflist[0])
        HasRefs.ref_ref.pop_unique = True

    def test_reflist_populate(self):
        pop_root = PopRoot(
            {"lists": {}, "referrers": {"reflist": ["lists", "lists", "lists"]}}
        )
        _bootstrap(pop_root, None)
        self.assertEqual(1, len(pop_root.lists.list), "Reflist did not populate uniquely")
        self.assertEqual(pop_root.referrers, pop_root.lists.list[0])

    def test_no_unique_reflist_populate(self):
        HasRefs.reflist.pop_unique = False
        pop_root = PopRoot(
            {"lists": {}, "referrers": {"reflist": ["lists", "lists", "lists"]}}
        )
        _bootstrap(pop_root, None)
        self.assertEqual(3, len(pop_root.lists.list))
        self.assertEqual(pop_root.referrers, pop_root.lists.list[0])
        HasRefs.reflist.pop_unique = True


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


@config.dynamic(required=True)
class DynamicBaseRequired:
    pass


class DynamicChildNotRequired(DynamicBaseRequired):
    pass


class UndecoratedChildNotRequired(DynamicBaseRequired):
    pass


class NotInherited:
    pass


class TestDynamic(unittest.TestCase):
    def test_dynamic_requirements(self):
        with self.assertRaisesRegex(RequirementError, "must contain a 'cls' attribute"):
            DynamicBase({})
        with self.assertRaisesRegex(RequirementError, "must contain a 'test' attribute"):
            DynamicAttrBase({})

    def test_dynamic(self):
        self.assertTrue(
            isinstance(
                DynamicBaseDefault({"name": "ello"}),
                DynamicBaseDefault,
            ),
            "Dynamic cast with default 'DynamicBaseDefault' should produce instance"
            " of type 'DynamicBaseDefault'",
        )

    def test_dynamic_inheritance(self):
        # Test that inheritance is enforced.
        # The cast should raise an UnfitClassCastError while the direct _load_class call
        # should raise a DynamicClassInheritanceError
        with self.assertRaises(UnfitClassCastError):
            DynamicBase({"name": "ello", "cls": "NotInherited"})
        with self.assertRaises(DynamicClassInheritanceError):
            sys.modules["bsb.config._make"]._load_class(
                NotInherited,
                [],
                interface=DynamicBase,
            )
        # TODO: Test error message shows the mapped class name if a classmap exists

    def test_dynamic_missing(self):
        # Test that non existing classes raise the UnresolvedClassCastError.
        with self.assertRaises(UnresolvedClassCastError):
            DynamicBase({"name": "ello", "cls": "DoesntExist"}, _parent=TestRoot())

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
            DynamicObjectNotFoundError,
            sys.modules["bsb.config._make"]._load_class,
            "NotInherited",
            [],
        )

    def test_direct_dynamic_child(self):
        # Test that even if a dynamic attr is required on the parent, we can directly
        # construct the child. This has to be so because the chosen child class already
        # explicitly satisfies the dynamic requirement.
        direct_child = DynamicChildNotRequired()
        # Undecorated child classes are not endorsed, but just in case.
        undecorated_direct_child = UndecoratedChildNotRequired()


@config.dynamic(
    classmap={
        "a": "ClassmapChildA",
        "b": "ClassmapChildB",
        "d": "ClassmapChildD",
    }
)
class ClassmapParent:
    pass


class ClassmapChildA(ClassmapParent):
    pass


class ClassmapChildB(ClassmapParent):
    pass


class TestClassmaps(unittest.TestCase):
    def test_dynamic_classmap(self):
        a = ClassmapParent({"cls": "a"}, _parent=TestRoot())
        self.assertEqual(ClassmapChildA, a.__class__, "Classmap failed")
        b = ClassmapParent({"cls": "b"}, _parent=TestRoot())
        self.assertEqual(ClassmapChildB, b.__class__, "Classmap failed")

    def test_missing_classmap_entry(self):
        with self.assertRaises(UnresolvedClassCastError):
            ClassmapParent({"cls": "c"}, _parent=TestRoot())

    def test_missing_classmap_class(self):
        with self.assertRaisesRegex(
            UnresolvedClassCastError, "'d' \\(mapped to 'ClassmapChildD'\\)"
        ):
            ClassmapParent({"cls": "d"}, _parent=TestRoot())


@config.dynamic(auto_classmap=True)
class CleanAutoClassmap:
    pass


class AutoClassmapChildA(CleanAutoClassmap, classmap_entry="a"):
    pass


class AutoClassmapChildB(CleanAutoClassmap, classmap_entry="b"):
    pass


class UnregisteredAutoClassmapChildC(CleanAutoClassmap, classmap_entry=None):
    pass


class SnakeChild(CleanAutoClassmap):
    pass


@config.dynamic(
    auto_classmap=True, classmap={"d": "AutoClassmapChildD"}, classmap_entry=None
)
class DirtyAutoClassmap:
    pass


class AutoClassmapChildC(DirtyAutoClassmap, classmap_entry="c"):
    pass


class AutoClassmapChildD(DirtyAutoClassmap, classmap_entry=None):
    pass


class TestAutoClassmap(unittest.TestCase):
    def test_dynamic_autoclassmap(self):
        self.assertEqual(
            {
                "a": AutoClassmapChildA,
                "b": AutoClassmapChildB,
                # Test snake casing, see #880
                "snake_child": SnakeChild,
            },
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

        b = Root({"smth": {"att": "hello", "deep": {"ey": [1, 2, 3]}}})
        iter_collected = [*sys.modules["bsb.config._make"].walk_node_values(b)]
        self.assertEqual(len(iter_collected), 7)


class TestTypes(unittest.TestCase):
    def test_in(self):
        @config.node
        class Test:
            c = config.attr(type=types.in_([1, 2, 3]))

        b = Test({"c": 3}, _parent=TestRoot())
        self.assertEqual(b.c, 3)
        self.assertRaises(CastError, Test, {"c": 4}, _parent=TestRoot())

    def test_int(self):
        @config.root
        class Test:
            a = config.attr(type=types.int())
            b = config.attr(type=types.int(min=0))
            c = config.attr(type=types.int(max=0))
            d = config.attr(type=types.int(min=0, max=10))

        # Test basics
        cfg = Test({"a": 5, "b": 5, "c": -5, "d": 5})
        self.assertEqual(5, cfg.a)
        self.assertEqual(5, cfg.b)
        self.assertEqual(-5, cfg.c)
        self.assertEqual(5, cfg.d)
        # Test edge cases
        Test(b=0)
        Test(c=0)
        with self.assertRaises(CastError):
            Test(b=-10)
        with self.assertRaises(CastError):
            Test(c=10)
        with self.assertRaises(CastError):
            Test(d=-5)
        with self.assertRaises(CastError):
            Test(d=15)
        # Test rounding & conversion
        self.assertEqual(5, Test(a=5.5).a)
        self.assertEqual(5, Test(a=5.4).a)
        self.assertEqual(5, Test(a=5.6).a)
        self.assertEqual(5, Test(a=5.0).a)
        self.assertEqual(5, Test(a="5").a)

    def test_float(self):
        @config.root
        class Test:
            a = config.attr(type=types.float())
            b = config.attr(type=types.float(min=0))
            c = config.attr(type=types.float(max=0))
            d = config.attr(type=types.float(min=0, max=10))

        # Test basics
        cfg = Test({"a": 5.2, "b": 5.2, "c": -5.2, "d": 5.2})
        self.assertEqual(5.2, cfg.a)
        self.assertEqual(5.2, cfg.b)
        self.assertEqual(-5.2, cfg.c)
        self.assertEqual(5.2, cfg.d)
        # Test edge cases
        Test(b=0)
        Test(c=0)
        with self.assertRaises(CastError):
            Test(b=-10)
        with self.assertRaises(CastError):
            Test(c=10)
        with self.assertRaises(CastError):
            Test(d=-5)
        with self.assertRaises(CastError):
            Test(d=15)
        # Test rounding & conversion
        self.assertEqual(5.0, Test(a=5).a)
        self.assertEqual(5.5, Test(a=5.5).a)
        self.assertEqual(5.0, Test(a="5").a)
        self.assertEqual(5.0, Test(a="5.").a)
        self.assertEqual(5.6, Test(a="5.6").a)
        self.assertEqual(0.074, Test(a="7.4e-02").a)

    def test_number(self):
        @config.root
        class Test:
            a = config.attr(type=types.number())
            b = config.attr(type=types.number(min=0))
            c = config.attr(type=types.number(max=0))
            d = config.attr(type=types.number(min=0, max=10))

        # Test basics
        cfg = Test({"a": 5, "b": 5.0, "c": -5.2, "d": 5.2})
        self.assertEqual(5, cfg.a)
        self.assertEqual(int, type(cfg.a))
        self.assertEqual(5.0, cfg.b)
        self.assertEqual(float, type(cfg.b))
        with self.assertRaises(CastError):
            Test(b=-10)
        with self.assertRaises(CastError):
            Test(c=10)
        with self.assertRaises(CastError):
            Test(d=-5)
        with self.assertRaises(CastError):
            Test(d=15)

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

        Test({"c": 13}, _parent=TestRoot())
        self.assertRaisesRegex(
            CastError, "fibonacci", Test, {"c": 14}, _parent=TestRoot()
        )

    def test_multiple_types(self):
        @config.node
        class TestS:
            c = config.attr(type=types.or_(int, str))

        @config.node
        class TestF:
            c = config.attr(type=types.or_(int, int))

        b = TestS({"c": "h"}, _parent=TestRoot())
        self.assertEqual(b.c, "h")
        self.assertRaises(CastError, TestF, {"c": "h"}, _parent=TestRoot())

    def test_scalar_expand(self):
        @config.node
        class Test:
            c = config.attr(type=types.scalar_expand(int, expand=lambda s: [s, s]))

        b = Test({"c": 2}, _parent=TestRoot())
        self.assertEqual(b.c, [2, 2])

    def test_list(self):
        @config.node
        class Test:
            c = config.attr(type=types.list(int))
            d = config.attr(type=types.list(int, size=3))

        b = Test({"c": [2, 2]}, _parent=TestRoot())
        self.assertEqual([2, 2], b.c)
        b = Test({"c": None}, _parent=TestRoot())
        self.assertEqual(None, types.list()(None))
        self.assertEqual(None, b.c)
        self.assertRaises(CastError, Test, {"c": [2, "f"]}, _parent=TestRoot())
        self.assertRaises(CastError, Test, {"d": [2, 2]}, _parent=TestRoot())

    def test_dict(self):
        @config.root
        class Test:
            c = config.attr(type=types.dict())
            d = config.attr(type=types.dict(int))

        self.assertEqual({"a": "b"}, Test({"c": {"a": "b"}}).c)
        self.assertEqual({"a": "5"}, Test({"c": {"a": 5}}).c)
        self.assertEqual({"a": 5}, Test({"d": {"a": 5}}).d)
        self.assertEqual(None, types.dict()(None))

    def test_fraction(self):
        @config.node
        class Test:
            c = config.attr(type=types.fraction())

        b = Test({"c": 0.1}, _parent=TestRoot())
        self.assertEqual(b.c, 0.1)
        self.assertRaises(CastError, Test, {"c": -0.1}, _parent=TestRoot())

    def test_constant_distribution(self):
        @config.root
        class Test:
            c = config.attr(type=types.distribution())

        a = Test({"c": 1})
        self.assertTrue(np.array_equal(np.ones(5), a.c.draw(5)))

    def test_distribution(self):
        import scipy.stats.distributions

        @config.root
        class Test:
            c = config.attr(type=types.distribution())

        # Check basic function
        a = Test({"c": {"distribution": "alpha", "a": 3, "loc": 2, "scale": 2.5}})
        equivalent = scipy.stats.distributions.alpha(a=3, loc=2, scale=2.5)
        self.assertEqual(type(equivalent), type(a.c._distr))
        self.assertEqual(equivalent.pdf(5.9), a.c.pdf(5.9))

        with self.assertRaises(RequirementError):
            a = Test({"c": {"a": 3, "loc": 2, "scale": 2.5}})

        with self.assertRaises(CastError):
            # Check that underlying errors are also caught
            a = Test({"c": {"distribution": "alpha"}})
            # Should we add a test to see that the underlying message is passed?

        with self.assertRaises(CastError):
            # Check that unknown distributions throw a CastError
            a = Test({"c": {"distribution": "alphaa"}})

    def test_evaluation(self):
        @config.root
        class Test:
            c = config.attr(type=types.evaluation())

        def _eval(statement, **vars):
            return Test(c={"statement": statement, "variables": vars}).c

        self.assertEqual(3, _eval(3))
        self.assertEqual(3, _eval("3"))
        self.assertEqual(3, _eval("5 - 2"))
        self.assertEqual(3, _eval("v - 2", v=5))
        self.assertEqual(3, _eval("np.array([v - 2])[0]", v=5))

    def test_class(self):
        @config.root
        class Test:
            a = config.attr(type=types.class_())
            b = config.attr(type=types.class_(module_path=["test_configuration"]))

        cfg = Test({"a": "test_configuration.MyTestClass", "b": "MyTestClass"})
        self.assertEqual(MyTestClass, cfg.a)
        self.assertEqual(MyTestClass, cfg.b)

        with self.assertRaises(CastError):
            cfg = Test({"a": "MyTestClass"})

    def test_in_classmap(self):
        @config.root
        class Test:
            a = config.attr(type=types.in_classmap())
            c = config.attr(type=Classmap2Parent)

        t = Test({"c": {"cls": "a"}})
        # The `Test` class itself has no classmap so using the `in_classmap` validator is
        # incorrect and should raise an error.
        with self.assertRaises(ClassMapMissingError):
            Test.a.type("a", _parent=t, _key="a")
        # `in_classmap` is a restrictive type handler that should only allow the classmap
        # strings to be given and not the classes themselves.
        with self.assertRaises(CastError):
            Test(c={"cls": Classmap2ChildA})
        # If a string is valid it should be left untouched.
        self.assertEqual("a", Classmap2Parent.cls.type("a", _parent=t.c, _key="cls"))
        # If a string is invalid a cast error should be raised
        with self.assertRaises(CastError):
            Classmap2Parent.cls.type("aa", _parent=t.c, _key="cls")
        # `d` is in the classmap, but not mapped to an actual class. This test verifies
        # that the `in_classmap` type handler will nod and allow `d` to be pass and burn
        # later on, where it is supposed to burn.
        with self.assertRaises(UnresolvedClassCastError):
            self.assertEqual("d", Test(c={"cls": "d"}))


@config.dynamic(
    type=types.in_classmap(),
    classmap={
        "a": "Classmap2ChildA",
        "b": "Classmap2ChildB",
        "d": "Classmap2ChildD",
    },
)
class Classmap2Parent:
    pass


class Classmap2ChildA(Classmap2Parent):
    pass


class Classmap2ChildB(Classmap2Parent):
    pass


class MyTestClass:
    pass


class TestTreeing(unittest.TestCase):
    def surjective(self, name, cls, ref, tree):
        # Test that the tree projects onto the ref
        with self.subTest(name=name):
            cfg = cls(tree)
            new_tree = cfg.__tree__()
            self.assertEqual(json.dumps(ref, indent=2), json.dumps(new_tree, indent=2))

    def bijective(self, name, cls, tree):
        # Test that the tree and its config projection are the same in JSON
        with self.subTest(name=name):
            cfg = cls(tree)
            new_tree = cfg.__tree__()
            self.assertEqual(json.dumps(tree, indent=2), json.dumps(new_tree, indent=2))
            return cfg, new_tree

    def test_empty(self):
        @config.root
        class Test:
            pass

        self.bijective("empty", Test, {})

    def test_single(self):
        @config.root
        class Test:
            a = config.attr()

        self.bijective("single", Test, {"a": "hehe"})

    def test_pristine(self):
        @config.root
        class Test:
            a = config.attr(default=5)

        self.bijective("pristine", Test, {})

    def test_dirty(self):
        @config.root
        class Test:
            a = config.attr(default=5)

        cfg = Test({})
        cfg.a = 5
        new_tree = cfg.__tree__()
        self.assertEqual(json.dumps({"a": 5}, indent=2), json.dumps(new_tree, indent=2))

    def test_multi(self):
        @config.root
        class Test:
            a = config.attr(default=5)
            b = config.attr()
            c = config.attr()

        self.bijective("multi", Test, {"b": "3", "c": "hello"})

    def test_insertion_order(self):
        @config.root
        class Test:
            a = config.attr(default=5)
            b = config.attr()
            c = config.attr()

        self.bijective("multi", Test, {"a": 3, "b": "hi", "c": "hello"})
        self.bijective("multi", Test, {"c": "hello", "b": "hi", "a": 3})

    def test_autocorrect(self):
        @config.root
        class Test:
            a = config.attr(default=5)
            b = config.attr(type=float)
            c = config.attr()

        cfg = Test({"a": "5", "b": "5.", "c": 3})
        test_tree = cfg.__tree__()
        ref_tree = {"a": 5, "b": 5.0, "c": "3"}
        self.surjective("autocorrect", Test, ref_tree, test_tree)

    @unittest.expectedFailure
    def test_full(self):
        self.bijective("full", Configuration, as_json(full_config))

    def test_eval(self):
        @config.root
        class Test:
            a = config.attr(type=types.evaluation())

        cfg, tree = self.bijective("eval", Test, {"a": {"statement": "[1, 2, 3]"}})
        self.assertEqual([1, 2, 3], cfg.a)
        self.assertEqual({"statement": "[1, 2, 3]"}, tree["a"])


class TestDictScripting(unittest.TestCase):
    def test_add(self):
        netw = Scaffold()
        cfg = netw.configuration
        ct = cfg.cell_types.add("test", spatial=dict(radius=2))
        # Check that the dict operation completed succesfully
        self.assertEqual(1, len(cfg.cell_types), "add failed")
        self.assertEqual(["test"], list(cfg.cell_types.keys()), "wrong key")
        self.assertEqual("test", ct.name, "wrong name")
        self.assertEqual("{root}.cell_types.test", ct.get_node_name(), "wrong node name")
        # Check that the `scaffold` attribute gets set.
        self.assertIs(ct.scaffold, netw, "not booted")
        with self.assertRaises(KeyError):
            # Check that `add` doesn't overwrite keys
            cfg.cell_types.add("test")

    def test_ref(self):
        # Check that references get resolved when dynamically added.
        cfg = Configuration.default()
        part = cfg.partitions.add("test", thickness=10)
        reg = cfg.regions.add("ello", children=["test"])
        self.assertIs(reg, part.region, "reference not resolved")

    def test_clear(self):
        netw = Scaffold()
        netw.regions.add("ello", children=[])
        netw.regions.add("ello2", children=[])
        r3 = netw.regions.add("ello3", children=[])
        self.assertEqual(RegionGroup, type(r3), "expected group")
        self.assertEqual(3, len(netw.regions), "not added")
        self.assertIs(netw, r3.scaffold, "scaffold not set")
        netw.regions.clear()
        self.assertEqual(0, len(netw.regions), "not cleared")
        # Check that the objects aren't associated with the config tree anymore
        self.assertFalse(hasattr(r3, "scaffold"), "scaffold not cleared")
        self.assertIs(r3, _attrs._get_root(r3), "chain not cleared")

    def test_pop(self):
        netw = Scaffold()
        netw.regions.add("ello", children=[])
        netw.regions.add("ello2", children=[])
        r3 = netw.regions.add("ello3", children=[])
        self.assertEqual(RegionGroup, type(r3), "expected group")
        self.assertEqual(3, len(netw.regions), "not added")
        self.assertIs(netw, r3.scaffold, "scaffold not set")
        popped = netw.regions.pop("ello3")
        self.assertIs(r3, popped, "weird item popped")
        self.assertEqual(2, len(netw.regions), "should be 2 items left")
        # Check that the objects aren't associated with the config tree anymore
        self.assertFalse(hasattr(r3, "scaffold"), "scaffold not cleared")
        self.assertIs(r3, _attrs._get_root(r3), "chain not cleared")

    def test_popitem(self):
        netw = Scaffold()
        netw.regions.add("ello", children=[])
        netw.regions.add("ello2", children=[])
        r3 = netw.regions.add("ello3", children=[])
        self.assertEqual(RegionGroup, type(r3), "expected group")
        self.assertEqual(3, len(netw.regions), "not added")
        self.assertIs(netw, r3.scaffold, "scaffold not set")
        key, popped = netw.regions.popitem()
        self.assertIs(r3, popped, "weird item popped")
        self.assertEqual(2, len(netw.regions), "should be 2 items left")
        # Check that the objects aren't associated with the config tree anymore
        self.assertFalse(hasattr(r3, "scaffold"), "scaffold not cleared")
        self.assertIs(r3, _attrs._get_root(r3), "chain not cleared")

    def test_setdefault(self):
        netw = Scaffold()
        default = netw.regions.setdefault("ello", dict(children=[]))
        self.assertEqual(RegionGroup, type(default), "expected group")
        newer = netw.regions.setdefault("ello", dict(children=[]))
        self.assertIs(default, newer, "default not respected")

    def test_ior(self):
        n1 = Scaffold()
        n2 = Scaffold()
        n1.regions.add("test", children=[])
        n2.regions.add("test2", children=[])
        n2.regions.add("test", children=[], type="stack")
        n1.regions |= n2.regions
        self.assertEqual(["test", "test2"], list(n1.regions.keys()), "merge right failed")
        self.assertEqual("stack", n1.regions.test.type, "merge right failed")


class TestListScripting(unittest.TestCase):
    def setUp(self):
        self.netw = Scaffold()
        self.list = self.netw.cell_types.add(
            "test", spatial=dict(radius=2, morphologies=[])
        ).spatial.morphologies

    def assertList(self, len_, prev=[]):
        self.assertEqual(len_, len(self.list), f"expected {len_} elements")
        for i in range(len_):
            with self.subTest(i=i):
                item = self.list[i]
                self.assertEqual(
                    i,
                    item._config_index,
                    f"incorrect indices: {[v._config_index for v in self.list]}",
                )
                self.assertEqual("NameSelector", type(item).__name__, "cast failed")
                self.assertEqual(self.netw, item.scaffold, "scaffold assignment failed")
        for i, elem in enumerate(prev):
            with self.subTest(i=i):
                self.assertFalse(hasattr(elem, "scaffold"), "scaffold not cleared")
                self.assertEqual(None, elem._config_index, "index not removed")

    def test_indexing(self):
        self.list[:] = [{"names": []}]
        self.assertList(1)
        prev = list(self.list)
        self.list[:] = [{"names": []}] * 5
        self.assertList(5, prev)
        prev = [self.list[3]]
        self.list[3] = {"names": ["ey"]}
        self.assertList(5, prev)
        self.assertEqual(["ey"], self.list[3].names, "slice replace failed")
        prev = list(self.list[1:4])
        self.list[1:4] = [{"names": []}]
        self.assertList(3, prev)
        self.assertEqual("{removed}", prev[0].get_node_name(), "removed node name failed")

    def test_append(self):
        item = self.list.append({"names": []})
        self.assertEqual("NameSelector", type(item).__name__, "Expected cast to default.")
        self.assertEqual(1, len(self.list), "append failed")
        self.assertEqual(0, item._config_index, "weird index")
        with self.assertRaises(RequirementError):
            item = self.list.append({})
        self.assertEqual(1, len(self.list), "append should have failed")

    def test_insert(self):
        self.list[:] = [{"names": []}] * 3
        self.list.insert(1, {"names": ["ey"]})
        self.assertList(4)
        self.assertEqual(["ey"], self.list[1].names, "inserted names incorrect")
        with self.assertRaises(RequirementError):
            self.list.insert(0, {})

    def test_order(self):
        self.list[:] = [{"names": []}] * 3
        # No default sorting mechanism for nodes. Which makes sense, it's all insertion
        # order based.
        with self.assertRaises(TypeError):
            self.list.sort()
        self.list.reverse()
        self.assertList(3)

    def test_pop(self):
        self.list[:] = [{"names": []}] * 3
        self.list.pop()
        self.assertList(2)

    def test_clear(self):
        self.list[:] = [{"names": []}] * 3
        self.list.clear()
        self.assertList(0)


class TestScripting(unittest.TestCase):
    def test_booted_root(self):
        cfg = Configuration.default()
        self.assertIsNone(_attrs._booted_root(cfg), "shouldnt be booted yet")
        self.assertIsNone(_attrs._booted_root(cfg.partitions), "shouldnt be booted yet")
        Scaffold(cfg)
        self.assertIsNotNone(_attrs._booted_root(cfg), "now it should be booted")
