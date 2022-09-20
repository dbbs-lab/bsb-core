import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb import config
from bsb.config import from_json
from bsb.exceptions import *
from bsb.unittest import get_data_path


def get_content(f):
    with open(get_data_path("parser_tests", f), "r") as fh:
        return fh.read()


class TestJsonBasics(unittest.TestCase):
    def test_get_parser(self):
        config.get_parser("json")
        self.assertRaises(PluginError, config.get_parser, "doesntexist")

    def test_parse_empty_doc(self):
        tree, meta = config.get_parser("json").parse(get_content("doc.json"))
        self.assertEqual({}, tree, "'doc.json' parse should produce empty dict")

    def test_parse_basics(self):
        tree, meta = config.get_parser("json").parse(get_content("basics.json"))
        self.assertEqual(3, tree["list"][2], "Incorrectly parsed basic JSON")
        self.assertEqual(
            "just like that",
            tree["nest me hard"]["oh yea"],
            "Incorrectly parsed nested JSON",
        )
        self.assertEqual(
            "<parsed json config '[1, 2, 3, 'waddup']' at '/list'>", str(tree["list"])
        )


class TestJsonRef(unittest.TestCase):
    def test_indoc_reference(self):
        tree, meta = config.get_parser("json").parse(get_content("intradoc_refs.json"))
        self.assertNotIn("$ref", tree["refs"]["whats the"], "Ref key not removed")
        self.assertEqual("key", tree["refs"]["whats the"]["secret"])
        self.assertEqual("is hard", tree["refs"]["whats the"]["nested secrets"]["vim"])
        self.assertEqual("convoluted", tree["refs"]["whats the"]["nested secrets"]["and"])
        self.assertEqual(tree["refs"]["whats the"], tree["refs"]["omitted_doc"])
        with self.assertRaises(JsonReferenceError, msg="Should raise 'ref not a dict'"):
            tree, meta = config.get_parser("json").parse(
                get_content("intradoc_nodict_ref.json")
            )

    def test_far_references(self):
        tree, meta = config.get_parser("json").parse(
            get_content("interdoc_refs.json"),
            path=get_data_path("parser_tests", "interdoc_refs.json"),
        )
        self.assertIn("was", tree["refs"]["far"])
        self.assertEqual("in another folder", tree["refs"]["far"]["was"])
        self.assertIn("oh yea", tree["refs"]["whats the"])
        self.assertEqual("just like that", tree["refs"]["whats the"]["oh yea"])

    def test_double_ref(self):
        tree, meta = config.get_parser("json").parse(
            get_content("doubleref.json"),
            path=get_data_path("parser_tests", "doubleref.json"),
        )

    def test_ref_str(self):
        parser = config.get_parser("json")
        tree, meta = parser.parse(
            get_content("doubleref.json"),
            path=get_data_path("parser_tests", "doubleref.json"),
        )
        self.assertTrue(str(parser.references[0]).startswith("<json ref '"))
        # Convert windows backslashes
        wstr = str(parser.references[0]).replace("\\", "/")
        self.assertTrue(
            wstr.endswith("/bsb/unittest/data/parser_tests/interdoc_refs.json#/target'>")
        )


class TestJsonImport(unittest.TestCase):
    def test_indoc_import(self):
        tree, meta = config.get_parser("json").parse(get_content("indoc_import.json"))
        self.assertEqual(["with", "importable"], list(tree["imp"].keys()))
        self.assertEqual("are", tree["imp"]["importable"]["dicts"]["that"])

    def test_indoc_import_list(self):
        from bsb.config.parsers.json import parsed_list

        tree, meta = config.get_parser("json").parse(
            get_content("indoc_import_list.json")
        )
        self.assertEqual(["with", "importable"], list(tree["imp"].keys()))
        self.assertEqual("a", tree["imp"]["with"][0])
        self.assertEqual(parsed_list, type(tree["imp"]["with"][2]), "message")

    def test_indoc_import_value(self):
        tree, meta = config.get_parser("json").parse(
            get_content("indoc_import_other.json")
        )
        self.assertEqual(["with", "importable"], list(tree["imp"].keys()))
        self.assertEqual("a", tree["imp"]["with"])

    def test_import_merge(self):
        tree, meta = config.get_parser("json").parse(
            get_content("indoc_import_merge.json")
        )
        self.assertEqual(2, len(tree["imp"].keys()))
        self.assertIn("importable", tree["imp"])
        self.assertIn("with", tree["imp"])
        self.assertEqual(
            ["importable", "with"],
            list(tree["imp"].keys()),
            "Imported keys should follow on original keys",
        )
        self.assertEqual(4, tree["imp"]["importable"]["dicts"]["that"])
        self.assertEqual("eh", tree["imp"]["importable"]["dicts"]["even"]["nested"])
        self.assertEqual(["new", "list"], tree["imp"]["importable"]["dicts"]["with"])

    def test_import_overwrite(self):
        with self.assertWarns(ConfigurationWarning) as warning:
            tree, meta = config.get_parser("json").parse(
                get_content("indoc_import_overwrite.json")
            )
        self.assertEqual(2, len(tree["imp"].keys()))
        self.assertIn("importable", tree["imp"])
        self.assertIn("with", tree["imp"])
        self.assertEqual(
            ["importable", "with"],
            list(tree["imp"].keys()),
            "Imported keys should follow on original keys",
        )
        self.assertEqual(10, tree["imp"]["importable"])

    def test_far_import(self):
        pass
