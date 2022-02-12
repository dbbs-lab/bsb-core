import unittest
import pathlib
import os
import toml
import tempfile

from bsb.exceptions import *
from bsb.option import _pyproject_content, _pyproject_bsb, _save_pyproject_bsb
from bsb import options


class TestOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.old_path = pathlib.Path.cwd()
        cls.dir = tempfile.TemporaryDirectory()
        cls.path = pathlib.Path(cls.dir.name)
        cls.path.mkdir(exist_ok=True, parents=True)
        os.chdir(cls.path)
        cls.proj = pathlib.Path("pyproject.toml").resolve()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        try:
            cls.proj.unlink()
        except:
            pass
        os.chdir(cls.old_path)
        cls.dir.cleanup()

    def tearDown(self):
        _pyproject_content.cache_clear()
        _pyproject_bsb.cache_clear()
        try:
            self.proj.unlink()
        except:
            pass

    def create_toml(self, content=None, proj=None):
        if content is None:
            content = {}
        if proj is None:
            proj = self.proj
        with open(proj, "w") as f:
            toml.dump(content, f)

    def test_root_toml_detection(self):
        path, content = _pyproject_content()
        self.assertEqual(None, path, f"detected toml file while we should not: {path}")
        self.create_toml({"_verify_": True})
        deep = self.path / "deep" / "deeper" / "deepest"
        deep.mkdir(parents=True, exist_ok=True)
        os.chdir(deep)
        try:
            _pyproject_content.cache_clear()
            path, content = _pyproject_content()
        finally:
            os.chdir(self.path)
        self.assertIsNotNone(path, "should have found a pyproject.toml")
        self.assertEqual(path, self.proj, "Found wrong toml")
        self.assertTrue(content.get("_verify_", False), "Wrong toml content?")
        self.create_toml({"_dbl_": True}, proj=deep / "pyproject.toml")
        os.chdir(deep)
        try:
            _pyproject_content.cache_clear()
            path, content = _pyproject_content()
        finally:
            os.chdir(self.path)
        self.assertIsNotNone(path, "should have found a pyproject.toml")
        self.assertEqual(path, deep / "pyproject.toml", f"detected wrong toml: {path}")
        self.assertTrue(content.get("_dbl_", False), "Wrong toml content?")
        path.unlink()

    def test_project_set(self):
        with self.assertRaises(OptionError):
            options.store("config", "hello.json")
        _pyproject_content.cache_clear()
        with open(self.proj, "w") as f:
            toml.dump({}, f)
        options.store("config", "hello.json")
        self.assertEqual("hello.json", options.read("config"), "message")
