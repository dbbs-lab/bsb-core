import unittest
import pathlib
import os
import sys
import toml
import tempfile

from bsb.exceptions import *
from bsb.option import _pyproject_content, _pyproject_bsb, _save_pyproject_bsb
from bsb import options
from bsb.cli import handle_command
from bsb._contexts import get_cli_context


class TestCLIOption(unittest.TestCase):
    def run_command(self, *args, dryrun=True):
        context = handle_command(args, dryrun=dryrun)
        return context

    def test_cli_opt(self):
        context = self.run_command("--version")
        force = context.options["force"].get(prio="cli")
        self.assertFalse(force, "force arg was not given, still True")
        context = self.run_command("--version", "--force")
        force = context.options["force"].get(prio="cli")
        self.assertTrue(force, "force arg given, still False")

    def test_cli_env_prio(self):
        os.environ["BSB_FOOTGUN_MODE"] = "ON"
        context = self.run_command("compile")
        f1 = context.options["force"].get(prio="env")
        f2 = context.options["force"].get(prio="cli")
        f3 = context.options["force"].get()
        self.assertTrue(f1, "env flag option not True when value ON.")
        self.assertFalse(f2, "force not set, still True")
        self.assertTrue(f3, "cli not set, still obscured the env value")
        os.environ["BSB_FOOTGUN_MODE"] = "OFF"
        context = self.run_command("compile", "--force")
        f1 = context.options["force"].get(prio="env")
        f2 = context.options["force"].get(prio="cli")
        f3 = context.options["force"].get()
        self.assertFalse(f1, "env flag option not False when value OFF.")
        self.assertTrue(f2, "force not set, still True")
        self.assertTrue(f3, "env value not overridden by cli value")
        del os.environ["BSB_FOOTGUN_MODE"]

    def test_del(self):
        # del is important for CLI context resetting before starting a new command.
        context = self.run_command("compile", "--force")
        descr = type(context.options["force"]).cli
        self.assertTrue(descr.is_set(context.options["force"]), "args didnt set cli")
        del context.options["force"].cli
        self.assertFalse(descr.is_set(context.options["force"]), "del didnt clear cli")


class TestEnvOption(unittest.TestCase):
    def setUp(self):
        self.opt = options.get_options()

    def test_env_get(self):
        for opt in self.opt.values():
            self.assertFalse(type(opt).env.is_set(opt), "No BSB env vars should be set.")

    def test_env_set(self):
        v_opt = self.opt["verbosity"]
        v_opt.env = 4
        self.assertEqual("4", os.environ["BSB_VERBOSITY"], "opt setting failed")
        self.assertEqual(4, v_opt.env, "opt getting failed")
        self.assertTrue(type(v_opt).env.is_set(v_opt), "should be set before del")
        del v_opt.env
        self.assertFalse(type(v_opt).env.is_set(v_opt), "should not be set after del")
        self.assertNotIn("BSB_VERBOSITY", os.environ, "opt deleting failed")
        # Double del shouldn't error
        del v_opt.env

    def test_parse(self):
        self.assertIsNone(self.opt["force"].env, "unset env opt should be None")
        self.opt["force"].env = True
        self.assertEqual("ON", os.environ["BSB_FOOTGUN_MODE"], "env opt not rev parsed")
        self.assertTrue(self.opt["force"].env is True, "env opt not parsed")
        self.opt["force"].env = False
        self.assertEqual("OFF", os.environ["BSB_FOOTGUN_MODE"], "env opt rev parsed bad")


class TestProjectOption(unittest.TestCase):
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

    def test_project_settings(self):
        with self.assertRaises(OptionError):
            options.store("version", "hello.json")
        with self.assertRaises(OptionError):
            options.store("versionnnn", "hello.json")
        with self.assertRaises(OptionError):
            options.read("versionnnn")
        _pyproject_content.cache_clear()
        with open(self.proj, "w") as f:
            toml.dump({}, f)
        options.store("config", "hello.json")
        self.assertEqual("hello.json", options.read("config"), "not stored/read")
        self.assertEqual("hello.json", options.get("config", prio="project"), "prio bork")
        self.assertEqual({"config": "hello.json"}, options.read(), "read all failed")
        opt = options.get_options()["config"]
        self.assertTrue(type(opt).project.is_set(opt), "written and read but not is_set")
        del opt.project
        self.assertEqual(None, options.read("config"), "not deleted")


class TestScriptOption(unittest.TestCase):
    def setUp(self):
        self.opt = options.get_options()

    def test_script_get(self):
        from bsb import __version__

        ver = self.opt["version"].script
        self.assertEqual(__version__, ver, "script get mismatch")
        with self.assertRaises(ReadOnlyOptionError):
            self.opt["version"].script = "5.0"
        # Read one without bindings:
        self.assertIsNone(self.opt["version"].env, "no bindings should be None")

    def test_script_isset(self):
        script = type(self.opt["version"]).script
        self.opt["version"]
        self.assertFalse(script.is_set(self.opt["version"]), "script def counts as isset")

    def test_script_set(self):
        self.opt["force"].script = True
        self.assertTrue(self.opt["force"].script, "script opt not set")
        #  No options without script descr atm
        # with self.assertRaises(OptionError, msg="no script binding opt may not set"):
        #     self.opt["config"].script = True
        self.assertTrue(self.opt["force"].get(), "script prio broken")

    def test_script_del(self):
        v_descr = type(self.opt["verbosity"]).script
        preset = options.verbosity
        self.assertFalse(v_descr.is_set(self.opt["verbosity"]), "verbosity set?")
        options.verbosity = 4
        self.assertTrue(v_descr.is_set(self.opt["verbosity"]), "verbosity not set")
        del options.verbosity
        self.assertFalse(v_descr.is_set(self.opt["verbosity"]), "verbosity not set")

    def test_script_register(self):
        self.opt["verbosity"].unregister()
        self.opt["verbosity"].register()
        with self.assertRaises(OptionError):
            self.opt["verbosity"].register()


class TestOptions(unittest.TestCase):
    def setUp(self):
        self.opt = options.get_options()

    def test_get_option(self):
        cfg_opt = options.get_option("config")
        with self.assertRaises(OptionError):
            options.get_option("doesntexist")

    def test_discovery(self):
        cls = options.get_option_classes()
        self.assertTrue(len(cls) > 0, "no discovery")

    def test_options_get_fallback(self):
        v_descr = type(self.opt["verbosity"]).script
        self.assertFalse(
            v_descr.is_set(self.opt["verbosity"]),
            "verbosity should not be set before test start",
        )
        self.assertEqual(1, options.verbosity, "verbosity should fall back to default")

    def test_options_set(self):
        options.verbosity = 2
        self.assertEqual(2, options.verbosity, "verbosity not set")
        # Clean up the script value we set for this test.
        del self.opt["verbosity"].script
        # Double reset shouldn't error
        del self.opt["verbosity"].script

    def test_set_module_option(self):
        with self.assertRaises(OptionError):
            options.set_module_option("doesntexist", 3)
        options.set_module_option("verbosity", 3)
        del options.verbosity
        with self.assertRaises(AttributeError):
            del options.verbosityy
