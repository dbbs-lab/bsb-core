import pathlib
import unittest

from bsb import FileDependency


class TestUtil(unittest.TestCase):
    pass


class TestFileDeps(unittest.TestCase):
    def test_str_file_dep(self):
        file = FileDependency("hello.txt")
        self.assertEqual((pathlib.Path() / "hello.txt").absolute().as_uri(), file.uri)
        self.assertEqual("file:///", file.uri[:8])
        self.assertEqual("txt", file.extension)

    def test_str_file_dep_noext(self):
        file = FileDependency("hello")
        self.assertEqual((pathlib.Path() / "hello").absolute().as_uri(), file.uri)
        self.assertEqual(None, file.extension)

    def test_pathlib_file_dep(self):
        file = FileDependency(pathlib.Path("hello.txt"))
        self.assertEqual((pathlib.Path() / "hello.txt").absolute().as_uri(), file.uri)
        self.assertEqual("file:///", file.uri[:8])
        self.assertEqual("txt", file.extension)
