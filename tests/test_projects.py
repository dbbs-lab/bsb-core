import os
import shutil
import subprocess
import unittest

from bsb_test import skip_parallel

from bsb.config import get_configuration_template, parse_configuration_file


class TestProjects(unittest.TestCase):
    @skip_parallel
    def test_bsb_new(self):
        message = subprocess.check_output(
            "bsb new test_bsb_new --quickstart --json".split()
        )
        self.assertEqual(
            message.split(b"\n")[-2], b"Created 'test_bsb_new' project structure."
        )
        for folder in os.listdir():
            if folder == "test_bsb_new":
                self.assertTrue(os.path.isdir(folder))
                files = os.listdir(folder)
                self.assertEqual(len(files), 4)
                parsed_files = 0
                for filename in files:
                    if filename.endswith(".json"):
                        self.assertTrue(filename == "network_configuration.json")
                        cfg_copied = parse_configuration_file(
                            get_configuration_template("starting_example.json")[0]
                        )
                        cfg = parse_configuration_file(os.path.join(folder, filename))
                        self.assertEqual(cfg_copied.__tree__(), cfg.__tree__())
                        parsed_files += 1
                    elif filename.endswith(".py"):
                        parsed_files += 1
                    elif filename.endswith(".toml"):
                        self.assertEqual(filename, "pyproject.toml")
                        parsed_files += 1

                self.assertEqual(parsed_files, 4)
                shutil.rmtree(folder)
                break
        else:
            self.fail("The project folder was not created properly")

    @skip_parallel
    def test_bsb_new_override(self):
        folder = os.path.abspath("test_bsb_new")
        command = "bsb new test_bsb_new --quickstart --json"
        # Create the project folder
        message = subprocess.check_output(command.split())
        self.assertEqual(
            message.split(b"\n")[-2], b"Created 'test_bsb_new' project structure."
        )
        # Cannot create folder if it exists already
        message = subprocess.check_output(command.split())
        self.assertEqual(
            message.split(b"\n")[-2],
            f"Could not create '{folder}', directory exists.".encode(),
        )
        # Force recreate the project folder
        command += " --exists"
        message = subprocess.check_output(command.split())
        self.assertEqual(
            message.split(b"\n")[-2], b"Created 'test_bsb_new' project structure."
        )
        shutil.rmtree(folder)
