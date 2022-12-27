import os
import unittest

import bsb.options
from bsb import profiling

print("BSB_PROFILING" in os.environ, os.environ.get("BSB_PROFILING"))


class TestEnvProfiling(unittest.TestCase):
    @unittest.skipIf(
        "BSB_PROFILING" not in os.environ,
        "required test env not set",
    )
    @unittest.skipIf(not bsb.options.profiling, "profiling not enabled")
    def test_root_meter_present(self):
        session = profiling.get_active_session()
        self.assertTrue(
            [m for m in session._meters if m.name == "root_module"],
            "root meter absent but BSB_PROFILING is set",
        )

    @unittest.skipIf("BSB_PROFILING" in os.environ, "required test env set")
    def test_root_meter_absent(self):
        session = profiling.get_active_session()
        self.assertFalse(
            [m for m in session._meters if m.name == "root_module"],
            "root meter present but BSB_PROFILING is not set",
        )
