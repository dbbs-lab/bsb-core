import os
import unittest

from bsb import profiling


class TestEnvProfiling(unittest.TestCase):
    @unittest.skipIf(
        "BSB_PROFILING" not in os.environ,
        "required test env not set",
    )
    def test_session_active(self):
        session_cache = profiling.get_active_session.cache_info()
        self.assertEqual(
            1, session_cache.misses, "session inactive while BSB_PROFILING is set"
        )
