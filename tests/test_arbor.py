from bsb.config import from_file
from bsb.core import Scaffold
from bsb.services import MPI
from bsb.unittest import RandomStorageFixture, get_config_path
import unittest


@unittest.skipIf(MPI.get_size() > 1, "Skipped during parallel testing.")
class TestArbor(RandomStorageFixture, unittest.TestCase, engine_name="hdf5"):
    def test_brunel(self):
        cfg = from_file(get_config_path("test_brunel_wbsb.json"))
        simcfg = cfg.simulations.test_arbor

        network = Scaffold(cfg, self.storage)
        network.compile()
        result = network.run_simulation("test_arbor")

        spiketrains = result.block.segments[0].spiketrains
        sr_exc, sr_inh = None, None
        for st in spiketrains:
            if st.annotations["device"] == "sr_exc":
                sr_exc = st
            elif st.annotations["device"] == "sr_inh":
                sr_inh = st

        self.assertIsNotNone(sr_exc)
        self.assertIsNotNone(sr_inh)

        rate_ex = len(sr_exc) / simcfg.duration * 1000.0 / sr_exc.annotations["pop_size"]
        rate_in = len(sr_inh) / simcfg.duration * 1000.0 / sr_inh.annotations["pop_size"]

        # These are temporary circular values, taken from the output. May be incorrect.
        self.assertAlmostEqual(rate_in, 34.2, delta=1)
        self.assertAlmostEqual(rate_ex, 34.2, delta=1)
