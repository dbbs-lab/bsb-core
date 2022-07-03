import unittest, os, sys, numpy as np, h5py, json, shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold
from bsb.config import from_json
from bsb.storage import Chunk
from bsb.exceptions import *
from test_setup import get_config
from bsb.unittest import skip_parallel, timeout


class TestChunks(unittest.TestCase):
    @skip_parallel
    @timeout(3)
    # Single process; this does not test any parallel read/write validity, just the
    # basic chunk properties
    def test_default_chunk(self):
        # Test that when we don't specify a chunk the default is to read all chunks

        # Use the single chunk test to generate a network with some data in the default
        # (0,0,0) chunk and some data the other test is supposed to ignore in chunk 001
        self.test_single_chunk()
        # Continue using this network, checking that we pick up on the data in chunk 001
        chunk0 = self.ps.load_positions()
        # Unloaded the chunks loaded by the other test. The desired behavior is then to
        # read all chunks.
        self.ps.unload_chunk(Chunk((0, 0, 0), (100, 100, 100)))
        chunk_all = self.ps.load_positions()
        self.assertGreater(len(chunk_all), len(chunk0))

    @skip_parallel
    @timeout(3)
    # Single process; this does not test any parallel read/write validity, just the
    # basic chunk properties. For example uses `.place` directly.
    def test_single_chunk(self):
        # Test that specifying a single chunk only reads the data from that chunk
        cfg = from_json(get_config("test_single"))
        self.network = network = Scaffold(cfg, clear=True)
        self.ps = ps = network.get_placement_set("test_cell")
        ps.load_chunk(Chunk((0, 0, 0), (100, 100, 100)))
        pos = ps.load_positions()
        self.assertEqual(0, len(pos), "Cell pos found before cell placement. Cleared?")
        p = network.placement.test_placement
        cs = network.network.chunk_size
        p.place(Chunk(np.array([0, 0, 0]), cs), p.get_indicators())
        pos = ps.load_positions()
        self.assertGreater(len(pos), 0, "No data loaded from chunk 000 after placement")
        # Force the addition of garbage data in another chunk, to be ignored by this
        # PlacementSet as it is set to load data only from chunk (0,0,0)
        ps.append_data(Chunk((0, 0, 1), (100, 100, 100)), [0])
        pos2 = ps.load_positions()
        self.assertEqual(
            pos.tolist(), pos2.tolist(), "PlacementSet loaded extraneous chunk data"
        )
