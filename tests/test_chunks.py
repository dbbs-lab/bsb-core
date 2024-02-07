import unittest

import numpy as np
from bsb_test import (
    NumpyTestCase,
    skip_parallel,
    timeout,
    get_test_config,
    RandomStorageFixture,
    NetworkFixture,
)

from bsb.storage import Chunk


class TestChunks(unittest.TestCase, NumpyTestCase):
    def test_comp(self):
        self.assertTrue(Chunk([0, 0, 0], None) == Chunk([0, 0, 0], None), "eq chunk fail")
        self.assertFalse(Chunk([0, 0, 0], None) != Chunk([0, 0, 0], None), "ne chunkfail")
        self.assertTrue(Chunk([0, 0, 1], None) > Chunk([0, 0, 0], None), "gt chunk fail")
        self.assertTrue(Chunk([0, 1, 0], None) < Chunk([0, 0, 1], None), "lt chunk fail")
        self.assertTrue(Chunk([0, 1, 1], None) >= Chunk([0, 1, 0], None), "ge chunk fail")
        self.assertTrue(Chunk([0, 1, 1], None) >= Chunk([0, 1, 1], None), "ge chunk fail")
        self.assertTrue(Chunk([0, 1, 1], None) <= Chunk([0, 1, 1], None), "le chunk fail")
        self.assertTrue(Chunk([0, 1, 1], None) <= Chunk([0, 2, 1], None), "le chunk fail")

    def test_id(self):
        tests = (
            ([0, 0, 0], 0),
            ([1, 0, 0], 1),
            ([0, 1, 0], 65536),
            ([0, 0, 1], 4294967296),
            ([-1, 0, 0], 65535),
            ([0, -1, 0], 4294901760),
            ([0, 0, -1], 281470681743360),
            ([2, -3, -9], 281440616775682),
            ([-32768, 0, 0], 32768),
            ([-32768, 1, 0], 98304),
            ([-32768, -1, 0], 4294934528),
        )
        self.assertEqual(
            281440616775682, Chunk.from_id(281440616775682, None).id, "Problem case"
        )
        for coords, id_ in tests:
            with self.subTest(coords=coords, id=id_):
                chunk = Chunk.from_id(id_, None)
                self.assertEqual(id_, chunk.id, "Chunk.from_id not an identity op.")
                self.assertClose(coords, chunk, "Chunk coordinates didn't match test.")
        # Check a bunch of random cases, error out after the first failed subtest to
        # prevent 10k errors.
        try:
            for coords in np.random.default_rng().integers(
                np.iinfo(np.int16).min,
                np.iinfo(np.int16).max,
                size=(10000, 3),
                dtype=np.int16,
            ):
                self.assertClose(
                    coords,
                    Chunk.from_id(Chunk(coords, None).id, None),
                    "Chunks not bijective.",
                )
        except AssertionError:
            # Reraises the caught assertion error under a subtest with the input coords.
            with self.subTest(coords=coords):
                self.assertClose(
                    coords,
                    Chunk.from_id(Chunk(coords, None).id, None),
                    "Chunks not bijective.",
                )


class TestChunkedPS(
    RandomStorageFixture,
    NetworkFixture,
    unittest.TestCase,
    NumpyTestCase,
    engine_name="hdf5",
):
    def setUp(self):
        self.cfg = get_test_config("single")
        super().setUp()

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
        self.ps.exclude_chunk(Chunk((0, 0, 0), None))
        chunk_all = self.ps.load_positions()
        self.assertGreater(len(chunk_all), len(chunk0))

    @skip_parallel
    @timeout(3)
    # Single process; this does not test any parallel read/write validity, just the
    # basic chunk properties. For example uses `.place` directly.
    def test_single_chunk(self):
        # Test that specifying a single chunk only reads the data from that chunk
        self.ps = ps = self.network.get_placement_set("test_cell")
        ps.include_chunk(Chunk((0, 0, 0), None))
        pos = ps.load_positions()
        self.assertEqual(0, len(pos), "Cell pos found before cell placement. Cleared?")
        p = self.network.placement.test_placement
        cs = self.network.network.chunk_size
        p.place(Chunk(np.array([0, 0, 0]), cs), p.get_indicators())
        pos = ps.load_positions()
        self.assertGreater(len(pos), 0, "No data loaded from chunk 000 after placement")
        # Force the addition of garbage data in another chunk, to be ignored by this
        # PlacementSet as it is set to load data only from chunk (0,0,0)
        ps.append_data(Chunk((0, 0, 1), None), [0])
        pos2 = ps.load_positions()
        self.assertEqual(
            pos.tolist(), pos2.tolist(), "PlacementSet loaded extraneous chunk data"
        )
