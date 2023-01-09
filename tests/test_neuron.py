import h5py
import importlib
import numpy as np
import unittest

from bsb.config import from_json
from bsb.core import Scaffold, from_storage
from bsb.services import MPI
from bsb.unittest import get_config_path

config = get_config_path("legacy_mouse_cerebellum_cortex.json")
miniature_config = get_config_path("test_nrn_miniature.json")
mf_grc_config = get_config_path("test_nrn_mf_granule.json")
mf_gol_config = get_config_path("test_nrn_mf_golgi.json")
aa_goc_config = get_config_path("test_nrn_aa_goc.json")
aa_pc_config = get_config_path("test_nrn_aa_pc.json")
grc_sc_config = get_config_path("test_nrn_grc_sc.json")
sc_pc_config = get_config_path("test_nrn_sc_pc.json")


def neuron_installed():
    return importlib.util.find_spec("neuron")


@unittest.skip("Simulators fixed last in v4, NEURON extra-special last.")
class TestMiniature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if MPI.get_rank():
            MPI.barrier()
            network = from_storage("nrn_miniature.hdf5")
        else:
            config = from_json(miniature_config)
            network = Scaffold(config)
            network.place_cell_types()
            network.compile_output()
            goc = network.get_placement_set("golgi_cell").identifiers
            sc = network.get_placement_set("stellate_cell").identifiers
            pc = network.get_placement_set("purkinje_cell").identifiers
            sc_pc = network.get_connection_type("stellate_to_purkinje")
            goc_gap = network.get_connection_type("gap_goc")
            # Connect 2 out of 3 Golgi cells with bidirect halfgap junctions
            c = np.array([[goc[0], goc[1]], [goc[1], goc[0]]])
            gm = network.morphology_repository.load("GolgiCell")
            comp_id = gm.get_compartments(labels=["basal_dendrites"])[0].id
            m = np.zeros((len(c), 2))
            mmap = ["GolgiCell"]
            comp = np.array([[comp_id] * 2] * 2)
            network.connect_cells(goc_gap, c, None, m, comp, None, mmap)
            # Connect one stellate cell to one Purkinje cell
            c = np.array([[sc[0], pc[0]]])
            sm = network.morphology_repository.load("StellateCell")
            pre_comp_id = sm.get_compartments(labels=["axon"])[0].id
            pm = network.morphology_repository.load("PurkinjeCell")
            post_comp_id = pm.get_compartments(labels=["sc_targets"])[0].id
            m = np.array([[1, 0]])
            mmap = ["PurkinjeCell", "StellateCell"]
            comp = np.array([[pre_comp_id, post_comp_id]])
            network.connect_cells(sc_pc, c, None, m, comp, None, mmap)
            network.compile_output()
            MPI.barrier()
        network.run_simulation("test")
        from glob import glob

        cls.result_path = glob("results_test_*")[-1]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        # os.remove("nrn_miniature.hdf5")

    def test_issue_205(self):
        with h5py.File(self.result_path, "r") as f:
            soma_spike_recorders = f["recorders/soma_spikes"]
            self.assertEqual(7, len(soma_spike_recorders), "Missing spike recorders")
            for sr in soma_spike_recorders.values():
                self.assertEqual(
                    2, sr.shape[1], "Regression #205. Flipped spike datasets"
                )

    def test_issue_249(self):
        with h5py.File(self.result_path, "r") as f:
            fixed_spike_generators = f["recorders/input/fixed_spike_generator"]
            self.assertEqual(
                3,
                len(fixed_spike_generators),
                "Regression #249. Missing fixed spike generators.",
            )
            for sr in fixed_spike_generators.values():
                self.assertEqual(4, len(sr), "Incorrect fixed spike data")


# Manual tests, to be included later in v4 perhaps


@unittest.skip("NEURON tests need to be run manually")
class NeuronTest(unittest.TestCase):
    def test_mf_granule(self):
        config = JSONConfig(mf_grc_config)
        scaffold = Scaffold(config)
        scaffold.place_cell_types()
        scaffold.compile_output()
        mf_to_glom = scaffold.configuration.connection_types["mossy_to_glomerulus"]
        glom_to_grc = scaffold.configuration.connection_types["glomerulus_to_granule"]
        mfs = scaffold.get_placement_set("mossy_fibers").identifiers
        gloms = scaffold.get_placement_set("glomerulus").identifiers
        grcs = scaffold.get_placement_set("granule_cell").identifiers
        scaffold.connect_cells(
            mf_to_glom, np.array([[mfs[0], gloms[0]], [mfs[1], gloms[1]]])
        )
        scaffold.connect_cells(
            mf_to_glom, np.array([[mfs[0], gloms[0]], [mfs[1], gloms[1]]])
        )
        conns = np.array([[gloms[0], grcs[0]], [gloms[1], grcs[0]]])
        morpho_map = ["GranuleCell"]
        morphologies = np.array([[0, 0], [0, 0]])
        compartments = np.array([[0, 9], [0, 18]])
        scaffold.connect_cells(
            glom_to_grc,
            conns,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )
        scaffold.compile_output()
        scaffold = from_storage(scaffold.output_formatter.file)
        scaffold.run_simulation("test")

        from glob import glob
        from plotly import graph_objs as go

        results = glob("results_test_*")[-1]
        with h5py.File(results, "r") as f:
            go.Figure(
                go.Scatter(
                    x=f["recorders/soma_voltages/0"][:, 0],
                    y=f["recorders/soma_voltages/0"][:, 1],
                )
            ).show()

    def test_glom_golgi_granule(self):
        config = JSONConfig(mf_gol_config)
        scaffold = Scaffold(config)
        scaffold.place_cell_types()
        scaffold.compile_output()
        mf_to_glom = scaffold.configuration.connection_types["mossy_to_glomerulus"]
        glom_to_gc = scaffold.configuration.connection_types["glomerulus_to_golgi"]
        gc_to_grc = scaffold.configuration.connection_types["golgi_to_granule"]
        mfs = scaffold.get_placement_set("mossy_fibers").identifiers
        gloms = scaffold.get_placement_set("glomerulus").identifiers
        golgis = scaffold.get_placement_set("golgi_cell").identifiers
        granules = scaffold.get_placement_set("granule_cell").identifiers
        scaffold.connect_cells(mf_to_glom, np.array([[mfs[0], gloms[0]]]))
        conns = np.array([[gloms[0], golgis[0]]] * 20)
        m = scaffold.morphology_repository.load("GolgiCell")
        morpho_map = ["GolgiCell"]
        morphologies = np.zeros((20, 2))
        compartments = np.zeros((20, 2))
        compartments[:, 1] = np.random.choice(
            [c.id for c in m.compartments if c.type == 302], size=20
        )
        scaffold.connect_cells(
            glom_to_gc,
            conns,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )

        conns_grc = np.array([[golgis[0], granules[0]]] * 4)
        morpho_map_grc = ["GranuleCell", "GolgiCell"]
        morphologies_grc = np.zeros((4, 2))
        morphologies_grc[:, 0] = [1] * 4
        compartments_grc = np.zeros((4, 2))
        compartments_grc[:, 0] = [c.id for c in m.compartments if c.type == 2][0:4]
        compartments_grc[:, 1] = [9 * (i + 1) for i in range(4)]
        scaffold.connect_cells(
            gc_to_grc,
            conns_grc,
            morphologies=morphologies_grc,
            compartments=compartments_grc,
            morpho_map=morpho_map_grc,
        )
        scaffold.compile_output()
        scaffold = from_storage(scaffold.output_formatter.file)
        scaffold.run_simulation("test")

        from glob import glob
        from plotly import graph_objs as go

        results = glob("results_test_*")[-1]
        with h5py.File(results, "r") as f:
            g = f["recorders/soma_voltages"]
            a = f["recorders/axons"]

            def L(g, s):
                h = g[s]
                return {"x": h[:, 0], "y": h[:, 1], "name": h.attrs["label"]}

            go.Figure(
                [
                    *(go.Scatter(**L(g, i)) for i in g),
                    *(go.Scatter(**L(a, i)) for i in a),
                ]
            ).show()

    def test_aa_goc(self):
        # 5) GrC (aa) - GoC
        # To check it, 20syn on basal dendrites, not near the soma.
        # AMPA/NMDA syn with a burst of 5 spike at 100Hz. The response should be a burst
        # composed by 3 spikes
        config = JSONConfig(aa_goc_config)
        scaffold = Scaffold(config)
        scaffold.place_cell_types()
        scaffold.compile_output()
        grc_to_golgi = scaffold.configuration.connection_types["granule_to_golgi"]
        grcs = scaffold.get_placement_set("granule_cell").identifiers
        golgis = scaffold.get_placement_set("golgi_cell").identifiers
        m_gol = scaffold.morphology_repository.load("GolgiCell")
        m_grc = scaffold.morphology_repository.load("GranuleCell")
        comps = m_gol.get_compartments(["basal_dendrites"])

        conns = np.array([[grcs[0], golgis[0]]] * 20)
        morpho_map = ["GranuleCell", "GolgiCell"]
        morphologies = np.array([[0, 1]] * 20)
        compartments = np.ones((20, 2)) * m_grc.get_compartments(["ascending_axon"])[0].id
        compartments[:, 1] = np.random.choice([c.id for c in comps], size=20)
        scaffold.connect_cells(
            grc_to_golgi,
            conns,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )

        scaffold.compile_output()
        scaffold = from_storage(scaffold.output_formatter.file)
        scaffold.run_simulation("test")

        from glob import glob
        from plotly import graph_objs as go

        results = glob("results_test_*")[-1]
        with h5py.File(results, "r") as f:
            go.Figure(
                [
                    go.Scatter(
                        x=f["recorders/soma_voltages/0"][:, 0],
                        y=f["recorders/soma_voltages/0"][:, 1],
                    ),
                    go.Scatter(
                        x=f["recorders/soma_voltages/1"][:, 0],
                        y=f["recorders/soma_voltages/1"][:, 1],
                    ),
                ]
            ).show()

    def test_aa_pc(self):
        # 6) GrC (aa) - PC
        # 100 random syn, on the apical dendrites. AMPA only, 10 spikes
        # 500Hz. The response should be a burst composed by 3 spikes.
        config = JSONConfig(aa_pc_config)
        scaffold = Scaffold(config)
        scaffold.place_cell_types()
        scaffold.compile_output()
        grc_to_golgi = scaffold.configuration.connection_types["granule_to_purkinje"]
        grcs = scaffold.get_placement_set("granule_cell").identifiers
        golgis = scaffold.get_placement_set("purkinje_cell").identifiers
        m_gol = scaffold.morphology_repository.load("PurkinjeCell")
        m_grc = scaffold.morphology_repository.load("GranuleCell")
        comps = [c.id for c in m_gol.compartments if c.type == 3]

        conns = np.array([[grcs[0], golgis[0]]] * 100)
        morpho_map = ["GranuleCell", "PurkinjeCell"]
        morphologies = np.array([[0, 1]] * 100)
        compartments = (
            np.ones((100, 2)) * m_grc.get_compartments(["ascending_axon"])[0].id
        )
        compartments[:, 1] = np.random.choice(comps, size=100)
        scaffold.connect_cells(
            grc_to_golgi,
            conns,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )

        scaffold.compile_output()
        scaffold = from_storage(scaffold.output_formatter.file)
        scaffold.run_simulation("test")

        from glob import glob
        from plotly import graph_objs as go

        results = glob("results_test_*")[-1]
        with h5py.File(results, "r") as f:
            go.Figure(
                [
                    go.Scatter(
                        x=f["recorders/soma_voltages/0"][:, 0],
                        y=f["recorders/soma_voltages/0"][:, 1],
                    ),
                    go.Scatter(
                        x=f["recorders/soma_voltages/1"][:, 0],
                        y=f["recorders/soma_voltages/1"][:, 1],
                    ),
                ]
            ).show()

    def test_pf_pc(self):
        # 6) GrC (aa) - PC
        # 100 random syn, on the apical dendrites. AMPA only, 10 spikes
        # 500Hz. The response should be a burst composed by 3 spikes.
        config = JSONConfig(aa_pc_config)
        scaffold = Scaffold(config)
        scaffold.place_cell_types()
        scaffold.compile_output()
        grc_to_golgi = scaffold.configuration.connection_types["granule_to_purkinje"]
        grcs = scaffold.get_placement_set("granule_cell").identifiers
        golgis = scaffold.get_placement_set("purkinje_cell").identifiers
        m_gol = scaffold.morphology_repository.load("PurkinjeCell")
        m_grc = scaffold.morphology_repository.load("GranuleCell")
        comps = [c.id for c in m_gol.compartments if c.type == 3]

        conns = np.array([[grcs[0], golgis[0]]] * 80)
        morpho_map = ["GranuleCell", "PurkinjeCell"]
        morphologies = np.array([[0, 1]] * 80)
        compartments = np.ones((80, 2)) * m_grc.get_compartments(["parallel_fiber"])[0].id
        compartments[:, 1] = np.random.choice(comps, size=80)
        scaffold.connect_cells(
            grc_to_golgi,
            conns,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )

        scaffold.compile_output()
        scaffold = from_storage(scaffold.output_formatter.file)
        scaffold.run_simulation("test")

        from glob import glob
        from plotly import graph_objs as go

        results = glob("results_test_*")[-1]
        with h5py.File(results, "r") as f:
            go.Figure(
                [
                    go.Scatter(
                        x=f["recorders/soma_voltages/0"][:, 0],
                        y=f["recorders/soma_voltages/0"][:, 1],
                    ),
                    go.Scatter(
                        x=f["recorders/soma_voltages/1"][:, 0],
                        y=f["recorders/soma_voltages/1"][:, 1],
                    ),
                ]
            ).show()

    def test_pf_goc(self):
        # 7) GrC (pf) - GoC
        # The same as 5) except on 80 apical dendrites.
        config = JSONConfig(aa_goc_config)
        scaffold = Scaffold(config)
        scaffold.place_cell_types()
        scaffold.compile_output()
        grc_to_golgi = scaffold.configuration.connection_types["granule_to_golgi"]
        grcs = scaffold.get_placement_set("granule_cell").identifiers
        golgis = scaffold.get_placement_set("golgi_cell").identifiers
        m_gol = scaffold.morphology_repository.load("GolgiCell")
        m_grc = scaffold.morphology_repository.load("GranuleCell")
        comps = m_gol.get_compartments(["apical_dendrites"])

        conns = np.array([[grcs[0], golgis[0]]] * 80)
        morpho_map = ["GranuleCell", "GolgiCell"]
        morphologies = np.array([[0, 1]] * 80)
        compartments = np.ones((80, 2)) * m_grc.get_compartments(["ascending_axon"])[0].id
        compartments[:, 1] = np.random.choice([c.id for c in comps], size=80)
        scaffold.connect_cells(
            grc_to_golgi,
            conns,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )

        scaffold.compile_output()
        scaffold = from_storage(scaffold.output_formatter.file)
        scaffold.run_simulation("test")

        from glob import glob
        from plotly import graph_objs as go

        results = glob("results_test_*")[-1]
        with h5py.File(results, "r") as f:
            go.Figure(
                [
                    go.Scatter(
                        x=f["recorders/soma_voltages/0"][:, 0],
                        y=f["recorders/soma_voltages/0"][:, 1],
                    ),
                    go.Scatter(
                        x=f["recorders/soma_voltages/1"][:, 0],
                        y=f["recorders/soma_voltages/1"][:, 1],
                    ),
                ]
            ).show()

    def test_grc_sc(self):
        # 9) GrC - SC
        # 3 random synapses on the dendrites. AMPA/NMDA, 10 spikes at 100Hz.
        # It should do a burst of 5 spikes.
        config = JSONConfig(grc_sc_config)
        scaffold = Scaffold(config)
        scaffold.place_cell_types()
        scaffold.compile_output()
        grc_to_golgi = scaffold.configuration.connection_types["granule_to_stellate"]
        grcs = scaffold.get_placement_set("granule_cell").identifiers
        golgis = scaffold.get_placement_set("stellate_cell").identifiers
        m_grc = scaffold.morphology_repository.load("GranuleCell")
        m_gol = scaffold.morphology_repository.load("StellateCell")
        comps = m_gol.get_compartments(["dendrites"])

        conns = np.array([[grcs[0], golgis[0]]] * 3)
        morpho_map = ["GranuleCell", "StellateCell"]
        morphologies = np.array([[0, 1]] * 3)
        compartments = np.ones((3, 2)) * m_grc.get_compartments(["ascending_axon"])[0].id
        compartments[:, 1] = np.random.choice([c.id for c in comps], size=3)
        scaffold.connect_cells(
            grc_to_golgi,
            conns,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )

        scaffold.compile_output()
        scaffold = from_storage(scaffold.output_formatter.file)
        scaffold.run_simulation("test")

        from glob import glob
        from plotly import graph_objs as go

        results = glob("results_test_*")[-1]
        with h5py.File(results, "r") as f:
            go.Figure(
                [
                    go.Scatter(
                        x=f["recorders/soma_voltages/0"][:, 0],
                        y=f["recorders/soma_voltages/0"][:, 1],
                    ),
                    go.Scatter(
                        x=f["recorders/soma_voltages/1"][:, 0],
                        y=f["recorders/soma_voltages/1"][:, 1],
                    ),
                ]
            ).show()

    def test_sc_pc(self):
        # 9) GrC - SC
        # 3 random synapses on the dendrites. AMPA/NMDA, 10 spikes at 100Hz.
        # It should do a burst of 5 spikes.
        config = JSONConfig(sc_pc_config)
        scaffold = Scaffold(config)
        scaffold.place_cell_types()
        scaffold.compile_output()
        scaffold = from_storage(scaffold.output_formatter.file)
        scaffold.run_simulation("test")

        from glob import glob
        from plotly import graph_objs as go

        results = glob("results_test_*")[-1]
        with h5py.File(results, "r") as f:
            go.Figure(
                [
                    go.Scatter(
                        x=f["recorders/soma_voltages/0"][:, 0],
                        y=f["recorders/soma_voltages/0"][:, 1],
                    ),
                ]
            ).show()
