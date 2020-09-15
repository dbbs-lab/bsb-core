import unittest, os, sys, numpy as np, h5py, importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold, from_hdf5
from bsb.config import JSONConfig
from bsb.simulators.nest import NestCell
from bsb.models import Layer, CellType
from bsb.exceptions import *


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


config = relative_to_tests_folder("../bsb/configurations/mouse_cerebellum_cortex.json")
mf_grc_config = relative_to_tests_folder("configs/test_nrn_mf_granule.json")
mf_gol_config = relative_to_tests_folder("configs/test_nrn_mf_golgi.json")
aa_goc_config = relative_to_tests_folder("configs/test_nrn_aa_goc.json")
aa_pc_config = relative_to_tests_folder("configs/test_nrn_aa_pc.json")
grc_sc_config = relative_to_tests_folder("configs/test_nrn_grc_sc.json")
sc_pc_config = relative_to_tests_folder("configs/test_nrn_sc_pc.json")


def neuron_installed():
    return importlib.util.find_spec("neuron")


class MockedCell:
    _package = None


@unittest.skip("Our model's synapses are not multiplicative")
class MultiplicityTest(unittest.TestCase):
    def test_cortex_model_synapses(self):
        cfg = JSONConfig(config)
        _ = Scaffold(cfg)
        for name, model in cfg.simulations["poc"].cell_models.items():
            if model.relay:
                continue
            with self.subTest(model=name):
                self._test_model(model.model_class)

    def _test_model(self, model_class):
        for name, synapse_config in model_class.synapse_types.items():
            with self.subTest(synapse=name):
                synapse_factory = self._get_synapse_factory(synapse_config)
                finit = 0 if "GABA" in name else -65
                self._test_synapse_multiplicity(name, synapse_factory, finit=finit)

    def _test_synapse_multiplicity(self, name, synapse_factory, finit=-65):
        from patch import p

        section_single = p.Section()
        section_single.record()
        section_multi = p.Section()
        section_multi.record()
        synapse_single = synapse_factory(section_single)
        synapse_multi_1 = synapse_factory(section_multi)
        synapse_multi_2 = synapse_factory(section_multi)
        for s in [synapse_multi_1, synapse_single, synapse_single]:
            s.stimulate(delay=0, number=4, interval=25, weight=1)

        p.finitialize(finit)
        p.continuerun(150)

        # TODO: Check here that both section's recorded voltages are almost equal

    def _get_synapse_factory(self, synapse_config):
        from arborize.synapse import Synapse

        def synapse_factory(section):
            cell = self._mock_cell()
            synapse_point_process = synapse_config["point_process"]
            synapse_variant = None
            if isinstance(synapse_point_process, tuple):
                synapse_variant = synapse_point_process[1]
                synapse_point_process = synapse_point_process[0]
            return Synapse(
                cell, section, synapse_point_process, {}, variant=synapse_variant
            )

        return synapse_factory

    def _mock_cell(self):
        return MockedCell()


# Absolute dogshit code; do not use. We just quickly needed to validaate all cerebellar
# network components. Kept for future debugging.


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
        scaffold = from_hdf5(scaffold.output_formatter.file)
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
        m = scaffold.morphology_repository.get_morphology("GolgiCell")
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
        scaffold = from_hdf5(scaffold.output_formatter.file)
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
        m_gol = scaffold.morphology_repository.get_morphology("GolgiCell")
        m_grc = scaffold.morphology_repository.get_morphology("GranuleCell")
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
        scaffold = from_hdf5(scaffold.output_formatter.file)
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
        m_gol = scaffold.morphology_repository.get_morphology("PurkinjeCell")
        m_grc = scaffold.morphology_repository.get_morphology("GranuleCell")
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
        scaffold = from_hdf5(scaffold.output_formatter.file)
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
        m_gol = scaffold.morphology_repository.get_morphology("PurkinjeCell")
        m_grc = scaffold.morphology_repository.get_morphology("GranuleCell")
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
        scaffold = from_hdf5(scaffold.output_formatter.file)
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
        m_gol = scaffold.morphology_repository.get_morphology("GolgiCell")
        m_grc = scaffold.morphology_repository.get_morphology("GranuleCell")
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
        scaffold = from_hdf5(scaffold.output_formatter.file)
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
        m_grc = scaffold.morphology_repository.get_morphology("GranuleCell")
        m_gol = scaffold.morphology_repository.get_morphology("StellateCell")
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
        scaffold = from_hdf5(scaffold.output_formatter.file)
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
        scaffold = from_hdf5(scaffold.output_formatter.file)
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
