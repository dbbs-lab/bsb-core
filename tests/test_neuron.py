import h5py
import importlib
import numpy as np
import unittest
#import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from bsb.config import from_json, from_file
from bsb.core import Scaffold, from_storage
from bsb.services import MPI
from bsb.unittest import get_config_path, get_morphology_path, RandomStorageFixture

golgi_autorythm_config = get_config_path("test_nrn_goc_autorythm.json")
pc_autorythm_config = get_config_path("test_nrn_pc_autorythm.json")
golgi_current_clamp_200_pa = get_config_path("test_nrn_goc_current_clamp_200_pa.json")
grc_pc_config = get_config_path("test_nrn_grc_pc.json")

def neuron_installed():
    return importlib.util.find_spec("neuron")

#Run these tests with mpirun -n 1
#@unittest.skip("NEURON tests need to be run manually")
class NeuronTest(RandomStorageFixture,unittest.TestCase,engine_name="hdf5"):
    def test_golgi_autorythm(self):
        # Read the config and build a network with a single Golgi cell
        config = from_json(golgi_autorythm_config)
        scaffold = Scaffold(config)
        scaffold.compile()
        # Run a simulation without any stimulus to the Golgi cell.
        # The expected result is the autorythm of the Golgi cell at 8 pm 1 Hz
        result = scaffold.run_simulation("neurontest_goc_test")
        simulation_time = float(config.simulations.neurontest_goc_test.duration)
        resolution = float(config.simulations.neurontest_goc_test.resolution)
        time = np.arange(200,simulation_time+resolution,resolution)      
        avg = np.mean(result.block.segments[0].analogsignals[0], axis=1)
        after_transient = int(200./resolution)
        voltage = avg[after_transient:]
        #plt.plot(time,voltage)
        #plt.show()
        #plt.close()
        peaks, _ = find_peaks(voltage, height=0)
        self.assertAlmostEqual(len(peaks), 8, msg="The expected result is 8 pm 1 Hz", delta = 1)

    def test_golgi_current_clamp_200_pa(self):
        # Build a network with a single Golgi cell
        config = from_json(golgi_current_clamp_200_pa)
        scaffold = Scaffold(config)
        scaffold.compile()
        # Run a simulation stimulating the Golgi cell with a 200 pA current.
        # The expected result is a burst at 40 pm 2 Hz
        result = scaffold.run_simulation("neurontest_goc_test")
        simulation_time = float(config.simulations.neurontest_goc_test.duration)
        resolution = float(config.simulations.neurontest_goc_test.resolution)
        time = np.arange(200,simulation_time+resolution,resolution)      
        avg = np.mean(result.block.segments[0].analogsignals[1], axis=1)
        after_transient = int(200./resolution)
        voltage = avg[after_transient:]
        #plt.plot(time,voltage)
        #plt.show()
        #plt.close()
        peaks, _ = find_peaks(voltage, height=0)
        self.assertAlmostEqual(len(peaks), 40, msg="The expected result is 65 pm 7 Hz", delta = 2)

    def test_pc_autorythm(self):
        # Build a network with a single Purkinje cell
        config = from_json(pc_autorythm_config)
        scaffold = Scaffold(config)
        scaffold.compile()
        # Run a simulation without any stimulus to the Purkinje cell.
        # The expected result is the autorythm of the Purkinje cell (17-148 Hz)
        # (Raman IM, Bean BP. Ionic currents underlying spontaneous action potentials 
        # in isolated cerebellar Purkinje neurons. J Neurosci. 1999 )
        result = scaffold.run_simulation("neurontest_pc_test")
        simulation_time = float(config.simulations.neurontest_pc_test.duration)
        resolution = float(config.simulations.neurontest_pc_test.resolution)
        time = np.arange(200,simulation_time+resolution,resolution)      
        avg = np.mean(result.block.segments[0].analogsignals[0], axis=1)
        after_transient = int(200./resolution)
        voltage = avg[after_transient:]
        #plt.plot(time,voltage)
        #plt.show()
        #plt.close()
        peaks, _ = find_peaks(voltage, height=0)
        self.assertGreaterEqual(len(peaks), 17)
        self.assertLessEqual(len(peaks), 148)
    
    def test_granule_purkinje(self):
        # Build a network with a single Purkinje cell and ~ 70 GrCs connected to the Purkinje
        config = from_json(grc_pc_config)
        scaffold = Scaffold(config)
        scaffold.compile()
        # Stimulate GrCs with a baseline of 20 pA and a 25 pA current starting at 70 ms.
        # The expected result is a spike at ~ 80 ms
        result = scaffold.run_simulation("neurontest_grc_pc_test")
        simulation_time = float(config.simulations.neurontest_grc_pc_test.duration)
        resolution = float(config.simulations.neurontest_grc_pc_test.resolution)
        time = np.arange(68,simulation_time+resolution,resolution)      
        avg = np.mean(result.block.segments[0].analogsignals[-1], axis=1)
        after_transient = int(68./resolution)
        voltage = avg[after_transient:]
        #plt.plot(time,voltage)
        #plt.show()
        #plt.close()
        peaks, _ = find_peaks(voltage, height=0)
        self.assertEqual(len(peaks), 1, msg="The expected result is one peak")
