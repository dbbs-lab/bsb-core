# A module to read HDF5 data
import h5py
# A module to run NEURON simulations in isolation
import nrnsub

from bsb.core import from_storage

# This decorator runs each call to the function in isolation
# on a separate process. Does not work with MPI.
@nrnsub.isolate
def sweep(param):
    # Open the network file
    network = from_storage("my_network.hdf5")

    # Here you can set whatever simulation parameter you want.
    network.simulations.my_sim.devices.my_stim.rate = param

    # Run the simulation
    results = network.run_simulation("my_sim")
    # Tjese are the recorded spiketrains and signals
    print(results.spiketrains)
    print(results.analogsignals)


for i in range(11):
    # Sweep parameter from 0 to 1 in 0.1 increments
    sweep(i / 10)
