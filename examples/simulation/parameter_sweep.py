from bsb.core import from_storage

# A module with cerebellar cell models
import dbbs_models

# A module to run NEURON simulations in isolation
import nrnsub

# A module to read HDF5 data
import h5py

# Read the network file
network = from_storage("my_network.hdf5")


@nrnsub.isolate
def sweep(param):
    # Get an adapter to the simulation
    adapter = network.create_adapter("my_sim_name")
    # Modify the parameter to sweep
    dbbs_models.GranuleCell.synapses["AMPA"]["U"] = param
    # Prepare simulator & instantiate all the cells and connections
    simulation = adapter.prepare()

    # (Optionally perform more custom operations before the simulation here.)

    # Run the simulation
    adapter.simulate(simulation)

    # (Optionally perform more operations or even additional simulation steps here.)

    # Collect all results in an HDF5 file and get the path to it.
    result_file = adapter.collect_output()
    return result_file


for i in range(11):
    # Sweep parameter from 0 to 1 in 0.1 increments
    result_file = sweep(i / 10)

    # Analyze each run's results here
    with h5py.File(result_file, "r") as results:
        print("What did I record?", list(results["recorders"].keys()))
