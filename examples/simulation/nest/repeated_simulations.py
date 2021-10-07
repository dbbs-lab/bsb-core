from bsb.core import from_hdf5
import os

network = from_hdf5("my_network.hdf5")
simulation = network.create_adapter("my_simulation")

for i in range(10):
    input_rate = i * 10.0
    # Clear NEST
    adapter.reset()
    # Set the stimulation rate on the fictitious `my_stimulus` device
    simulation.devices["my_stimulus"].parameters["rate"] = input_rate
    # Let the adapter translate the simulation config into
    # simulator specific instructions
    simulation_backend = simulation.prepare()
    # You have free access to the `simulation_backend` here, to tweak
    # or augment the framework's instructions.

    # ...

    # Let the adapter run the simulation on the backend.
    simulation.simulate(simulation_backend)
    # Let the adapter collect the simulation output
    data_file = simulation.collect_output(simulation_backend)
    # Organize the HDF5 data file into your data workflow by tagging it,
    # renaming it, moving it, giving it metadata, ...
    with h5py.File(data_file, "r") as f:
        # Add the used parameters to the file.
        f.attrs["stimulation_rate"] = input_rate

    os.rename(data_file, f"experiments/{input_rate}Hz.hdf5")
