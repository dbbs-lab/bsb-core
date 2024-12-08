from bsb_nest import NestAdapter

from bsb import from_storage

network = from_storage("my_network.hdf5")
simulation = network.get_simulation("my_simulation")
adapter = NestAdapter()

for i in range(10):
    input_rate = i * 10.0
    # Clear NEST
    adapter.reset_kernel()
    # Set the stimulation rate on the fictitious `my_stimulus` device
    simulation.devices["my_stimulus"].parameters["rate"] = input_rate
    # Let the adapter translate the simulation config into
    # simulator specific instructions
    simulation_backend = adapter.prepare(simulation)
    # You have free access to the `simulation_backend` here, to tweak
    # or augment the framework's instructions.

    # ...

    # Let the adapter run the simulation and collect the output.
    results = adapter.run(simulation)[0]
    # Organize the Neo data file into your data workflow by tagging it,
    # renaming it, moving it, giving it metadata, ...
    output_file = f"my_simulation_results_{input_rate}Hz.nio"
    results.write(output_file, "ow")
