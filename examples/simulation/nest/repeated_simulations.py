from bsb.core import from_hdf5

network = from_hdf5("my_network.hdf5")
simulation = network.create_adapter("my_simulation")

for i in range(10):
    simulation.reset()
    simulation.devices["my_stimulus"].parameters["rate"] = i * 10.0
    simulator = simulation.prepare()
    simulation.simulate(simulator)
    data_file = simulation.collect_output(simulator)
    with h5py.File(data_file, "r") as f:
        print("Captured", len(f["recorders"], "datasets"))
