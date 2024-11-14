import numpy as np
from neo import io

# Read simulation data
sim = io.NixIO("simulation-results/NAME_OF_YOUR_NEO_FILE.nio", mode="ro")
blocks = sim.read_all_blocks()
block = blocks[0].segments[0]


import matplotlib.pylab as plt  # you might have to pip install matplotlib

# Plot recorders data for every analog signal, results are stored in simulation-results folder
# Iterate over all the analog signals recorded ( for every device consider every target)
for signal in block.analogsignals:

    name_device = signal.name  # Retrieve the name of the device
    cell_type = signal.annotations["cell_type"]  # Retrieve the type of the cell
    cell_id = signal.annotations["cell_id"]  # Retrieve the cell ID
    dev_unit = signal.annotations["unit"]  # Retrieve the unit of measure
    rate = signal.sampling_rate  # Retrieve the sampling resolution
    my_resolution = 1 / rate  # Compute time resolution of the simulation
    sim_time = (
        range(len(signal)) * my_resolution
    )  # generate the time of simulation points

    out_filename = (
        "simulation-results/" + name_device + "_" + str(cell_id) + ".png"
    )  # Name of the plot file
    if (
        "synapse_type" in signal.annotations.keys()
    ):  # In the case of synapse recorder the synapse type could be retrived
        synapse_type = signal.annotations["synapse_type"]
        out_filename = (
            "simulation-results/"
            + name_device
            + "_"
            + str(cell_id)
            + "_"
            + synapse_type
            + ".png"
        )

    # Plot and save figure to file in images folder
    plt.figure()
    plt.xlabel(f"Time (ms)")
    plt.ylabel(f"{dev_unit}")
    plt.plot(sim_time, signal)
    plt.savefig(out_filename)
    plt.close()
