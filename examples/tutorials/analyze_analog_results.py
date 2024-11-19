import numpy as np
from neo import io

# Read simulation data
sim = io.NixIO("simulation-results/NAME_OF_YOUR_NEO_FILE.nio", mode="ro")
blocks = sim.read_all_blocks()
block = blocks[0].segments[0]

import matplotlib.pylab as plt  # you might have to pip install matplotlib

# Plot recorders data for every analog signal, results are stored in simulation-results folder
# Iterate over all the analog signals recorded ( for every device consider every target)
has_plotted_neuron = False  # We will only plot one neuron recording here
has_plotted_synapse = False  # We will only plot one synapse recording here
for signal in block.analogsignals:

    name_device = signal.name  # Retrieve the name of the device
    cell_id = signal.annotations["cell_id"]  # Retrieve the cell ID
    # If the signal comes from a synapse recorder, i.e., the synapse type could be retrieved
    # and if we did not plot a synapse recording yet
    if "synapse_type" in signal.annotations.keys() and not has_plotted_synapse:
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
        has_plotted_synapse = True
    # If the signal comes from a voltage recorder, i.e., the synapse type could not be retrieved
    # and if we did not plot a neuron recording yet
    elif "synapse_type" not in signal.annotations.keys() and not has_plotted_neuron:
        out_filename = (
            "simulation-results/" + name_device + "_" + str(cell_id) + ".png"
        )  # Name of the plot file
        has_plotted_neuron = True
    # If we plotted both types of recording, we exit the loop
    elif has_plotted_neuron and has_plotted_neuron:
        break
    # We still have some plotting to do
    else:
        continue

    sim_time = signal.times  # Time points of simulation recording

    # Plot and save figure to file in images folder
    plt.figure()
    plt.xlabel(f"Time ({signal.times.units.dimensionality.string})")
    plt.ylabel(f"{signal.units.dimensionality.string}")
    plt.plot(sim_time, signal.magnitude)
    plt.savefig(out_filename)
    plt.close()
