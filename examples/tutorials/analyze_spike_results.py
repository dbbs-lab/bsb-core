from neo import io

# Read simulation data
my_file_name = "simulation-results/NAME_OF_YOUR_NEO_FILE.nio"
sim = io.NixIO(my_file_name, mode="ro")
block = sim.read_all_blocks()[0]
segment = block.segments[0]
my_spiketrains = segment.spiketrains

import matplotlib.pylab as plt  # you might have to pip install matplotlib

nb_spike_trains = len(my_spiketrains)
fig, ax = plt.subplots(nb_spike_trains, sharex=True, figsize=(10, nb_spike_trains * 6))
for i, spike_t in enumerate(my_spiketrains):  # Iterate over all spike trains
    name = spike_t.annotations["device"]  # Retrieve the device name
    cell_list = spike_t.annotations["senders"]  # Retrieve the ids of the cells spiking
    spike_times = spike_t.magnitude  # Retrieve the spike times
    ax[i].scatter(spike_times, cell_list, c=f"C{i}")
    ax[i].set_xlabel(f"Time ({spike_t.times.units.dimensionality.string})")
    ax[i].set_ylabel(f"Neuron ID")
    ax[i].set_title(f"Spikes from {name}")
plt.tight_layout()
plt.savefig("simulation-results/raster_plot.png", dpi=200)
