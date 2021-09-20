from bsb.output import MorphologyRepository as MR
from bsb.plotting import plot_morphology

mr = MR("morphologies.hdf5")
pc = mr.get_morphology("PurkinjeCellNew")
plot_morphology(pc, color={"soma": "red", "dendrites": "blue", "axon": "green"})
