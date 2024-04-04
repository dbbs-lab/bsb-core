from bsb_plot import plot_morphology

from bsb import parse_morphology_file

morphology = parse_morphology_file("cell.swc")
plot_morphology(morphology, color={"soma": "red", "dendrites": "blue", "axon": "green"})
