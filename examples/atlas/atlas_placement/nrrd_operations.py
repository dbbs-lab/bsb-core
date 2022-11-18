import nrrd
import numpy as np

#### to extract the excitatory neuron densisties
# dens_neuron, h = nrrd.read("data/neu_density.nrrd")
# dens_inh, h = nrrd.read("data/inh_density.nrrd")
# dens_exc =  dens_neuron - dens_inh
# nrrd.write('exc_density.nrrd', dens_exc)
# print(dens_neuron[200,200,200], dens_inh[200,200,200], dens_exc[200,200,200])

#### to transform the data from number of neurons in each voxel, to density values in /um^3
voxel_size = 25 #um
dens_exc, h = nrrd.read("data/exc_density_voxel.nrrd")
dens_inh, h = nrrd.read("data/inh_density_voxel.nrrd")
print(dens_inh[200,200,200], dens_exc[200,200,200])
dens_exc =  dens_exc/voxel_size**3
dens_inh=  dens_inh/voxel_size**3
nrrd.write('exc_density.nrrd', dens_exc)
nrrd.write('inh_density.nrrd', dens_inh)
print(dens_inh[200,200,200], dens_exc[200,200,200])
