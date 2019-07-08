### Import libraries ###
import os
import h5py
import nest
import numpy as np
import scipy.io as sio
import time
import os
import psutil

from scaffold_params import *
from mpi4py import MPI


def remove_files():
    '''.gdf and .dat files are output of nest simulations;
    if any of such files is present in the current wd, it should be deleted'''
    for f in os.listdir('.'):
        if '.gdf' in f or '.dat' in f:
            os.remove(f)
            
remove_files()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

start_time = time.time()

nest.ResetKernel()
nest.set_verbosity('M_ERROR')
nest.SetKernelStatus({"local_num_threads" : 12, "resolution":0.01})

print "This is rank number {}".format(rank)


# Load file with positions and connections data
filename = 'connected_scaffold_full_dcn_158.0x158.0.hdf5'

f = h5py.File(filename, 'r+')
# Take positions
positions = np.array(f['positions'])

### Load connections:
## this method is still very 'manual'; btw, by now we can keep it and improve it later
# From granules
aa_goc = np.array(f['connections/aa_goc']) # from granule-ascending axon to golgi cells
aa_pc = np.array(f['connections/aa_pc']) # from granule-ascending axon to Purkinje cells
pf_bc = np.array(f['connections/pf_bc']) # from granule-parallel fibers to basket cells
pf_goc = np.array(f['connections/pf_goc']) # from granule-parallel fibers to golgi cells
pf_pc = np.array(f['connections/pf_pc']) # from granule-parallel fibers to Purkinje cells
pf_sc = np.array(f['connections/pf_sc']) # from granule-parallel fibers to stellate cells
# From glomeruli
glom_goc = np.array(f['connections/glom_goc']) # from glomeruli to golgi cells
glom_grc = np.array(f['connections/glom_grc']) # from glomeruli to granule cells -- remember to add few lines of code to connectome.py for it
# From Golgi cells
goc_glom = np.array(f['connections/goc_glom']) # from golgi cells to glomeruli
goc_grc = np.array(f['connections/goc_grc']) # from golgi cells to glomeruli
# From basket cells
bc_pc = np.array(f['connections/bc_pc']) # from basket to Purkinje cells
gj_bc = np.array(f['connections/gj_bc']) # gap junctions among basket cells
# From stellate cells
gj_sc = np.array(f['connections/gj_sc']) # gap junctions among stellate cells
sc_pc = np.array(f['connections/sc_pc']) # from stellate cells to Purkinje cells
# From Purkinje to deep cerebellar nuclei
pc_dcn = np.array(f['connections/pc_dcn'])
f.close()


### Creating neuron populations ###
# Create stimulus
# Create a dictionary where keys = nrntype IDs, values = cell names (strings)
id_2_cell_type = {val: key for key, val in cell_type_ID.iteritems()}
# Sort nrntype IDs 
sorted_nrn_types = sorted(list(cell_type_ID.values()))
# Create a dictionary; keys = cell names, values = lists to store neuron models
neuron_models = {key: [] for key in cell_type_ID.iterkeys()}

# At the moment, all the cells are LIF models;
# with the only exception of Glomeruli (not cells, just modeled as
# relays; i.e., parrot neurons)
for cell_id in sorted_nrn_types:
    cell_name = id_2_cell_type[cell_id] 
    if cell_name  != 'glomerulus':
        if cell_name not in nest.Models():
            nest.CopyModel('iaf_cond_exp', cell_name)
        if cell_name == 'golgi':
            nest.SetDefaults(cell_name, {
                't_ref': 2.0, #ms 
                'C_m' : 76.0, #pF
                'V_th' : -55.0, #mV
                'V_reset': -75.0, #mV
                'g_L': 3.6, #nS
                'E_L' : -65.0, #mV
                'I_e': 36.75, #pA # tonic 6-12 Hz
                'tau_syn_ex': 0.5,
                'tau_syn_in': 10.0})
        elif cell_name == 'granule':
            nest.SetDefaults(cell_name, {
                't_ref': 1.5, #ms 
                'C_m' : 3.0, #pF
                'V_th' :  -42.0, #mV 
                'V_reset': -84.0, #mV
                'g_L': 1.5, #nS
                'E_L' : -74.0, #mV
                'I_e': 0.0, #pA 
                'tau_syn_ex': 0.5,
                'tau_syn_in': 10.0})
        elif cell_name == 'purkinje':
            nest.SetDefaults(cell_name, {
                't_ref': 0.8, #ms 
                'C_m' : 620.0, #pF
                'V_th' : -47.0, #mV
                'V_reset': -72.0, #mV
                'g_L': 7.0, #nS
                'E_L' : -62.0, #mV
                'I_e': 700.0, #pA #tonic 40-70 Hz
                'tau_syn_ex': 0.5,
                'tau_syn_in': 1.6})
		# Reference for stellate params: Lennon, Hecht-Nielsen, Yamazaki, 2014
        elif cell_name == 'stellate' or cell_name == 'basket':
            nest.SetDefaults(cell_name, {
                't_ref': 1.59, #ms 
                'C_m' : 14.6, #pF
                'V_th' : -53.0, #mV
                'V_reset': -78.0, #mV
                'g_L': 1.0, #nS
                'E_L' : -68.0, #mV
                'I_e': 15.6, #pA #tonic 17 Hz
                'tau_syn_ex': 0.64,
                'tau_syn_in': 2.0})
        else: # The case of DCpN
            nest.SetDefaults(cell_name, {
                't_ref': 3.7,
                'C_m' : 89.0,  #pF
                'V_th' : -48.0,
                'V_reset': -69.0,
                'g_L': 1.56,
                'E_L':-59.0,  #mV
                'I_e': 55.75 , #pA     #tonic 10 Hz
                'tau_syn_ex': 7.1,
                'tau_syn_in': 13.6})

    else:
        if cell_name not in nest.Models():
            nest.CopyModel('parrot_neuron', cell_name)
            
    cell_pos = positions[positions[:,1]==cell_id, :]
    neuron_models[cell_name] = nest.Create(cell_name, cell_pos.shape[0])

	
def connect_neuron(conn_mat, pre, post, syn_param, conn_param='one_to_one'):
    '''This function connets 2 neuron populations:
    conn_mat = n x 2 matrix; first column contains GIDs
              of presynaptic neurons, the second contains GIDS of
              postsynaptic ones. 
    pre : models of presynaptic neurons, 
    post : models of postsynaptic neurons,
    '''
    pre_idx = [np.where(pre==x)[0][0] for x in conn_mat[:,0]+1]
    post_idx = [np.where(post==x)[0][0] for x in conn_mat[:,1]+1]
    nest.Connect(map(pre.__getitem__, pre_idx), map(post.__getitem__, post_idx), conn_spec = conn_param, syn_spec = syn_param)
    check = nest.GetConnections(pre, post)

    return check


# 1 - From granule-aa to Golgi cells
syn_param = {"model" : "static_synapse", "weight" : 20.0, "delay": 2.0}
check_aa_goc = connect_neuron(aa_goc, neuron_models['granule'], neuron_models['golgi'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('granule-aa', 'golgi')

# 2 - From granule-aa to Purkinje cells
syn_param = {"model" : "static_synapse", "weight" : 75.0, "delay": 0.9}
check_aa_pc = connect_neuron(aa_pc, neuron_models['granule'], neuron_models['purkinje'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('granule-aa', 'Purkinje')


# 3 - From granule-parallel fibers to basket cells
syn_param = {"model" : "static_synapse", "weight" : 0.2, "delay": 5.0} # w = 5.0
check_pf_bc = connect_neuron(pf_bc, neuron_models['granule'], neuron_models['basket'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('granule-pfs', 'basket')

# 4 - From granule-parallel fibers to Golgi cells
syn_param = {"model" : "static_synapse", "weight" : 0.2, "delay": 5.0} # w = 5.0
check_pf_goc = connect_neuron(pf_goc, neuron_models['granule'], neuron_models['golgi'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('granule-pfs', 'golgi')

# 5 - From granule-parallel fibers to Purkinje cells
syn_param = {"model" : "static_synapse", "weight" : 0.2, "delay": 5.0} # w = 5.0 
check_pf_pc = connect_neuron(pf_pc, neuron_models['granule'], neuron_models['purkinje'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('granule-pfs', 'Purkinje')


# 6 - From granule-parallel fibers to stellate cells
syn_param = {"model" : "static_synapse", "weight" : 0.2, "delay": 5.0} # w = 5.0
check_pf_sc = connect_neuron(pf_sc, neuron_models['granule'], neuron_models['stellate'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('granule-pfs', 'stellate')


# 7 - From glomeruli to Golgi cells
syn_param = {"model" : "static_synapse", "weight" : 2.0, "delay": 4.0}
check_glom_goc = connect_neuron(glom_goc, neuron_models['glomerulus'], neuron_models['golgi'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('glomerulus', 'golgi')


# 8 - From glomeruli to granule cells
syn_param = {"model" : "static_synapse", "weight" : 9.0, "delay": 4.0}
check_glom_grc = connect_neuron(glom_grc, neuron_models['glomerulus'], neuron_models['granule'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('glomerulus', 'granule')


# # # 9 - From golgi cells to granules
syn_param = {"model" : "static_synapse", "weight" : -5.0, "delay": 2.0}
check_goc_grc = connect_neuron(goc_grc, neuron_models['golgi'], neuron_models['granule'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('golgi', 'granule')


# 10 - From basket to purkinje
syn_param = {"model" : "static_synapse", "weight" : -10.0, "delay": 0.5}
check_bc_pc = connect_neuron(bc_pc, neuron_models['basket'], neuron_models['purkinje'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('basket', 'purkinje')


# 11 - From basket to basket (gap)
syn_param = {"model" : "static_synapse", "weight" : -9.0, "delay": 4.0}
check_bc_bc = connect_neuron(gj_bc, neuron_models['basket'], neuron_models['basket'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('basket', 'basket')


# 12 - From stellate to purkinje
syn_param = {"model" : "static_synapse", "weight" : -8.5, "delay": 5.0}
check_sc_pc = connect_neuron(sc_pc, neuron_models['stellate'], neuron_models['purkinje'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('stellate', 'purkinje')


# 14 - From stellate to stellate (gap)
syn_param = {"model" : "static_synapse", "weight" : -2.0, "delay": 1.0}
check_sc_sc = connect_neuron(gj_sc, neuron_models['stellate'], neuron_models['stellate'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('stellate', 'stellate')


# 15 - From Purkinje to deep cerebellar nuclei
syn_param = {"model" : "static_synapse", "weight" : -0.03, "delay": 4.0} # w = -2.0?
check_pc_dcn = connect_neuron(pc_dcn, neuron_models['purkinje'], neuron_models['dcn'], syn_param=syn_param)

if rank == 0: print "Connections from {} to {} done".format('purkinje', 'dcn')


### Define stimulus features ###
RECORD_VM = False
TOT_DURATION = 300. # mseconds
STIM_START = 100. # beginning of stimulation
STIM_END = 200.   # end of stimulation
STIM_FREQ = 1000. # Frequency in Hz
RADIUS = 30. # Micron

# Stimulate a bunch of glomeruli within a volume
# Select glomeruli position
gloms_pos = positions[positions[:,1]==cell_type_ID['glomerulus'], :]
# find center of 'glomerular sphere'
x_c, y_c, z_c = np.median(gloms_pos[:,2]), np.median(gloms_pos[:,3]), np.median(gloms_pos[:,4])
#x_c, y_c, z_c = 75., 75., 75.    

# Find glomeruli falling into the selected volume
target_gloms_idx = np.sum((gloms_pos[:,2::] - np.array([x_c, y_c, z_c]))**2, axis=1).__lt__(RADIUS**2)
target_gloms = gloms_pos[target_gloms_idx,0]+1

#########
all_gloms = 0
if all_gloms:
    target_gloms = gloms_pos[:,0]+1
#########

target_granules = np.unique(glom_grc[np.in1d(glom_grc[:,0], target_gloms-1), 1])

np.save('target_granules.npy', target_granules)

id_stim = [glom for glom in neuron_models['glomerulus'] if glom in target_gloms]
n = len(target_gloms)
spike_nums = np.int(np.round((STIM_FREQ * (STIM_END - STIM_START)) / 1000.))
stim_array = np.round(np.linspace(STIM_START, STIM_END, spike_nums))

stim_array = np.arange(100., 201.)

stimulus = nest.Create("spike_generator", n,
                        params = {'spike_times': stim_array})
nest.Connect(stimulus, list(id_stim), conn_spec = 'one_to_one',
             syn_spec = {'weight': 1.0, 'delay': 0.05}) # The weight setted here is actually irrelevant


print "Execution time (in seconds) is {}".format(time.time() - start_time)

## Record spikes from granule and Golgi cells
grc_spikes = nest.Create("spike_detector",
                         params={"withgid": True, "withtime": True, "to_file": True, "label": "granule_spikes"})
goc_spikes = nest.Create("spike_detector",
                         params={"withgid": True, "withtime": True, "to_file": True, "label": "golgi_spikes"})
glom_spikes = nest.Create("spike_detector",
                         params={"withgid": True, "withtime": True, "to_file": True, "label": "glomerulus_spikes"})
pc_spikes = nest.Create("spike_detector",
                         params={"withgid": True, "withtime": True, "to_file": True, "label": "purkinje_spikes"})
bc_spikes = nest.Create("spike_detector",
                         params={"withgid": True, "withtime": True, "to_file": True, "label": "basket_spikes"})
sc_spikes = nest.Create("spike_detector",
                         params={"withgid": True, "withtime": True, "to_file": True, "label": "stellate_spikes"})
dcn_spikes = nest.Create("spike_detector",
                         params={"withgid": True, "withtime": True, "to_file": True, "label": "dcn_spikes"})
nest.Connect(neuron_models['granule'], grc_spikes)
nest.Connect(neuron_models['golgi'], goc_spikes)
nest.Connect(neuron_models['glomerulus'], glom_spikes)
nest.Connect(neuron_models['purkinje'], pc_spikes)
nest.Connect(neuron_models['basket'], bc_spikes)
nest.Connect(neuron_models['stellate'], sc_spikes)
nest.Connect(neuron_models['dcn'], dcn_spikes)

        
if RECORD_VM:
    print "Recording membrane voltage"
    grc_vm = nest.Create("multimeter")
    goc_vm = nest.Create("multimeter")
    pc_vm = nest.Create("multimeter")
    bc_vm = nest.Create("multimeter")
    sc_vm = nest.Create("multimeter")                     
    dcn_vm = nest.Create("multimeter")

    nest.SetStatus(grc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": True, "label": "granule_vm"})
    nest.SetStatus(goc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": True, "label": "golgi_vm"})
    nest.SetStatus(pc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": True, "label": "purkinje_vm"})
    nest.SetStatus(bc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": True, "label": "basket_vm"})
    nest.SetStatus(sc_vm, {"withtime": True, "record_from": ["V_m"], "to_file": True, "label": "stellate_vm"})
    nest.SetStatus(dcn_vm, {"withtime": True, "record_from": ["V_m"], "to_file": True, "label": "dcn_vm"})


    nest.Connect(grc_vm, neuron_models['granule'])
    nest.Connect(goc_vm, neuron_models['golgi'])
    nest.Connect(pc_vm, neuron_models['purkinje'])
    nest.Connect(bc_vm, neuron_models['basket'])
    nest.Connect(sc_vm, neuron_models['stellate'])
    nest.Connect(dcn_vm, neuron_models['dcn'])

nest.Simulate(TOT_DURATION)

GoCspikes = nest.GetStatus(goc_spikes, keys='events')[0]
goc_evs = GoCspikes['senders']
goc_ts = GoCspikes['times']

GrCspikes = nest.GetStatus(grc_spikes, keys='events')[0]
grc_evs = GrCspikes['senders']
grc_ts = GrCspikes['times']

Glomspikes = nest.GetStatus(glom_spikes, keys='events')[0]
glom_evs = Glomspikes['senders']
glom_ts = Glomspikes['times']

Purkinjespikes = nest.GetStatus(pc_spikes, keys='events')[0]
pc_evs = Purkinjespikes['senders']
pc_ts = Purkinjespikes['times']

Basketspikes = nest.GetStatus(bc_spikes, keys='events')[0]
bc_evs = Basketspikes['senders']
bc_ts = Basketspikes['times']

Stellatespikes = nest.GetStatus(sc_spikes, keys='events')[0]
sc_evs = Stellatespikes['senders']
sc_ts = Stellatespikes['times']

Dcnspikes = nest.GetStatus(dcn_spikes, keys='events')[0]
dcn_evs = Dcnspikes['senders']
dcn_ts = Dcnspikes['times']

np.save('grc_spike_gaba.npy', np.column_stack([grc_ts, grc_evs]))

def multim2mat(multim):
	return np.column_stack((multim['times'], multim['senders'], multim['V_m']))


if RECORD_VM:
	GoC_vm = nest.GetStatus(goc_vm)[0]['events']
	GrC_vm = nest.GetStatus(grc_vm)[0]['events']
	Purkinje_vm = nest.GetStatus(pc_vm)[0]['events']
	Basket_vm = nest.GetStatus(bc_vm)[0]['events']
	Stellate_vm = nest.GetStatus(sc_vm)[0]['events']
	DCN_vm = nest.GetStatus(dcn_vm)[0]['events']
	
	grc_vm_mat = multim2mat(GrC_vm)
	goc_vm_mat = multim2mat(GoC_vm)
	pc_vm_mat = multim2mat(Purkinje_vm)
	bc_vm_mat = multim2mat(Basket_vm)
	sc_vm_mat = multim2mat(Stellate_vm)
	dcn_vm_mat = multim2mat(DCN_vm)
	
	def_string = 'cntrl'
	np.save('grc_{}.npy'.format(def_string), grc_vm_mat)
	np.save('goc_{}.npy'.format(def_string), goc_vm_mat)
	np.save('pc_{}.npy'.format(def_string), pc_vm_mat)
	np.save('bc_{}.npy'.format(def_string), bc_vm_mat)
	np.save('sc_{}.npy'.format(def_string), sc_vm_mat)	
	np.save('dcn_{}.npy'.format(def_string), dcn_vm_mat)
	
	
	
	
	
	
