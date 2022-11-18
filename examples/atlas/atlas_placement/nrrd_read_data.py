#Adapted and modified from the code of Alice Geminiani

import nrrd
import numpy as np
from json_read_data import *
import plotly.graph_objects as go
import plotly.express as px
import plotly as py
from plotly.subplots import make_subplots
import random
import matplotlib.pyplot as plt
import statistics


PLOTTING = True

#data_path = "Flocculus3.0_Lingula/"

# Literature info on GR layer inhibitory neurons
# Number of Lugaro cells according to ratio from Dieudonne et al (2011)
# 1.0/15.0 * number of Purkinje cells
fac_Lugaro = 15.0

# Density of UBC according to Sekerkova et al (2014)
# 1.5*4*1000 = 6000 UBC/mm^3 for Lingula, Central Lobule, Culmen, Declive, Follum-tuber vermis, Pyramus, Simple lobule, Ansiform lobule
# 6.4*4*1000 = 25600 UBC/mm^3 for Uvula, Paramedian lobule, Copula pyramidis
# 24.6*4*1000 = 98400 UBC/ mm^3 for Nodulus
# 19.5*4*1000 = 78000 UBC/mm^3 for Flocculus
# 18272.36 UBC/mm^3 for Paraflocculus TODO: Redo computation after annotation corrections
# reference_UBC_density = {"Lingula (I)": 6000, "Uvula": 25600, "Nodulus": 98400, "Flocculus": 78000, "Paraflocculus": 18272.36}
reference_UBC_density = {"Declive (VI)": 6000}


ann, h = nrrd.read("data/annotation.nrrd")
print(h)
print(ann.shape)
print(ann[0,0,0])
print(ann[395,191,121])
dens_cell, h = nrrd.read("data/cell_density.nrrd")
print("dens cell ",dens_cell.shape)
dens_neuron, h = nrrd.read("data/neu_density.nrrd")
dens_inh, h = nrrd.read("data/inh_density.nrrd")
orientations, h = nrrd.read("data/orientations_cereb.nrrd")
print("orientation: ",orientations.shape)
print(orientations[0].shape)

jsontextfile = open("brain_regions_data.json", "r")
jsoncontent = json.loads(jsontextfile.read())
search_children(jsoncontent['msg'][0], 0)

region_name = "Declive (VI)"
#region_name = "Lingula (I)"

id_region = []
id_mol = -1
id_pc = -1
id_gr = -1

for id_reg, name in id_to_region_dictionary_ALLNAME.items():
    if region_name in name:
        if region_name + "," in name:
            id_region.append(id_reg)
            if "molecular layer" in name:
                id_mol = id_reg
            elif "granular layer" in name:
                id_gr=id_reg
            elif "Purkinje layer" in name:
                id_pc = id_reg
        else:
            id_current_region = id_reg

#mask_current_region = annis in id_region
VOXEL_SIZE = 25.0 # um
region_names = []
number_cells = []
number_neurons = []
number_inhibitory = []
volumes = []
cell_densities = []
neuron_densities = []
print(id_region)
print(id_mol)
mask = {}
vox_in_layer = {}
for id_ in id_region:
    region_names.append(id_to_region_dictionary[id_])
    region, layer = id_to_region_dictionary[id_].split(", ")
    print(layer)
    mask[layer] = ann==id_
    vox_in_layer[layer] = len(np.where(mask[layer])[0])
    number_cells.append(     np.round(np.sum(dens_cell[  mask[layer]])))
    number_neurons.append(   np.round(np.sum(dens_neuron[mask[layer]])))
    number_inhibitory.append(np.round(np.sum(dens_inh[   mask[layer]])))
    volumes.append(len(np.where(mask[layer])[0])/ (4.0**3) /1000.) # in mm3
    cell_densities.append(number_cells[-1]/volumes[-1])
    neuron_densities.append(number_neurons[-1]/volumes[-1])

mask[region_name] = ann==id_current_region

region_names.append(region_name)
number_cells.append(np.sum(number_cells))
number_neurons.append(np.sum(number_neurons))
number_inhibitory.append(np.sum(number_inhibitory))
volumes.append(np.sum(volumes))
cell_densities.append(number_cells[-1]/volumes[-1])
neuron_densities.append(number_neurons[-1]/volumes[-1])

layers_per_cell = {'granule': 'granular layer', 'golgi': 'granular layer', 'purkinje': 'Purkinje layer', 'stellate': 'Stellate layer', 'basket': 'Basket layer'}



i_granular = -1
i_molecular = -1
i_purkinje = -1
print(region_names)
for i in range(len(region_names)):
    if 'granular layer' in region_names[i]:
        i_granular = i
    if 'molecular layer' in region_names[i]:
        i_molecular = i
    if 'Purkinje layer' in region_names[i]:
        i_purkinje = i


print("i_molecular: ", i_molecular)
# Volumes print
print("Volume molecular layer: " + str(volumes[i_molecular]))
print("Volume purkinje layer: " + str(volumes[i_purkinje]))
print("Volume granular layer: " + str(volumes[i_granular]))


# Number of basket and stellate cells according to relative layer distance
bounds, h = nrrd.read("data/boundaries_mo.nrrd")
thickness_ratio = 2.0/3.0 # ratio of molecular layer space for stellate cells
up_layer_distance = np.abs(bounds[0])
down_layer_distance = np.abs(bounds[1])
print(np.amax(up_layer_distance))
print(np.amin(up_layer_distance))
print(up_layer_distance[up_layer_distance!=0])
relative_layer_distance = np.zeros(up_layer_distance.shape)

mask_mol = ann==id_mol

up_layer_distance = up_layer_distance[mask_mol]
down_layer_distance = down_layer_distance[mask_mol]
print("Mean down layer: ", statistics.mean(down_layer_distance))
print("Mean up layer: ", statistics.mean(up_layer_distance))
np.seterr(divide='ignore', invalid='ignore')
relative_layer_distance[mask_mol] = up_layer_distance / (up_layer_distance
                                                    + down_layer_distance)


mask_of_stellate = (relative_layer_distance * mask_mol)
mask_of_stellate = (mask_of_stellate > 0) * (mask_of_stellate < thickness_ratio)
mask['Stellate layer'] = mask_of_stellate
mask_of_basket = ~mask_of_stellate*mask_mol
mask['Basket layer'] = mask_of_basket
vox_in_layer['Stellate layer'] = len(np.where(mask['Stellate layer'])[0])
vox_in_layer['Basket layer'] = len(np.where(mask['Basket layer'])[0])

volumes.append(len(np.where(mask['Stellate layer'])[0])/ (4.0**3) /1000.) # in mm3
volumes.append(len(np.where(mask['Basket layer'])[0])/ (4.0**3) /1000.) # in mm3
print("Volume Stellate layer: " + str(volumes[4]))
print("Volume Basket layer: " + str(volumes[5]))
print("Total volume: "+ str(volumes[3]))

print("Voxels Granular layer: " + str(vox_in_layer['granular layer']))
print("Voxels Purkinje layer: " + str(vox_in_layer['Purkinje layer']))
print("Voxels Stellate layer: " + str(vox_in_layer['Stellate layer']))
print("Voxels Basket layer: " + str(vox_in_layer['Basket layer']))


if __name__ == '__main__':
    num_stellate = np.round(np.sum(dens_neuron[mask_of_stellate]))
    density_stellate_all = (np.round(dens_neuron[mask_of_stellate])/(VOXEL_SIZE**3))*(10**9)
    plt.hist(density_stellate_all, bins=50)
    plt.show()
    #print("Density stellate cells - max: ", np.amax(density_stellate_all),", min: ", np.amin(density_stellate_all))
    num_basket = np.round(np.sum(dens_neuron[mask_of_basket]))
    density_basket_all = (np.round(dens_neuron[mask_of_basket])/(VOXEL_SIZE**3))*(10**9)
    plt.hist(density_basket_all, bins=50)
    plt.show()
    print("Density basket cells - max: ", np.amax(density_basket_all),", min: ", np.amin(density_basket_all))
    num_mol = np.round(np.sum(dens_neuron[mask_mol]))
    vol_stellate = len(np.where(mask_of_stellate)[0])/ 4.0**3 /1000.
    vol_basket = len(np.where(mask_of_basket)[0])/ 4.0**3 /1000.
    density_stellate = num_stellate/vol_stellate
    density_basket = num_basket/vol_basket

    # Number of purkinje cells as the boundary between granular and molecular layers. Already computed
    num_purkinje = number_neurons[i_purkinje]
    print("Number of purkinje cells: " + str(num_purkinje))
    print("Estimate number of cells for " + region_name)
    print("Number of basket cells: " + str(num_basket))
    print("Number of stellate cells: " + str(num_stellate))
    print("Number of MLI: " + str(num_mol))
    print("Volume of voxels with basket cells: " + str(vol_basket))
    print("Volume of voxels with stellate cells: " + str(vol_stellate))
    print("Density of basket cells: " + str(density_basket))
    print("Density of stellate cells: " + str(density_stellate))


    # UBC
    num_UBC = np.round(reference_UBC_density[region_name] * volumes[i_granular])
    print("Number of UBC: " + str(num_UBC))
    density_UBC = num_UBC/volumes[i_granular]            # The first element in volumes is the Granule layer volume
    print("Density of UBC cells: " + str(density_UBC))

    # Lugaro cells
    num_Lugaro =  np.round(num_purkinje/ fac_Lugaro)
    print("Number of Lugaro cells: " + str(num_Lugaro))
    density_Lugaro =  num_Lugaro/volumes[i_purkinje]
    print("Density of Lugaro cells: " + str(density_Lugaro))

    # Rest of inhibitory neurons in Granular layer are Golgi cells
    num_Golgi =  np.round(number_inhibitory[i_granular] - num_Lugaro)
    print("Number of Golgi cells: " + str(num_Golgi))
    density_Golgi = num_Golgi/volumes[i_granular]            # The first element in volumes is the Granule layer volume
    print("Density of Golgi cells: " + str(density_Golgi))

    # Excitatory cells in Granular layer are Granule cells
    num_Granule =  number_neurons[i_granular] - number_inhibitory[i_granular]
    print("Number of Granule cells: " + str(num_Granule))
    density_Granule = num_Granule/volumes[i_granular]
    print("Density of Granule cells: " + str(density_Granule))


    # For each cell type, saving total number and density in a single variable
    num_Declive = {}
    num_Declive['granule'] = num_Granule
    num_Declive['golgi'] = num_Golgi
    num_Declive['purkinje'] = num_purkinje
    num_Declive['stellate'] = num_stellate
    num_Declive['basket'] = num_basket

    density_Declive = {}
    density_Declive['granule'] = density_Granule
    density_Declive['golgi'] = density_Golgi
    density_Declive['purkinje'] = num_purkinje/volumes[i_purkinje]
    density_Declive['stellate'] = density_stellate
    density_Declive['basket'] = density_basket



    ######################################### PLOTS ###############################################
    if PLOTTING:
        sliding_dir = 1


        # PC layer
        maskPC = ann==id_pc
        maskPC = maskPC*1
        print(maskPC.shape)
        PCsurface = np.where(maskPC)
        #print(PCsurface[0])
        print(id_pc)

        # ML
        maskMLI = ann==id_mol
        maskMLI = maskMLI*1
        MLIsurface = np.where(maskMLI)
        SCsurface = np.where(mask_of_stellate)
        BCsurface = np.where(mask_of_basket)

        # GL
        maskGL = ann==id_gr
        maskGL = maskGL*1
        GLsurface = np.where(maskGL)



        ############################ Scatter plot 3D #######################
        fig_scatter = go.Figure(data=[go.Scatter3d(x=PCsurface[0], y=PCsurface[1], z=PCsurface[2],
                                          mode='markers', marker=dict(size=3, color='rgb(238,250,2)'))])

        fig_scatter['data'][0]['marker']['symbol'] = 'square'

        fig_scatter.add_trace(go.Scatter3d(x=MLIsurface[0], y=MLIsurface[1], z=MLIsurface[2],
                                          mode='markers', marker=dict(size=3, color='rgb(0,255,0)')))

        fig_scatter['data'][1]['marker']['symbol'] = 'square'

        fig_scatter.add_trace(go.Scatter3d(x=GLsurface[0], y=GLsurface[1], z=GLsurface[2],
                                          mode='markers', marker=dict(size=3, color='rgb(255,0,0)')))

        fig_scatter['data'][2]['marker']['symbol'] = 'square'

        fig_scatter.show()
        fig_scatter.write_html("fig_scatter_plot_3D.html")




        ########################## Sliding sections 3D #################
        # Extract all region
        mask_all = np.isin(ann,id_region)
        print(mask_all.shape)
        region = ann - id_region[2] + 1            # Scale to have granular layer = 1, PC layer = 2, molecular layer = 3
        region[~mask_all] = 0                        # Outside of the region = 0
        region[mask_of_stellate] = 4             # To differentiate SC and BC in the ML



        # Cut around the region
        region_index = np.nonzero(region)
        print(len(region_index[0]))
        print(len(region_index[1]))
        region = region[np.amin(region_index[0])-10:np.amax(region_index[0])+10, np.amin(region_index[1])-10:np.amax(region_index[1])+10, np.amin(region_index[2])-10:np.amax(region_index[2])+10]
        dim = region.shape
        print("dim cut: ",dim)
        example_slice = np.take(region, 0, axis = sliding_dir)
        r, c = example_slice.shape


        # Define frames
        sample_factor = 1           # How much we subsample the selected dimension (sliding_dir) to plot slices
        nb_frames = int(dim[sliding_dir]/sample_factor)
        print("frames ",nb_frames)
        frames = np.linspace(0, dim[sliding_dir], nb_frames, dtype = int)

        # Define colorscale
        color_region = [
                # External to region: gray
                [0, "rgb(220, 220, 220)"],
                [0.2, "rgb(220, 220, 220)"],

                # Granular layer: red
                [0.2, "rgb(255, 0, 0)"],
                [0.4, "rgb(255, 0, 0)"],

                # PC layer: yellow
                [0.4, "rgb(238, 250, 2)"],
                [0.6, "rgb(238, 250, 2)"],

                # Molecular layer - BC: green
                [0.6, "rgb(0, 255, 0)"],
                [0.8, "rgb(0, 255, 0)"],

                # Molecular layer - SC: green olive
                [0.8, "rgb(39, 112, 39)"],
                [1.0, "rgb(39, 112, 39)"]
            ]


        # Plot Declive (VI) sections
        fig_section3 = go.Figure(frames=[go.Frame(data=go.Surface(
            z=((k) * 0.1) * np.ones((r, c)),
            surfacecolor=(np.take(region, k-1, axis = sliding_dir)),
            colorscale=color_region,
            cmin=0, cmax=3
            ),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in frames])

        print(int(frames[nb_frames-1]))

        # Add data to be displayed before animation starts
        fig_section3.add_trace(go.Surface(
            z=((frames[nb_frames-1])*0.1) * np.ones((r, c)),
            surfacecolor=np.take(region, int(frames[nb_frames-1]-1), axis = sliding_dir),
            colorscale=color_region,
            cmin=0, cmax=200,
            colorbar=dict(thickness=20, ticklen=4)
            ))
        z =((frames[nb_frames-1])*0.1) * np.ones((r, c))
        print(z.shape)
        selected = (np.take(region, int(frames[nb_frames-1]-1), axis = sliding_dir))

        print(selected.shape)

        def frame_args(duration):
            return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},
                }

        sliders = [
                    {
                        "pad": {"b": 10, "t": 60},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [
                            {
                                "args": [[f.name], frame_args(0)],
                                #"label": str(k),
                                "method": "animate",
                            }
                            for k, f in enumerate(fig_section3.frames)
                        ],
                    }
                ]

        # Axes
        plane = [r,c]
        if abs(r-c) > 2*(min(plane)):
            ax = [None]*len(plane)
            ax[plane.index(max(plane))]=[-1, max(plane)+1]
            ax[plane.index(min(plane))] = [-max(plane)/2, max(plane)/2]
        else:
            ax = [None]*len(plane)
            ax[plane.index(max(plane))]=[-1, max(plane)+1]
            ax[plane.index(min(plane))] = [-1, max(plane)+1]


        # Layout
        fig_section3.update_layout(
                 title='Declive (VI) voxelization',
                 width=1000,
                 height=1000,
                 scene=dict(
                            zaxis=dict(range=[-0.1, frames[nb_frames-1]*0.1], autorange=False),
                            yaxis=dict(range=ax[0], autorange=False),
                            xaxis=dict(range=ax[1], autorange=False),
                            aspectratio=dict(x=1, y=1, z=1),
                            ),
                 updatemenus = [
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(50)],
                                "label": "&#9654;", # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;", # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                 ],
                 sliders=sliders
        )

        fig_section3.show()
        fig_section3.write_html("fig_section3.html")


        ######################### Sections 2D as heatmaps ############################
        # Plot Region sections as heatmaps
        dim_names = ["x","y","z"]
        nb_sections = 10
        sections = []
        for sd in range(dim[sliding_dir]):
            if np.any(np.take(region, sd-1, axis = sliding_dir)):
                sections.append(sd)
        for k in range(nb_sections):
            sections_sel = random.sample(sections,nb_sections)


        sections_sel.sort()
        sections_sel[0] = 50

        titles = [dim_names[sliding_dir]+" = "+str(k) for k in sections_sel]
        fig_section2 = make_subplots(rows=2, cols=5, subplot_titles=(titles[0], titles[1], titles[2], titles[3], titles[4], titles[5], titles[6], titles[7], titles[8], titles[9]))
        ind = 0
        for k in sections_sel:
             data = go.Heatmap(
                    z=np.take(region, k-1, axis = sliding_dir),
                    colorscale=color_region
             )
             fig_section2.append_trace(data, int(ind/5) + 1, ind%5 + 1)
             ind = ind+1


        fig_section2.show()
        fig_section2.write_html("fig_section2.html")



        # fig_section2.update_layout(height=600, width=800, title_text="Subplots")
        # fig_section2.show()


##############################################################################

sliding_dir = 2

# PC layer
maskPC = ann==id_pc
maskPC = maskPC*1
PCsurface = np.where(maskPC)

# ML
maskMLI = ann==id_mol
maskMLI = maskMLI*1
MLIsurface = np.where(maskMLI)
#SCsurface = np.where(mask_of_stellate)
BCsurface = np.where(mask_of_basket)

# GL
maskGL = ann==id_gr
maskGL = maskGL*1
GLsurface = np.where(maskGL)

# Extract all region
mask_all = np.isin(ann,id_region)
region = ann - id_region[2] + 1            # Scale to have granular layer = 1, PC layer = 2, molecular layer = 3
region[~mask_all] = 0                        # Outside of the region = 0
#region[mask_of_stellate] = 4             # To differentiate SC and BC in the ML

# Cut around the region
region_index = np.nonzero(region)
region = region[np.amin(region_index[0])-10:np.amax(region_index[0])+10, np.amin(region_index[1])-10:np.amax(region_index[1])+10, np.amin(region_index[2])-10:np.amax(region_index[2])+10]
dim = region.shape
example_slice = np.take(region, 0, axis = sliding_dir)
r, c = example_slice.shape

# Define colorscale
color_region = [
                # External to region: gray
                [0, "rgb(220, 220, 220)"],
                [0.2, "rgb(220, 220, 220)"],

                # Granular layer: red
                [0.2, "rgb(255, 0, 0)"],
                [0.4, "rgb(255, 0, 0)"],

                # PC layer: yellow
                [0.4, "rgb(238, 250, 2)"],
                [0.6, "rgb(238, 250, 2)"],

                # Molecular layer - BC: green
                [0.6, "rgb(0, 255, 0)"],
                [0.8, "rgb(0, 255, 0)"],

                # Molecular layer - SC: green olive
                [0.8, "rgb(39, 112, 39)"],
                [1.0, "rgb(39, 112, 39)"]
            ]

#for sl in range(10, 20):    #ciclo for da 10 a 19
#    print(sl)


# 69 – 386 ( range 317 )
low =   np.amin(region_index[2])-10    #69
high  = np.amax(region_index[2])+10    #387
rang = high -low     #317

"""

for j in range(rang ):   #per ogni slice z
    matrix = np.take(region, j, axis = sliding_dir)
    count=0
    num_zero =0
    sum = 0
    sum_PC = 0

    for i in matrix:
        #per ogni riga
        num_zero = 0
        num_uno = 0
        num_due = 0
        num_tre = 0
        num_quattro = 0
        for k in i:   #per ogni elemento della riga
            if k == 0 :
                num_zero +=1
            if k == 1 :
                num_uno +=1
            if k == 2 :
                num_due +=1
            if k == 3 :
                num_tre +=1
            if k == 4 :
                num_quattro +=1
        sum += num_uno +num_due + num_tre + num_quattro
        #print(" Voxels outside:", num_zero, " | GL:", num_uno, " | PCL:", num_due, " BL:", num_tre, " | SL:", num_quattro, " --- TOT = ", num_uno+num_due+num_tre+num_quattro)
        sum_PC += num_due
    print("slice num:", j, "(", j+70, ") - tot voxels", sum,"|| PCL: ", sum_PC )

"""


fig_scatter6 = go.Figure(
    data =  go.Heatmap(
    z=np.take(region, 39, axis = sliding_dir),
    colorscale=color_region
    #cmin=0, cmax=200,
    #colorbar=dict(thickness=20, ticklen=4)
    )
    )
fig_scatter6.show()
fig_scatter6.write_html("fig_scatter6.html")

