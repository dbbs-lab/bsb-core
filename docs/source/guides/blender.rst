#######
Blender
#######

The BSB features a blender module capable of creating the network inside of Blender and
animating the network activity. On top of that it completely prepares the scene including
camera and lighting and contains rendering and sequencing pipelines so that videos of the
network can be produced from start to finish with the BSB framework.

This guide assumes familiarity with Blender but can probably be succesfully reproduced by
a panicking PhD student with a deadline tomorrow aswell.

Blender mixin module
====================

To use a ``network`` in the Blender context invoke the blender mixins using the
``for_blender()`` function. This will load all the blender functions onto the network
object:

.. code-block:: python

  import bpy, bsb.core

  network = bsb.core.from_hdf5("mynetwork.hdf5")
  network.for_blender()
  # `network` now holds a reference to each BSB blender mixin/blendin function

Blending
========

Some of the functions in the blender module set the scene state state-independently. This
means that whatever state your blender scene used to be in before calling the function,
afterwards some aspect of the scene state will always be the same. The function calls ...
blend in. A concrete example would be the :func:`network.load_population
<.blender._mixin.load_population>` function: If the current scene does not contain the
population being loaded it will be created, anywhere in the script after the function call
you can safely assume the population exists in the scene. Since the function does nothing
if the population exists you can put it anywhere.

These blending functions are useful because you're likely to want to change some colors or
sizes or positions of large amounts of objects and the easiest way to do that is by
changing the declarative  value and repeating your script. This would not be possible if
the ``load_population`` function were to always recreate the population each time the
script was called.

The primary blending function is the ``network.blend(name, scene)`` function that blends
your network into the scene under the given name, blending in a root collection, cells
collection, camera and light for it. If there's nothing peculiar about any of the cell
types in your network fire up the ``load_populations`` blendin and your network will pop
up in the scene. From here on out you are either free to do with the blender objects what
you want or you can continue to use some of the BSB blendins:

.. code-block:: python

  import bpy, bsb.core, h5py, itertools

  network = bsb.core.from_hdf5("mynetwork.hdf5")
  # Blend the network into the current scene under the name `scaffold`
  network.for_blender().blend(bpy.context.scene, "scaffold")
  # Load all cell types
  network.load_populations()
  # Or, if you'd like to use the populations:
  populations = network.get_populations()
  cells = itertools.chain(*(p.cells for p in populations.values()))
  # Use the 'pulsar' animation to animate all cells with the simulation results
  with h5py.File("my_results.hdf5", "r") as f:
    network.animate.pulsar(f, cells)

.. note::

	While ``load_populations`` simply checks the existence, ``get_populations`` returns a
	BlenderPopulation object that holds references to each cell, and its Blender object.
	Some work goes into looking up the blender object for each cell so if you don't use the
	cells in every run of the script it might be better to open up with a
	``load_populations`` and call ``get_population(name)`` later when you need a specific
	population.

.. warning::

	It's easy to overload Blender with cell objects. It becomes quite difficult to use
	Blender around 20,000 cells. If you have significantly more cells be sure to save
	unpopulated versions of your Blender files, run the blendin script, save as another
	file, render it and make the required changes to the unpopulated version, repeating the
	process. Optimizations are likely to be added in the future.

Blender HPC workflow
====================

The ``devops/blender-pipe`` folder contains scripts to facilitate the rendering and
sequencing of BSB blendfiles on HPC systems. Copy them together to a directory on the HPC
system and make sure that the ``blender`` command opens Blender. The pipeline contains 2
steps, ``rendering`` each frame in parallel and ``sequencing`` the rendered images into a
video.

jrender.slurm
-------------

The render jobscript uses ``render.py`` to invoke Blender. Each Blender process will be
tasked with rendering a certain proportion of the frames. ``jrender.slurm`` takes 2
arguments, the blendfile and the output image folder:

.. code-block:: bash

	sbatch jrender.slurm my_file.blend my_file_imgs

jsequence.slurm
---------------

The sequencing jobscript stitches together the rendered frames into a video. This has to
be done in serial on a single node. It takes the blendfile and image folder as arguments:

.. code-block:: bash

	sbatch jsequence.slurm my_file.blend my_file_imgs
