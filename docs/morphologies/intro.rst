############
Morphologies
############

Morphologies are the 3D representation of a cell. In the BSB they consist of branches,
pieces of cable described as vectors of the properties of points. Consider the following
branch with 4 points ``p0, p1, p2, p3``::

  branch0 = [x, y, z, r]
  x = [x0, x1, x2, x3]
  y = [y0, y1, y2, y3]
  z = [z0, z1, z2, z3]
  r = [r0, r1, r2, r3]

Branches also specify which other branches they are connected to and in this way the
entire network of neuronal processes can be described. Those branches that do not have a
parent branch are called ``roots``. A morphology can have as many roots as it likes;
usually in the case of 1 root it represents the soma; in the case of many roots they each
represent the start of a process such as an axon on dendrite around an imaginary soma.

In the end a morphology can be summed up in pseudo-code as::

  m = Morphology(roots)
  m.roots = <all roots>
  m.branches = <all branches, depth first starting from the roots>

The ``branches`` attribute is the result of a depth-first iteration of the roots list. Any
kind of iteration over roots or branches will always follow this same depth-first order.

The data of these morphologies are stored in ``MorphologyRepositories`` as groups of
branches following the first vector-based branch description.

=========================
Constructing morphologies
=========================

Although morphologies are usually imported from files into storage, it can be useful to
know how to create them for debugging, testing and validating. First create your branches,
then attach them together and provide the roots to the Morphology constructor:

.. code-block:: python

  from bsb.morphologies import Branch, Morphology
  import numpy as np

  # x, y, z, radii
  branch = Branch(
    np.array([0, 1, 2]),
    np.array([0, 1, 2]),
    np.array([0, 1, 2]),
    np.array([1, 1, 1]),
  )
  child_branch = Branch(
    np.array([2, 3, 4]),
    np.array([2, 3, 4]),
    np.array([2, 3, 4]),
    np.array([1, 1, 1]),
  )
  branch.attach_child(child_branch)
  m = Morphology([branch])

.. note::

  Attaching branches is merely a graph-level connection that aids in iterating the
  morphology, no spatial connection information is inferred between the branches.
  Detaching and attaching it elsewhere won't result in any spatial changes, it will only
  affect iteration order. Keep in mind that that still affects how they are stored and
  still has drastic consequences if connections have already been made using that
  morphology (as connections use branch indices).

Using morphologies
------------------

For this introduction we're going to assume that you have a ``MorphologyRepository`` with
morphologies already present in them. To learn how to create your own morphologies stored
in ``MorphologyRepositories`` see :doc:`./repository`.

Let's start with loading a morphology and inspecting its root
:class:`~.morphologies.Branch`:

.. code-block:: python

  from bsb.core import from_hdf5
  from bsb.output import MorphologyRepository

  mr = MorphologyRepository("path/to/mr.hdf5")
  # Alternatively if you have your MR inside of a compiled network:
  network = from_hdf5("network.hdf5")
  mr = network.morphology_repository
  morfo = mr.load("my_morphology")

  # Use a local reference to the properties if you're not going to manipulate the
  # morphology, as they require a full search of the morphology to be determined every
  # time the property is accessed.
  roots = morfo.roots
  branches = morfo.branches
  print("Loaded a morphology with", len(roots), "roots, and", len(branches), "branches")
  # In most morphologies there will be a single root, representing the soma.
  soma_branch = roots[0]

  # Use the vectors of the branch (this is the most performant option)
  print("A branch can be represented by the following vectors:")
  print("x:", soma_branch.x)
  print("y:", soma_branch.y)
  print("z:", soma_branch.z)
  print("r:", soma_branch.radii)
  # Use the points property to retrieve a matrix notation of the branch
  # (Stacks the vectors into a 2d matrix)
  print("The soma can also be represented by the following matrix:", soma_branch.points)

  # There's also an iterator to walk over the points in the vectors
  print("The soma is defined as the following points:")
  for point in soma_branch.walk():
    print("*", point)

As you can see an individual branch contains all the positional data of the individual
points in the morphology. The morphology object itself then contains the collection of
branches. Normally you'd use the ``.branches`` but if you want to work with the positional
data of the whole morphology in an object you can do this by flattening the morphology:

.. code-block:: python

  from bsb.core import from_hdf5

  network = from_hdf5("network.hdf5")
  mr = network.morphology_repository
  morfo = mr.load("my_morphology")

  print("All the branches in depth-first order:", morfo.branches)
  print("All the points on those branches in depth first order:")
  print("- As vectors:", morfo.flatten())
  print("- As matrix:", morfo.flatten(matrix=True).shape)

=======================
Subtree transformations
=======================

A subtree is a (sub)set of a morphology defined by a set of *roots* and all of its
downstream branches (i.e. the branches *emanating* from a set of roots). A subtree with
roots equal to the roots of the morphology is equal to the entire morphology, and all
transformations valid on a subtree are also valid morphology transformations.

Selection
---------

Subtrees can be selected using label(s) on the morphology.

.. figure:: /images/m_trans/tuft_select.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  axon = morfo.select("axon")
  # Multiple labels can be given
  hybrid = morfo.select("proximal", "distal")

.. warning::

	Only branches that have all of their points labelled with a label will be selected.

Selection will always select all emanating branches as well:

.. figure:: /images/m_trans/emanating.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  tuft = morfo.select("dendritic_piece")

Translation
-----------

.. code-block:: python

  axon.translate([24, 100, 0])

Centering
---------

Subtrees may center themselves by offsetting the geometric mean of the origins of each
root.

Rotation
--------

Subtrees may be rotated around a singular point (by default around 0), by given 2
orientation vectors:

.. figure:: /images/m_trans/rotate_tree.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrites.rotate([0, 1, 0], [1, 0, 0])

.. figure:: /images/m_trans/rotate_dend.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrite.rotate([0, 1, 0], [1, 0, 0])


Root-rotation
-------------

Subtrees may rotate each subtree around their respective roots:

.. figure:: /images/m_trans/root_rotate_dend.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrite.root_rotate([0, 1, 0], [1, 0, 0])

.. figure:: /images/m_trans/root_rotate_tree.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrites.root_rotate([0, 1, 0], [1, 0, 0])

Gap closing
-----------

Subtree gaps between parent and child branches can be closed:

.. figure:: /images/m_trans/close_gaps.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrites.close_gaps()

.. note::

	The gaps between any subtree branch and its parent will be closed, even if the parent is
	not part of the subtree. This means that gaps of roots of a subtree may be closed as
	well.

.. note::

	Gaps between roots are not collapsed.

Collapsing
----------

Collapse the roots of a subtree onto a single point, by default the origin.

.. rubric:: Call chaining

Calls to any of the above functions can be chained together:

.. code-block:: python

  dendrites.close_gaps().center().rotate(r)

=====================
Morphology preloading
=====================

Reading the morphology data from the repository takes time. Usually morphologies are
passed around in the framework as :class:`StoredMorphologies
<.storage.interfaces.StoredMorphology>`. These objects have a
:meth:`.storage.interfaces.StoredMorphology.load` method to load the
:class:`.morphologies.Morphology` object from storage and a
:meth:`.storage.interfaces.StoredMorphology.get_meta` method to return the metadata.

====================
Morphology selectors
====================

The most common way of telling the framework which morphologies to use is through
:class:`MorphologySelectors <.objects.cell_type.MorphologySelector>`. A selector should
implement :meth:`~.objects.cell_type.MorphologySelector.validate` and
:meth:`~.objects.cell_type.MorphologySelector.pick` methods.

``validate`` can be used to assert that all the required morphologies are present, while
``pick`` needs to return ``True``/``False`` to include a morphology or not. Both methods
are handed :class:`.storage.interfaces.StoredMorphology` objects, only ``load``
morphologies if it is impossible to determine the outcome from the metadata.

.. code-block:: python

  from bsb.objects.cell_type import MorphologySelector
  from bsb import config

  @config.node
  class MySizeSelector(MorphologySelector, classmap_entry="by_size"):
    min_size = config.attr(type=float, default=20)
    max_size = config.attr(type=float, default=50)

    def validate(self, morphos):
      if not all("size" in m.get_meta() for m in morphos):
        raise Exception("Missing size metadata for the size selector")

    def pick(self, morpho):
      meta = morpho.get_meta()
      return meta["size"] > self.min_size and meta["size"] < self.max_size

===================
Morphology metadata
===================

Currently unspecified, up to the Storage and MorphologyRepository support to return a
dictionary of available metadata from
:meth:`~.storage.interfaces.MorphologyRepository.get_meta`.


=======================
Morphology distributors
=======================



==============
MorphologySets
==============

:class:`MorphologySets <.morphologies.MorphologySet>` are the result of
:class:`distributors <.placement.strategy.MorphologyDistributor>` assigning morphologies
to placed cells. They consist of a list of :class:`StoredMorphologies
<.storage.interfaces.StoredMorphology>`, a vector of indices referring to these stored
morphologies and a vector of rotations. You can use
:meth:`~.morphologies.MorphologySet.iter_morphologies` to iterate over each morphology.

.. code-block:: python

  ps = network.get_placement_set("my_detailed_neurons")
  positions = ps.load_positions()
  morphology_set = ps.load_morphologies()
  rotations = ps.load_rotations()
  cache = morphology_set.iter_morphologies(cache=True)
  for pos, morpho, rot in zip(positions, cache, rotations):
    morpho.rotate(rot)

=========
Reference
=========

.. automodule:: bsb.morphologies
  :members:
