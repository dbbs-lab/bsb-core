############
Morphologies
############

Morphologies are the 3D representation of a cell. In the BSB they consist of branches,
pieces of cables, consisting of points with spatial coordinates and a radius. On top of
that each point can also be given labels, and properties.

A morphology can be summed up as::

  m = Morphology(roots)
  m.roots == <all roots>
  m.branches == <all branches, depth first starting from the roots>

The ``branches`` attribute is the result of a depth-first iteration of the roots list. Any
kind of iteration over roots or branches will always follow this same depth-first order,
and this order is implicitly used as the branch id as well.

Morphologies can be stored in :class:`MorphologyRepositories
<.storage.interfaces.MorphologyRepository>`.

Constructing morphologies
=========================

Although morphologies are usually imported from files into storage, it can be useful to
know how to create them for debugging, testing and validating. First create your branches,
then attach them together and provide the roots to the Morphology constructor:

.. code-block:: python

  from bsb.morphologies import Branch, Morphology
  import numpy as np

  # XYZ, radius,
  root = Branch(
    np.array([
      [0, 1, 2],
      [0, 1, 2],
      [0, 1, 2],
    ]),
    np.array([1, 1, 1]),
  )
  child_branch = Branch(
    np.array([
      [2, 3, 4],
      [2, 3, 4],
      [2, 3, 4],
    ]),
    np.array([1, 1, 1]),
  )
  root.attach_child(child_branch)
  m = Morphology([root])

Importing
---------

More commonly you will import morphologies from SWC or ASC files:

.. literalinclude:: ../../examples/morphologies/import.py
  :lines: 2-5
  :language: python

Once we have our :class:`~.morphologies.Morphology` object we can save it in
:class:`~.storage.Storage`; storages and networks have a ``morphologies`` attribute that
links to a :class:`~.storage.interfaces.MorphologyRepository` that can save and load
morphologies:

.. literalinclude:: ../../examples/morphologies/import.py
  :lines: 8-11
  :language: python

Basic use
---------

Morphologies and branches contain spatial data in the ``points`` and ``radii`` attributes.
Points can be individually labelled with arbitrary strings, and additional properties for
each point can be assigned to morphologies/branches:

.. literalinclude:: ../../examples/morphologies/usage.py
  :lines: 1-6
  :language: python

Once loaded we can do :ref:`transformations <transform>`, label or assign properties on the
morphology:

.. literalinclude:: ../../examples/morphologies/usage.py
  :lines: 8-17
  :language: python

Once you're done with the morphology you can save it again:

.. literalinclude:: ../../examples/morphologies/usage.py
  :lines: 19
  :language: python

.. note::

  You can assign as many labels as you like (2^64 combinations max |:innocent:|)!
  Labels'll cost you almost no memory or disk space! You can also add as many properties
  as you like, but they'll cost you memory and disk space per point on the morphology.

.. rubric:: Labels

Branches or points can be labelled, and pieces of the morphology can be selected by their
label. Labels are also useful targets to insert biophysical mechanisms into parts of the
cell later on in simulation.

.. literalinclude:: ../../examples/morphologies/labels.py


.. rubric:: Properties

.. _transform:

=======================
Subtree transformations
=======================

A subtree is a (sub)set of a morphology defined by a set of *roots* and all of its
downstream branches (i.e. the branches *emanating* from a set of roots). A subtree with
roots equal to the roots of the morphology is equal to the entire morphology, and all
transformations valid on a subtree are also valid morphology transformations.

Creating subtrees
-----------------

Subtrees can be selected using label(s) on the morphology.

.. figure:: /images/m_trans/tuft_select.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  axon = morfo.subtree("axon")
  # Multiple labels can be given
  hybrid = morfo.subtree("proximal", "distal")

.. warning::

	Branches will be selected as soon as they have one or more points labelled with a
	selected label.

Selections will always include all the branches emanating (downtree) from the selection as
well:

.. figure:: /images/m_trans/emanating.png
  :figwidth: 350px
  :align: center

.. code-block:: python

  tuft = morfo.subtree("dendritic_piece")

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

Morphology classes in the framework
===================================

The framework deals quite a bit with morphologies, here are some interesting classes for
once you dig deeper into morphologies.

Morphology preloading
---------------------
Reading the morphology data from the repository takes time. Usually morphologies are
passed around in the framework as :class:`StoredMorphologies
<.storage.interfaces.StoredMorphology>`. These objects have a
:meth:`~.storage.interfaces.StoredMorphology.load` method to load the
:class:`~.morphologies.Morphology` object from storage and a
:meth:`~.storage.interfaces.StoredMorphology.get_meta` method to return the metadata.

Morphology selectors
--------------------

The most common way of telling the framework which morphologies to use is through
:class:`MorphologySelectors <.objects.cell_type.MorphologySelector>`. A selector should
implement :meth:`~.objects.cell_type.MorphologySelector.validate` and
:meth:`~.objects.cell_type.MorphologySelector.pick` methods.

``validate`` can be used to assert that all the required morphologies are present, while
``pick`` needs to return ``True``/``False`` to include a morphology or not. Both methods
are handed :class:`~.storage.interfaces.StoredMorphology` objects, only ``load``
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

Morphology metadata
-------------------

Currently unspecified, up to the Storage and MorphologyRepository support to return a
dictionary of available metadata from
:meth:`~.storage.interfaces.MorphologyRepository.get_meta`.


Morphology distributors
-----------------------

A :class:`~.placement.strategy.MorphologyDistributor` is a special type of
:class:`~.placement.strategy.Distributor` that is usually called after positions have been
generated by a :class:`~.placement.strategy.PlacementStrategy` to assign morphologies, and
optionally rotations. The :meth:`~.placement.strategy.MorphologyDistributor.distribute`
method is called with the partitions, the indicators for the cell type and the positions;
the method has to return a :class:`~.morphologies.MorphologySet` or a tuple together with
a :class:`~.morphologies.RotationSet`.

.. warning::

	The rotations returned by a morphology distributor may be ignored when a
	:class:`~.placement.strategy.RotationDistributor` is defined for the same placement
	block.


MorphologySets
--------------

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

Reference
=========

.. automodule:: bsb.morphologies
  :members:
