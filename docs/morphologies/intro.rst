############
Morphologies
############

Morphologies are the 3D representation of a cell. A morphology consists of head-to-tail
connected branches, and branches consist of a series of points with radii. Points can be
labelled and user-defined properties with one value per point can be declared on the
morphology.

.. figure:: /images/morphology.png
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/morphology_dark.png
  :figclass: only-dark
  :figwidth: 350px
  :align: center


1. The root branch, shaped like a soma because of its radii.
2. A child branch of the root branch.
3. Another child branch of the root branch.

Morphologies can be stored in :class:`MorphologyRepositories
<.storage.interfaces.MorphologyRepository>`.


Importing
=========

ASC or SWC files can be imported into a morphology repository:

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

Constructing morphologies
=========================

Create your branches, attach them in a parent-child relationship, and provide the roots to
the :class:`~.morphologies.Morphology` constructor:

.. code-block:: python

  from bsb.morphologies import Branch, Morphology
  import numpy as np

  root = Branch(
    # XYZ
    np.array([
      [0, 1, 2],
      [0, 1, 2],
      [0, 1, 2],
    ]),
    # radius
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

Basic use
=========

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

.. _morphology_labels:

.. rubric:: Labels

Branches or points can be labelled, and pieces of the morphology can be selected by their
label. Labels are also useful targets to insert biophysical mechanisms into parts of the
cell later on in simulation.

.. literalinclude:: ../../examples/morphologies/labels.py

.. rubric:: Properties

Branches and morphologies can be given additional properties. The basic properties are
``x``, ``y``, ``z``, ``radii`` and ``labels``. When you use
:meth:`~.morphologies.Morphology.from_swc`, it adds ``tags`` as an extra property.

.. _transform:

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
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/tuft_select_dark.png
  :figclass: only-dark
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
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/emanating_dark.png
  :figclass: only-dark
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

Subtrees may :meth:`~.morphologies.SubTree.center` themselves so that the point ``(0, 0,
0)`` becomes the geometric mean of the roots.

.. figure:: /images/m_trans/center.png
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/center_dark.png
  :figclass: only-dark
  :figwidth: 350px
  :align: center

Rotation
--------

Subtrees may be :meth:`rotated <.morphologies.SubTree.rotate>` around a singular point, by
giving a :class:`~scipy.spatial.transform.Rotation` (and a center, by default 0):

.. figure:: /images/m_trans/rotate_tree.png
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/rotate_tree_dark.png
  :figclass: only-dark
  :figwidth: 350px
  :align: center

.. code-block:: python

  from scipy.spatial.transform import Rotation

  r = Rotation.from_euler("xy", [90, 90], degrees=True)
  dendrites.rotate(r)

.. figure:: /images/m_trans/rotate_dend.png
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/rotate_dend_dark.png
  :figclass: only-dark
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrite.rotate(r)

Note that this creates a gap, because we are rotating around the center, root-rotation
might be preferred here.


Root-rotation
-------------

Subtrees may be :meth:`root-rotated <.morphologies.SubTree.root_rotate>` around each
respective root in the tree:

.. figure:: /images/m_trans/root_rotate_dend.png
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/root_rotate_dend_dark.png
  :figclass: only-dark
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrite.root_rotate(r)

.. figure:: /images/m_trans/root_rotate_tree.png
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/root_rotate_tree_dark.png
  :figclass: only-dark
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrites.root_rotate(r)

Additionally, you can :meth:`root-rotate <.morphologies.SubTree.root_rotate>` from a point of the
subtree instead of its root. In this case, points starting from the point selected will be rotated.

To do so, set the `downstream_of` parameter with the index of the point of your interest.

.. code-block:: python

  # rotate all points after the second point in the subtree
  # i.e.: points at index 0 and 1 will not be rotated.
  dendrites.root_rotate(r, downstream_of=2)

.. note::

  This feature can only be applied to subtrees with a single root

Gap closing
-----------

Subtree gaps between parent and child branches can be closed:

.. figure:: /images/m_trans/close_gaps.png
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/close_gaps_dark.png
  :figclass: only-dark
  :figwidth: 350px
  :align: center

.. code-block:: python

  dendrites.close_gaps()

.. note::

	The gaps between any subtree branch and its parent will be closed, even if the parent is
	not part of the subtree. This means that gaps of roots of a subtree may be closed as
	well. Gaps _between_ roots are never collapsed.

.. seealso::

	 `Collapsing`_

Collapsing
----------

Collapse the roots of a subtree onto a single point, by default the origin.

.. figure:: /images/m_trans/collapse.png
  :figclass: only-light
  :figwidth: 350px
  :align: center

.. figure:: /images/m_trans/collapse_dark.png
  :figclass: only-dark
  :figwidth: 350px
  :align: center

.. code-block:: python

  roots.collapse()

.. rubric:: Call chaining

Calls to any of the above functions can be chained together:

.. code-block:: python

  dendrites.close_gaps().center().rotate(r)

Advanced features
=================

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
:class:`MorphologySelectors <.morphologies.selector.MorphologySelector>`. Currently you
can select morphologies ``by_name`` or ``from_neuromorpho``:

.. tab-set-code::

  .. code-block:: json

    "morphologies": [
      {
        "select": "by_name",
        "names": ["my_morpho_1", "all_other_*"]
      },
      {
        "select": "from_neuromorpho",
        "names": ["H17-03-013-11-08-04_692297214_m", "cell010_GroundTruth"]
      }
    ]

If you want to make your own selector, you should implement the
:meth:`~.morphologies.selector.MorphologySelector.validate` and
:meth:`~.morphologies.selector.MorphologySelector.pick` methods.

``validate`` can be used to assert that all the required morphologies and metadata are
present, while ``pick`` needs to return ``True``/``False`` to include a morphology in the
selection. Both methods are handed :class:`~.storage.interfaces.StoredMorphology` objects.
Only :meth:`~.storage.interfaces.StoredMorphology.load` morphologies if it is impossible
to determine the outcome from the metadata alone.

The following example creates a morphology selector selects morphologies based on the
presence of a user defined metadata ``"size"``:

.. code-block:: python

  from bsb.cell_types import MorphologySelector
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

After installing your morphology selector as a plugin, you can use ``by_size`` as
selector:

.. tab-set-code::

  .. code-block:: json

    {
      "cell_type_A": {
        "spatial": {
          "morphologies": [
            {
              "select": "by_size",
              "min_size": 35
            }
          ]
        }
      }
    }

  .. code-block:: python

    network.cell_types.cell_type_A.spatial.morphologies = [MySizeSelector(min_size=35)]

Morphology metadata
-------------------

Currently unspecified, up to the Storage and MorphologyRepository support to return a
dictionary of available metadata from
:meth:`~.storage.interfaces.MorphologyRepository.get_meta`.


Morphology distributors
-----------------------

A :class:`~.placement.distributor.MorphologyDistributor` is a special type of
:class:`~.placement.distributor.Distributor` that is called after positions have been
generated by a :class:`~.placement.strategy.PlacementStrategy` to assign morphologies, and
optionally rotations. The :meth:`~.placement.distributor.MorphologyDistributor.distribute`
method is called with the partitions, the indicators for the cell type and the positions;
the method has to return a :class:`~.morphologies.MorphologySet` or a tuple together with
a :class:`~.morphologies.RotationSet`.

.. warning::

	The rotations returned by a morphology distributor may be overruled when a
	:class:`~.placement.distributor.RotationDistributor` is defined for the same placement
	block.

Distributor configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Each :guilabel:`placement` block may contain a
:class:`~.placement.distributor.DistributorsNode`, which can specify the morphology and/or
rotation distributors, and any other property distributor:

.. tab-set-code::

  .. code-block:: json

    {
      "placement": {
        "placement_A": {
          "strategy": "bsb.placement.RandomPlacement",
          "cell_types": ["cell_A"],
          "partitions": ["layer_A"],
          "distribute": {
            "morphologies": {
              "strategy": "roundrobin"
            }
          }
        }
      }
    }

  .. code-block:: python

    from bsb.placement.distributor import RoundRobinMorphologies

    network.placement.placement_A.distribute.morphologies = RoundRobinMorphologies()

Distributor interface
~~~~~~~~~~~~~~~~~~~~~

The generic interface has a single function: ``distribute(positions, context)``. The
``context`` contains ``.partitions`` and ``.indicator`` for additional placement context.
The distributor must return a dataset of N floats, where N is the number of ``positions``
you've been given, so that it can be stored as an additional property on the cell type.

The morphology distributors have a slightly different interface, and receive an additional
``morphologies`` argument: ``distribute(positions, morphologies, context)``. The
morphologies are a list of :class:`~.storage.interfaces.StoredMorphology`, that the user
has configured to use for the cell type under consideration and that the distributor
should consider the input, or template morphologies for the operation.

The morphology distributor is supposed to return an array of N integers, where each
integer refers to an index in the list of morphologies. e.g.: if there are 3 morphologies,
putting a ``0`` on the n-th index means that cell N will be assigned morphology ``0``
(which is the first morphology in the list). ``1`` and ``2`` refer to the 2nd and 3rd
morphology, and returning any other values would be an error.

If you need to break out of the morphologies that were handed to you, morphology
distributors are also allowed to return their own :class:`~.morphologies.MorphologySet`.
Since you're free to pass any list of morphology loaders to create a morphology set, you
can put and assign any morphology you like.

.. tip::

  :class:`MorphologySets <.morphologies.MorphologySet>` work on
  :class:`StoredMorphologies <.storage.interfaces.StoredMorphology>`! This means that it
  is your job to save the morphologies into your network first, and to use the returned
  values of the save operation as input to the morphology set:

  .. code-block:: python

    def distribute(self, positions, morphologies, context):
      # We're ignoring what is given, and make our own morphologies
      morphologies = [Morphology(...) for p in positions]
      # If we pass the `morphologies` to the `MorphologySet`, we create an error.
      # So we save the morphologies, and use the stored morphologies instead.
      loaders = [
        self.scaffold.morphologies.save(f"morpho_{i}", m)
        for i, m in enumerate(morphologies)
      ]
      return MorphologySet(loaders, np.arange(len(loaders)))

  This is cumbersome, so if you plan on generating new morphologies, use a `morphology
  generator`_ instead.

Finally, each morphology distributor is allowed to return an additional argument to assign
rotations to each cell as well. The return value must be a
:class:`~.morphologies.RotationSet`.

.. warning::

  The rotations returned from a morphology distributor may be ignored and replaced by the
  values of the rotation distributor, if the user configures one.

The following example creates a distributor that selects smaller morphologies the closer
the position is to the top of the partition:

.. literalinclude:: /../examples/distributors/space_aware_morphology_distributor.py
  :language: python


Then, after installing your distributor as a plugin, you can use ``small_top``:

.. tab-set-code::

  .. code-block:: json

    {
      "placement": {
        "placement_A": {
          "strategy": "bsb.placement.RandomPlacement",
          "cell_types": ["cell_A"],
          "partitions": ["layer_A"],
          "distribute": {
            "morphologies": {
              "strategy": "small_top"
            }
          }
        }
      }
    }

  .. code-block:: python

    network.placement.placement_A.distribute.morphologies = SmallerTopMorphologies()

.. _morphology generator:

Morphology generators
~~~~~~~~~~~~~~~~~~~~~

Continuing on the morphology distributor, one can also make a specialized generator of
morphologies. The generator takes the same arguments as a distributor, but returns a list
of :class:`~.morphologies.Morphology` objects, and the morphology indices to make use of
them. It can also return rotations as a 3rd return value.

This example is a morphology generator that generates a simple stick that drops down to
the origin for each position:

.. literalinclude:: /../examples/distributors/morphology_generator.py
  :language: python

Then, after installing your generator as a plugin, you can use ``touchdown``:

.. tab-set-code::

  .. code-block:: json

    {
      "placement": {
        "placement_A": {
          "strategy": "bsb.placement.RandomPlacement",
          "cell_types": ["cell_A"],
          "partitions": ["layer_A"],
          "distribute": {
            "morphologies": {
              "strategy": "touchdown"
            }
          }
        }
      }
    }

  .. code-block:: python

    network.placement.placement_A.distribute.morphologies = TouchTheBottomMorphologies()

MorphologySets
--------------

:class:`MorphologySets <.morphologies.MorphologySet>` are the result of
:class:`distributors <.placement.distributor.MorphologyDistributor>` assigning morphologies
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
