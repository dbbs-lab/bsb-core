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
in ``MorphologyRepositories`` see :doc:`morphologies/repository`.

Let's start with loading a morphology and inspecting its root
:class:`~.morphologies.Branch`:

.. code-block:: python

  from bsb.core import from_hdf5
  from bsb.output import MorphologyRepository

  mr = MorphologyRepository("path/to/mr.hdf5")
  # Alternatively if you have your MR inside of a compiled network:
  network = from_hdf5("network.hdf5")
  mr = network.morphology_repository
  morfo = mr.get_morphology("my_morphology")

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
  morfo = mr.get_morphology("my_morphology")

  print("All the branches in depth-first order:", morfo.branches)
  print("All the points on those branches in depth first order:")
  print("- As vectors:", morfo.flatten())
  print("- As matrix:", morfo.flatten(matrix=True).shape)


=========
Reference
=========

.. automodule:: bsb.morphologies
  :members:
