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

The points on the branch can also be described as individual ``Compartments``::

  branch0 = [c0, c1, c2]
  c0 = Comp(start=[x0, y0, z0], end=[x1, y1, z1], radius=r1)
  c1 = Comp(start=[x1, y1, z1], end=[x2, y2, z2], radius=r2)
  c2 = Comp(start=[x2, y2, z2], end=[x3, y3, z3], radius=r3)

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
branches following the first vector-based branch description. If you want to use
``compartments``  you'll have to call ``branch.to_compartments()`` or
``morphology.to_compartments()``. For a root branch this will yield ``n - 1`` compartments
formed as line segments between pairs of points on the branch. For non-root branches an
extra compartment is prepended between the last point of the parent branch and the first
point of the child branch. Compartments are individuals so branches are no longer used to
describe the network of points, instead each compartment lists their own parent
compartment.
