.. _components:

==================
Writing components
==================

.. todo::

  * Write this skeleton out to a full guide.
  * Start this out in a Getting Started style, where a toy problem is tackled.
  * Then, for each possible component type, write an example that covers the interface and
    common problems and important to know things.

The architecture of the framework organizes your model into reusable components. It offers
out of the box components for basic operations, but often you'll need to write your own.

.. rubric:: Importing

To use --> needs to be importable --> local code, package or plugin

.. rubric:: Structure

* Decorate with ``@config.node``
* Inherit from interface
* Parametrize with config attributes
* Implement interface functions

.. rubric:: Parametrization

Parameters defined as class attributes --> can be specified in config/init. Make things
explicitly visible and settable.

Type handling, validation, requirements

.. rubric:: Interface & implementation

Interface gives you a set of functions you must implement. If these functions are present,
framework knows how to use your class.

The framework allows you to plug in user code pretty much anywhere. Neat.

Here's how you do it (theoretically):

#. Identify which **interface** you need to extend. An interface is a programming concept
   that lets you take one of the objects of the framework and define some functions on it.
   The framework has predefined this set of functions and expects you to provide them.
   Interfaces in the framework are always classes.

#. Create a class that inherits from that interface and implement the required and/or
   interesting looking functions of its public API (which will be specified).

#. Refer to the class from the configuration by its importable module name, or use a
   :ref:`classmap`.

With a quick example, there's the ``MorphologySelector`` interface, which lets you specify
how a subset of the available morphologies should be selected for a certain group of
cells:

1. The interface is ``bsb.morphologies.MorphologySelector`` and the docs specify it has
   a ``validate(self, morphos)`` and ``pick(self, morpho)`` function.

2. Instant-Python |:tm:|, just add water:

.. code-block:: python

  from bsb.cell_types import MorphologySelector
  from bsb import config

  @config.node
  class MySizeSelector(MorphologySelector):
    min_size = config.attr(type=float, default=20)
    max_size = config.attr(type=float, default=50)

    def validate(self, morphos):
      if not all("size" in m.get_meta() for m in morphos):
        raise Exception("Missing size metadata for the size selector")

    def pick(self, morpho):
      meta = morpho.get_meta()
      return meta["size"] > self.min_size and meta["size"] < self.max_size

3. Assuming that that code is in a ``select.py`` file relative to the working directory
you can now access:

.. code-block:: json

  {
    "select": "select.MySizeSelector",
    "min_size": 30,
    "max_size": 50
  }

Share your code with the whole world and become an author of a :ref:`plugin <plugins>`!
|:heart_eyes:|

Main components
===============

Region
------

Partition
---------

PlacementStrategy
-----------------

ConnectivityStrategy
--------------------

Placement components
====================

MorphologySelector
------------------

MorphologyDistributor
---------------------

RotationDistributor
-------------------

Distributor
-----------

Indicator
---------
