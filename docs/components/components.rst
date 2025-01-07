.. _components:

==================
Writing components
==================

The architecture of the BSB framework organizes your model into reusable components. It offers
out of the box components for basic operations, but often you'll need to write your own.

If you want to read a step by step tutorial on how to make your own component, check this
:doc:`page </getting-started/guide_components>`

For each component, the BSB provides interfaces, each with a set of functions that you must
implement. By implementing these functions, the framework can seamlessly integrate your
custom components.

Here is how you do it (theoretically):

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

  from bsb import config, MorphologySelector

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
