#####
Nodes
#####

Nodes are the recursive backbone backbone of the Configuration object. Nodes can contain
other nodes under their attributes and in that way recurse deeper into the configuration.
Nodes can also be used as types of configuration dictionaries or lists.

Node classes contain the description of a node type in the configuration. Here's an example
to illustrate:

.. code-block:: python

  from scaffold import config

  @config.node
  class CellType:
    name = config.attr(key=True)
    color = config.attr()
    radius = config.attr(type=float, required=True)

This node class describes the following configuration:

.. code-block:: json

  {
    "cell_type_name": {
      "radius": 13.0,
      "color": "red"
    }
  }

The ``@config.node`` decorator takes the ordinary class and injects the logic it needs
to fulfill the tasks of a configuration node. Whenever a node of this type is used
in the configuration an instance of the node class is created and some work needs to happen:

* The parsed configuration dictionary needs to be cast into an instance of the node class.
* The configuration attributes of this node class and its parents need to be collected.
* The attributes on this instance need to be initialized with a default value or ``None``.
* The keys that are present in the configuration dictionary need to be transferred to the
  node instance and converted to the specified type (the default type is ``str``)

Dynamic nodes
=============

Dynamic nodes are those whose node class is configurable from inside the configuration node itself.
This is done through the use of the ``@dynamic`` decorator instead of the node decorator.
This will automatically create a required ``class`` attribute.

The value that is given to this class attribute will be used to import a class to instantiate
the node:

.. code-block:: python

  @config.dynamic
  class PlacementStrategy:
    @abc.abstractmethod
    def place(self):
      pass

And in the configuration:

.. code-block:: json

  {
    "class": "scaffold.placement.LayeredRandomWalk"
  }

This would import ``scaffold.placement`` and from the module use the ``LayeredRandomWalk``
class to create the node.

.. note::

	The child class needs to inherit from the node class on which the ``@dynamic`` decorator
	is applied.


Classmaps
---------

A preset map of shorter entries can be given to be mapped to an absolute or
relative class path, or a class object:

.. code-block:: python

   @dynamic(class_map={"short": "pkg.with.a.long.name.DynClass"})
   class Example:
       pass

If ``short`` is used the dynamic class will resolve to ``pkg.with.a.long.name.DynClass``.
Automatic class maps can be generated using the ``auto_class_map`` attribute:

.. code-block:: python

  @dynamic(auto_class_map=True)
  class Example:
    pass

  class MappedChild(Example, class_map_entry="short"):
    pass

This will generate a mapping from ``short`` to the ``MappedChild`` class.

Root node
=========

The root node is the Configuration object and is at the basis of the tree of nodes.
