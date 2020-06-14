########
Overview
########

====================
Role in the scaffold
====================

Configuration plays a key role in the scaffold builder. It is the main mechanism to
describe a model. A scaffold model can be initialized from a Configuration object, either
from a standalone file or provided by the :doc:`Storage </storage/overview>`. In both
cases the raw configuration string is parsed into a Python tree of dictionaries and lists.
This configuration tree is then passed to the Configuration class for :ref:`casting
<configuration-casting>`. How a tree is to be cast into a Configuration object can be
described using configuration unit syntax.

===================
Configuration units
===================

When the configuration tree is being cast into a Configuration object there are 5 key
units:

- A **configuration attribute** represented by a key-value pair.
- A **configuration reference** points to another location in the configuration.
- A **configuration node** represented by a dictionary.
- A **configuration dictionary** represented by a dictionary where each key-value pair
  represents another configuration unit.
- A **configuration list** represented by a list where each value represents another
  configuration unit.

.. note::

  If a list or dictionary contains regular values instead of other configuration units,
  the :func:`types.list <.config.types.list>` and :func:`types.dict
  <.config.types.dict>` are used instead of the :func:`conf.list <.config.list>` and
  :func:`conf.dict <.config.dict>`.

Configuration nodes
===================

A node in the configuration can be described by creating a class and applying the
``@config.node`` decorator to it. This decorator will look for ``config.attr`` and other
configuration unit constructors on the class to create the configuration information on
the class. This node class can then be used in the type argument of another configuration
attribute, dictionary, or list:

.. code-block:: python

  from scaffold import config

  @config.node
  class CandyNode:
    name = config.attr(type=str, required=True)
    sweetness = config.attr(type=float, default=3.0)

This candy node class now represents the following JSON dictionary:

.. code-block:: json

  {
    "name": "Lollypop",
    "sweetness": 12.0
  }

You will mainly design configuration nodes and other configuration logic when designing
custom strategies.

Dynamic nodes
-------------

An important part to the interfacing system of the scaffold builder are custom strategies.
Any user can implement a simple functional interface such as the :class:`PlacementStrategy
<.placement.strategy.PlacementStrategy>` to design a new way of placing cells. Placement
configuration nodes can then use these strategies by specifying the :guilabel:`class`
attribute:

.. code-block:: json

  {
    "my_cell_type": {
      "placement": {
        "class": "my_package.MyStrategy"
      }
    }
  }

This dynamic loading is achieved by creating a node class with the ``@config.dynamic``
decorator instead of the node decorator. This will add a configuration attribute ``class``
to the node class and use the value of this class to create an instance of another node
class, provided that the latter inherits from the former, enforcing the interface.

.. code-block:: python

  @config.dynamic
  class PlacementStrategy:
    @abc.abstractmethod
    def place(self):
      pass

Configuration attributes
========================

An attribute can refer to a singular value of a certain type, or to another node:

.. code-block:: python

  from scaffold import config

  @config.node
  class CandyStack:
    count = config.attr(type=int, required=True)
    candy = config.attr(type=CandyNode)

.. code-block:: json

  {
    "count": 12,
    "candy": {
      "name": "Hardcandy",
      "sweetness": 4.5
    }
  }

Configuration dictionaries
==========================

Configuration dictionaries hold configuration nodes. If you need a dictionary of values
use the :class:`types.dict <.config.types.dict>` syntax instead.

.. code-block:: python

  from scaffold import config

  @config.node
  class CandyNode:
    name = config.attr(key=True)
    sweetness = config.attr(type=float, default=3.0)

  @config.node
  class Inventory:
    candies = config.dict(type=CandyStack)

.. code-block:: json

  {
    "candies": {
      "Lollypop": {
        "sweetness": 12.0
      },
      "Hardcandy": {
        "sweetness": 4.5
      }
    }
  }

Items in configuration dictionaries can be accessed using dot notation or indexing:

.. code-block:: python

  inventory.candies.Lollypop == inventory.candies["Lollypop"]

Using the ``key`` keyword argument on a configuration attribute will pass the key in the
dictionary to the attribute so that ``inventory.candies.Lollypop.name == "Lollypop"``.

Configuration lists
===================

Configuration dictionaries hold unnamed collections of configuration nodes. If you need a
list of values use the :class:`types.list <.config.types.list>` syntax instead.

.. code-block:: python

  from scaffold import config

  @config.node
  class InventoryList:
    candies = config.list(type=CandyStack)

.. code-block:: json

  {
    "candies": [
      {
        "count": 100,
        "candy": {
          "name": "Lollypop",
          "sweetness": 12.0
        }
      },
      {
        "count": 1200,
        "candy": {
          "name": "Hardcandy",
          "sweetness": 4.5
        }
      }
    ]
  }

Configuration references
========================

References refer to other locations in the configuration. In the configuration the configured string
will be fetched from the referenced node:

.. code-block:: json

  {
    "locations": {"A": "very close", "B": "very far"},
    "where": "A"
  }

Assuming that ``where`` is a reference to ``locations``, location ``A`` will be retrieved
and placed under ``where`` so that in the config object:

.. code-block:: python

  >>> print(conf.locations)
  {'A': 'very close', 'B': 'very far'}

  >>> print(conf.where)
  'very close'

  >>> print(conf.where_reference)
  'A'

They are defined inside of configuration nodes by passing a reference object to the
``config.ref`` function. After the configuration has been cast all nodes are visited to
check if they contain are a reference and if so the value from elsewhere in the
configuration is retrieved, the original string from the configuration is also stored in
``node.<ref>_reference`` as demonstrated in the example above.

After the configuration is loaded it's possible to either give a new reference key
(usually a string) or a new reference value. In most cases the configuration will
automatically detect what you're passing into the reference:

.. code-block::

  >>> cfg = from_json("mouse_cerebellum.json")
  >>> cfg.cell_types.granule_cell.placement.layer.name
  'granular_layer'
  >>> cfg.cell_types.granule_cell.placement.layer = 'molecular_layer'
  >>> cfg.cell_types.granule_cell.placement.layer.name
  'molecular_layer'
  >>> cfg.cell_types.granule_cell.placement.layer = cfg.layers.purkinje_layer
  >>> cfg.cell_types.granule_cell.placement.layer.name
  'purkinje_layer'

As you can see, by passing the reference a string the object is fetched from the reference
location, but we can also directly pass the object the reference string would point to.

Reference object
----------------

The reference object is a callable object that takes 2 arguments: the configuration root
node and the referring node. Using these 2 locations it should return a configuration node
from which the reference value can be retrieved.

.. code-block:: python

  def locations_reference(root, here):
    return root.locations

This reference object would create the link seen in the first reference example. For more
advanced uses of the reference object see :doc:`/config/module/ref`.

.. _configuration-casting:

=======
Casting
=======

When the Configuration object is loaded it is cast from a tree to an object. This happens
recursively starting at a configuration root. The default :class:`Configuration
<.config.Configuration>` root is defined in ``scaffold/config/_config.py`` and describes
how the scaffold builder will read a configuration tree.

You can cast from configuration trees to configuration nodes yourself by using the class
method ``__cast__``:

.. code-block:: python

  inventory = {
    "candies": {
      "Lollypop": {
        "sweetness": 12.0
      },
      "Hardcandy": {
        "sweetness": 4.5
      }
    }
  }

  # The second argument would be the node's parent if it had any.
  conf = Inventory.__cast__(inventory, None)
  print(conf.candies.Lollypop.sweetness)
  >>> 12.0

Casting from a root node also resolves references.
