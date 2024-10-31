.. _config_attrs:

########################
Configuration attributes
########################

An attribute can refer to a singular value of a certain type, a dict, list, reference, or
to a deeper node. You can use the :func:`config.attr <.config.attr>` in node decorated
classes to define your attribute:

.. code-block:: python

  from bsb import config

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

Type validation
===============

Configuration types convert given configuration values. Values incompatible with the type
are rejected and the user is warned. The default type is ``str``.

Any callable that takes 1 argument can be used as a type handler. The :mod:`.config.types`
module provides extra functionality such as validation of list and dictionaries and even
more complex combinations of types. Every configuration node itself can be used as a type.

.. warning::

    All of the members of the :mod:`.config.types` module are factory methods: they need to
    be **called** in order to produce the type handler. Make sure that you use
    ``config.attr(type=types.any_())``, as opposed to ``config.attr(type=types.any_)``.

.. _config_dict:

Configuration dictionaries
==========================

Configuration dictionaries hold configuration nodes. If you need a dictionary of values
use the :func:`types.dict <.config.types.dict>` syntax instead.

.. code-block:: python

  from bsb import config

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

.. _config_list:

Configuration lists
===================

Configuration dictionaries hold unnamed collections of configuration nodes. If you need a
list of values use the :func:`types.list <.config.types.list>` syntax instead.

.. code-block:: python

  from bsb import config

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

.. _config_ref:

Configuration references
========================

References refer to other locations in the configuration. In the configuration the
configured string will be fetched from the referenced node:

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

References are defined inside of configuration nodes by passing a `reference object
<quick-reference-object>`_ to the :func:`.config.ref` function:

.. code-block:: python

  @config.node
  class Locations:
    locations = config.dict(type=str)
    where = config.ref(lambda root, here: here["locations"])

After the configuration has been cast all nodes are visited to check if they are a
reference and if so the value from elsewhere in the configuration is retrieved. The
original string from the configuration is also stored in ``node.<ref>_reference``.

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
This behavior is controlled by the ``ref_type`` keyword argument on the ``config.ref``
call and the ``is_ref`` method on the reference object. If neither is given it defaults to
checking whether the value is an instance of ``str``:

.. code-block:: python

  @config.node
  class CandySelect:
    candies = config.dict(type=Candy)
    special_candy = config.ref(lambda root, here: here.candies, ref_type=Candy)

  class CandyReference(config.refs.Reference):
    def __call__(self, root, here):
      return here.candies

    def is_ref(self, value):
      return isinstance(value, Candy)

  @config.node
  class CandySelect:
    candies = config.dict(type=Candy)
    special_candy = config.ref(CandyReference())

The above code will make sure that only ``Candy`` objects are seen as references and all
other types are seen as keys that need to be looked up. It is recommended you do this even
in trivial cases to prevent bugs.

.. _quick-reference-object:

Reference object
----------------

The reference object is a callable object that takes 2 arguments: the configuration root
node and the referring node. Using these 2 locations it should return a configuration node
from which the reference value can be retrieved.

.. code-block:: python

  def locations_reference(root, here):
    return root.locations

This reference object would create the link seen in the first reference example.

Reference lists
---------------

Reference lists are akin to references but instead of a single key they are a list of
reference keys:

.. code-block:: json

  {
    "locations": {"A": "very close", "B": "very far"},
    "where": ["A", "B"]
  }

Results in ``cfg.where == ["very close", "very far"]``. As with references you can set a
new list and all items will either be looked up or kept as is if they're a reference value
already.

.. warning::

  Appending elements to these lists currently does not convert the new value. Also note
  that reference lists are quite indestructible; setting them to `None` just resets them
  and the reference key list (``.<attr>_references``) to ``[]``.


Bidirectional references
------------------------

The object that a reference points to can be "notified" that it is being referenced by the
``populate`` mechanism. This mechanism stores the referrer on the referee creating a
bidirectional reference. If the ``populate`` argument is given to the ``config.ref`` call
the referrer will append itself to the list on the referee under the attribute given by
the value of the ``populate`` kwarg (or create a new list if it doesn't exist).

.. code-block:: json

  {
    "containers": {
      "A": {}
    },
    "elements": {
      "a": {"container": "A"}
    }
  }

.. code-block:: python

  @config.node
  class Container:
    name = config.attr(key=True)
    elements = config.attr(type=list, default=list, call_default=True)

  @config.node
  class Element:
    container = config.ref(container_ref, populate="elements")

This would result in ``cfg.containers.A.elements == [cfg.elements.a]``.

You can overwrite the default *append or create* population behavior by creating a
descriptor for the population attribute and define a ``__populate__`` method on it:

.. code-block:: python

  class PopulationAttribute:
    # Standard property-like descriptor protocol
    def __get__(self, instance, objtype=None):
      if instance is None:
        return self
      if not hasattr(instance, "_population"):
        instance._population = []
      return instance._population

    # Prevent population from being overwritten
    # Merge with new values into a unique list instead
    def __set__(self, instance, value):
      instance._population = list(set(instance._population) + set(value))

    # Example that only stores referrers if their name in the configuration is "square".
    def __populate__(self, instance, value):
      print("We're referenced in", value.get_node_name())
      if value.get_node_name().endswith(".square"):
        self.__set__(instance, [value])
      else:
        print("We only store referrers coming from a .square configuration attribute")

.. todo: Mention ``pop_unique``

Examples
========

.. code-block:: python

  from bsb import config, types

  @config.node
  class TestNode
    name = config.attr()

  @config.node
  class TypeNode
    # Default string
    some_string = config.attr()
    # Explicit & required string
    required_string = config.attr(type=str, required=True)
    # Float
    some_number = config.attr(type=float)
    # types.float / types.int
    bounded_float = config.attr(type=types.float(min=0.3, max=17.9))
    # Float, int or bool (attempted to cast in that order)
    combined = config.attr(type=types.or_(float, int, bool))
    # Another node
    my_node = config.attr(type=TestNode)
    # A list of floats
    list_of_numbers = config.attr(
      type=types.list(type=float)
    )
    # 3 floats
    list_of_numbers = config.attr(
      type=types.list(type=float, size=3)
    )
    # A scipy.stats distribution
    chi_distr = config.attr(type=types.distribution())
    # A python statement evaluation
    statement = config.attr(type=types.evaluation())
    # Create an np.ndarray with 3 elements out of a scalar
    expand = config.attr(
        type=types.scalar_expand(
            scalar_type=int,
            expand=lambda s: np.ones(3) * s
        )
    )
    # Create np.zeros of given shape
    zeros = config.attr(
        type=types.scalar_expand(
            scalar_type=types.list(type=int),
            expand=lambda s: np.zeros(s)
        )
    )
    # Anything
    any_ = config.attr(type=types.any_())
    # One of the following strings: "all", "some", "none"
    give_me = config.attr(type=types.in_(["all", "some", "none"]))
    # The answer to life, the universe, and everything else
    answer = config.attr(type=lambda x: 42)
    # You're either having cake or pie
    cake_or_pie = config.attr(type=lambda x: "cake" if bool(x) else "pie")
