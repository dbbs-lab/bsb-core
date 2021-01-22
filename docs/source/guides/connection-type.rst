================
Connection types
================

Connection types connect cell types together after they've been placed into the simulation
volume. They are defined in the configuration under ``connection_types``:

.. code-block:: json

  {
    "connection_types": {
      "cell_A_to_cell_B": {
        "class": "bsb.connectivity.VoxelIntersection",
        "from_cell_types": [
          {
            "type": "cell_A",
            "compartments": ["axon"]
          }
        ],
        "to_cell_types": [
          {
            "type": "cell_B",
            "compartments": ["dendrites", "soma"]
          }
        ]
      }
    }
  }

The :guilabel:`class` specifies which ``ConnectionStrategy`` to load for this conenction
type. The :guilabel:`from_cell_types` and :guilabel:`to_cell_types` specify which pre- and
postsynaptic cell types to use respectively. The cell type definitions in those lists have
to contain a :guilabel:`type` that links to an existing cell type and can optionally
contain hints to which :guilabel:`compartments` of the morphology to use.

Creating your own
=================

In order to create your own connection type, create an importable module (refer to the
`Python documentation <https://docs.python.org/3/tutorial/modules.html>`_) with inside
a class inheriting from :class:`.connectivity.ConnectionStrategy`. Let's start by
deconstructing a full code example:

.. code-block:: python

  from bsb.connectivity import ConnectionStrategy
  from bsb.exceptions import ConfigurationError
  import scipy.spatial.distance as dist

  class ConnectBetween(ConnectionStrategy):

    casts = {
      "min": float,
      "max": float,
    }

    defaults = {
      "min": 0.,
    }

    required = ["max"]

    def validate(self):
      if self.max < self.min:
        raise ConfigurationError("Max distance should be larger than min distance.")


    def connect(self):
      for ft in self.from_cell_types:
        ps_from = self.scaffold.get_placement_set(ft)
        fpos = ps_from.positions
        for tt in self.to_cell_types:
          ps_to = self.scaffold.get_placement_set(tt)
          tpos = ps_to.positions
          pairw_dist = dist.cdist(fpos, tpos)
          pairs = ((pairw_dist <= max) & (pairw_dist >= min)).nonzero()
          # More code to convert `pairs` into a Nx2 matrix of pre & post synaptic pair IDs
          # ...
          self.scaffold.connect_cells(f"{ft.name}_to_{tt.name}", pairs)

.. code-block:: json

  {
    "connection_types": {
      "cell_A_to_cell_B": {
        "class": "my_module.ConnectBetween",
        "min": 10,
        "max": 15.5,
        "from_cell_types": [
          {
            "type": "cell_A"
          }
        ],
        "to_cell_types": [
          {
            "type": "cell_B"
          }
        ]
      }
    }
  }

Configuration attributes
------------------------

All keys present on the connection type in the configuration will be available on the
connection strategy under ``self.<key>`` (e.g. :guilabel:`min` will become ``self.min``).
Additionally the scaffold object is available under ``self.scaffold``.

Configuration attributes will by default have the data type they have in JSON, which can
be any of ``int``, ``float``, ``str``, ``list`` or ``dict``. This data type can be
overridden by using the class attribute ``casts``. Any key present in this dictionary
will use the value as a conversion function if the configuration attribute is encountered.

In this example both :guilabel:`min` and :guilabel:`max` will be converted to ``float``.
You can also provide your own functions or lambdas as long as they take the configuration
value as only argument:

.. code-block:: python

  casts = {"cake_or_pie": lambda x: "pie" if x < 10 else "cake"}

You can provide default values for configuration attributes giving the ``defaults`` class
variable dictionary. You can also specify that certain attributes are ``required`` to be
provided. If they occur in the ``defaults`` dictionary the default value will be used
when no value is provided in the configuration.

Validation handling
-------------------

The given configuration attributes can be further validated using the ``validate`` method.
From inside the ``validate`` method a ``ConfigurationError`` can be thrown when the user
given values aren't valid. This method is required, if no validation is required a noop
function should be given:

.. code-block:: python

  def validate(self):
    pass

Connection handling
-------------------

Inside of the ``connect`` function the from and to cell types will be available. You can
access their placement data using ``self.scaffold.get_placement_set(type)``. The
properties of a ``PlacementSet`` are expensive IO operations, cache them:

.. code-block:: python

  # WRONG! Will read the data from file 200 times
  for i in range(100):
    ps1.positions - ps2.positions

  # Correct! Will read the data from file only 2 times
  pos1 = ps1.positions
  pos2 = ps2.Positions
  for i in range(100):
    pos1 - pos2

Finally you should call ``self.scaffold.connect_cells(tag, matrix)`` to connect the cells.
The tag is free to choose, the matrix should be rows of pre to post cell ID pairs.
