Distribution-based Placement Strategy
=====================================

.. note::
    This example presents in more advanced tutorial to write BSB components.
    If you are not familiarized with BSB components, check out
    the getting started :doc:`section </getting-started/guide_components>`
    on writing  components.

We will start from the following configuration file (corresponds to the first network file
from the getting started tutorial):

.. literalinclude:: /getting-started/configs/getting-started.json
    :language: json

Let's save this new configuration in our project folder under the name ``config_placement.json``

Description of the strategy to implement
----------------------------------------

We want here to implement a distribution placement strategy: cells will be placed within
their `Partition` following a probability ``distribution`` along a certain ``axis`` and
``direction``. For instance, let us use the
:doc:`alpha random distribution <scipy:reference/generated/scipy.stats.alpha>`.
The ``distribution`` should be a density function that produces random numbers, according
to the distance along the ``axis`` from a border of the Partition.

Here, we want to control the distribution of the cells within the ``base_layer``
with respect to the border with the ``top_layer``.

Components boiler plate
-----------------------

We will write our `PlacementStrategy` class in a ``placement.py`` file,

.. code-block:: python

    from bsb import config, PlacementStrategy

    @config.node
    class DistributionPlacement(PlacementStrategy):

        # add your attributes here

        def place(self, chunk, indicators):
            # write your code here
            pass

Here, our class will extend from
:class:`PlacementStrategy <.placement.strategy.PlacementStrategy>` which is an
abstract class that requires you to implement the
:meth:`place <.placement.strategy.PlacementStrategy.place>` function.

Note that this strategy leverages the ``@config.node`` `python decorator`.
The :doc:`configuration node decorator</config/nodes>` allows you to pass the parameters
defined in the configuration file to the class. It will also handle the
:doc:`type testing </config/types>` of your configuration attributes (e.g., make sure your
``axis`` parameter is a positive integer). We will see in the following sections how
to create your class configuration attributes.

Add configuration attributes
----------------------------

For our strategy, we need to pass a list of parameters to its class
through our configuration file:

- a density function ``distribution``, defined by the user, from the list of
  :doc:`scipy stats <scipy:reference/stats>` functions.
- an ``axis`` along which to apply the distribution. The latter should be in [0, 1, 2]
- a ``direction`` that if set will apply the distribution along the ``"negative"`` direction,
  i.e from top to bottom (``"positive"`` direction if not set).

This translates into 3 configuration attributes that you can add to your class:

.. code-block:: python

    from bsb import config, PlacementStrategy, types

    @config.node
    class DistributionPlacement(PlacementStrategy):

        distribution = config.attr(type=types.distribution(), required=True)
        axis: int = config.attr(type=types.int(min=0, max=2), required=False, default=2)
        direction: str = config.attr(type=types.in_(["positive", "negative"]),
                                     required=False, default="positive")

        def place(self, chunk, indicators):
            # write your code here
            pass

| In this case, ``distribution`` is required, and should correspond to a
  :class:`distribution <.config._distributions.Distribution>` node which interface scipy distributions.
| ``axis`` here is an optional integer attribute with a default value set to 2.
| Finally, ``direction`` is an optional string attribute that can be either the string ``"positive"``
  or ``"negative"`` (see :func:`in_ <.config.types.in_>`).

At this stage, you have created a python class with minimal code implementation, you should
now link it to your configuration file. To import our class in our configuration file, we
will modify the :guilabel:`placement` block:

.. code-block:: json

  "placement": {
    "alpha_placement": {
      "strategy": "placement.DistributionPlacement",
      "distribution": {
        "distribution": "alpha",
        "a": 8
      },
      "axis": 0,
      "direction": "negative",
      "cell_types": ["base_type"],
      "partitions": ["base_layer"]
    },
    "top_placement": {
      "strategy": "bsb.placement.RandomPlacement",
      "cell_types": ["top_type"],
      "partitions": ["top_layer"]
    }
  }

Implement the python methods
----------------------------
The `place` function will be used here to produce and store a
:class:`PlacementSet <.storage.interfaces.PlacementSet>` for each `cell type` population
to place in the selected `Partition`.
BSB is parallelizing placement jobs for each `Chunk` concerned.

The parameters of `place` includes a dictionary linking each cell type name to its
:class:`PlacementIndicator <.placement.indicator.PlacementIndicator>`, and the `Chunk`
in which to place the cells.

We need to apply our distribution to each Partition of our circuit to see how
the cells are distributed within, along our directed axis.
Let's make two for loops to iterate over the Partitions of each indicator.
Then, we extract the number of cells to place within the total Partition, using
the :meth:`guess <.placement.indicator.PlacementIndicator.guess>` function.
For that, we will convert the partition into a list of voxels.

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python
    :lines: 54-62

Now, our function is supposed to place cells within only one Chunk. Fortunately,
Partitions can be decomposed into Chunks. So, we can retrieve from the
distribution the number of cells to place within the ``chunk`` parameter of the
function, according to its position along the directed ``axis``.

To do so, we need to define the interval occupied by the chunk within the partition.
We will leverage the lowest and highest coordinates of the chunk and partition with
respectively the attributes ``ldc`` and ``mdc``.

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python
    :lines: 63-68

``bounds`` is here the interval of ratios of the space occupied by a Chunk within the Partition
along the chosen axis.

We also need to take into account the case where the direction is negative. In this case, we should
invert the interval. E.g., if the ``bounds`` is [0.2, 0.3] then the inverted interval is [0.7, 0.8].

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python
    :lines: 68-73

Great, we have the interval on which we want to place our cells.
Now, we will check how many cells to place within this interval according to our distribution
along the provided axis, knowing the total number of cells to place within the partition.
We will create a separate function for this called ``draw_interval``.
Additionally, we also need to take into account the two other dimensions. We will compute the
ratio of area occupied by the chunk along the two other directions:

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python
    :lines: 75-82

So, your final place function should look like this

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python
    :lines: 54-89

``draw_interval`` will leverage an
`acceptance-rejection method <https://en.wikipedia.org/wiki/Rejection_sampling>`_ .
In short, this method will draw n random values within [0, 1] and return the number which value
is less than the probability according to the the distribution to fall in the provided interval
boundaries.
We will retrieve the interval of definition of the distribution and within boundaries of our
provided ratio interval.

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python
    :lines: 34-41

From the distribution, we can retrieve the probability for a drawn random value be lesser than
the upper bound and the probability for it to be greater than the lower bound.

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python
    :lines: 42-45

Finally, we apply the acceptance-rejection algorithm to see how much of the cells to place in
the partition should be placed in the chunk:

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python
    :lines: 46-52

We are done with the Placement! Here is how the full strategy looks like:

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python

Enjoy
-----

You have done the hardest part! Now, you should be able to run the reconstruction once again
with your brand new component.

.. code-block:: bash

    bsb compile config_placement.json --verbosity 3
