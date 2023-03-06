###############
Type validation
###############

Configuration types convert given configuration values. Values incompatible with the type
are rejected and the user is warned. The default type is ``str``.

Any callable that takes 1 argument can be used as a type handler. The :mod:`.config.types`
module provides extra functionality such as validation of list and dictionaries and even
more complex combinations of types. Every configuration node itself can be used as a type.

.. warning::

	All of the members of the :mod:`.config.types` module are factory methods: they need to
	be **called** in order to produce the type handler. Make sure that you use
	``config.attr(type=types.any_())``, as opposed to ``config.attr(type=types.any_)``.

Examples
--------

.. code-block:: python

  from bsb import config
  from bsb.config import types

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
