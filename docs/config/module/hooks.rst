###################
Configuration hooks
###################

The BSB provides a small and elegant hook system. The system allows the user to hook
methods of classes. It is intended to be a hooking system that requires bidirectional
cooperation: the developer declares which hooks they provide and the user is supposed to
only hook those functions. Using the hooks in other places will behave slightly different,
see the note on `wild hooks`_.

For a list of BSB endorsed hooks see `list of hooks`_.

=============
Calling hooks
=============

A developer can call the user-registered hook using :func:`bsb.config.run_hook`:

.. code-block:: python

  import bsb.config

  bsb.config.run_hook(instance, "my_hook")

This will check the class of instance and all of its parent classes for implementations of
``__my_hook__`` and execute them in closest relative first order, starting from the class
of ``instance``. These ``__my_hook_`` methods are known as `essential hooks`_.

============
Adding hooks
============

Hooks can be added to class methods using the :func:`bsb.config.on` decorator (or
:func:`bsb.config.before`/:func:`bsb.config.after`). The decorated function will then be
hooked onto the given class:

.. code-block:: python

  from bsb import config
  from bsb.core import Scaffold
  from bsb.simulation import Simulation

  @config.on(Simulation, "boot")
  def print_something(self):
    print("We're inside of `Simulation`'s `boot` hook!")
    print(f"The {self.name} simulation uses {self.simulator}.")

  cfg = config.Configuration.default()
  cfg.simulations["test"] = Simulation(simulator="nest", ...)
  scaffold = Scaffold(cfg)
  # We're inside of the `Simulation`s `boot` hook!
  # The test simulation uses nest.


===============
Essential hooks
===============

Essential hooks are those that follow Python's "magic method" convention (``__magic__``).
Essential hooks allow parent classes to execute hooks even if child classes override the
direct ``my_hook`` method. After executing these essential hooks ``instance.my_hook`` is
called which will contain all of the non-essential class hooks. Unlike non-essential hooks
they are not run whenever the hooked method is executed but only when the hooked method is
invoked through

==========
Wild hooks
==========

Since the non-essential hooks are wrappers around the target method you could use the
hooking system to hook methods of classes that aren't ever invoked as a hook, but still
used during the operation of the class and your hook will be executed anyway. You could
even use the hooking system on any class not part of the BSB at all. Just keep in mind
that if you place an essential hook onto a target method that's never explicitly invoked
as a hook that it will never run at all.

=============
List of hooks
=============

``__boot__``?
