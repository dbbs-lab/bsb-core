.. DBBS Cerebellum Scaffold documentation master file, created by
   sphinx-quickstart on Tue Oct 29 12:24:53 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Brain Scaffold Builder
==========================

The BSB is a framework for reconstructing and simulating multi-paradigm neuronal network
models. It removes much of the repetitive work associated with writing the required code
and lets you focus on the parts that matter. It helps write organized, well-parametrized
and explicit code understandable and reusable by your peers.

.. grid:: 1 1 2 2
    :gutter: 1

    .. grid-item-card:: :octicon:`flame;1em;sd-text-warning` Get started
      :link: get-started
      :link-type: ref

      Get started with your first project!

    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Components
	    :link: components
	    :link-type: ref

	    Learn how to write your own components to e.g. place or connect cells.

    .. grid-item-card:: :octicon:`database;1em;sd-text-warning` Simulations
	    :link: simulations
	    :link-type: ref

	    Learn how to simulate your network models

    .. grid-item-card:: :octicon:`device-camera-video;1em;sd-text-warning` Examples
	    :link: examples
	    :link-type: ref

	    View examples explained step by step

    .. grid-item-card:: :octicon:`package-dependents;1em;sd-text-warning` Plugins
	    :link: plugins
	    :link-type: ref

	    Learn to package your code for others to use!

    .. grid-item-card:: :octicon:`octoface;1em;sd-text-warning` Contributing
	    :link: https://github.com/dbbs-lab/bsb

	    Help out the project by contributing code.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   usage/installation
   usage/top-level-guide
   usage/getting-started
   usage/projects

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   config/configuration-toc


.. toctree::
  :maxdepth: 2
  :caption: CLI

  cli/cli-toc

.. toctree::
  :maxdepth: 2
  :caption: Cell types

  cells/cells-toc

.. toctree::
  :maxdepth: 2
  :caption: Topology

  topology/topology-toc

.. toctree::
  :maxdepth: 2
  :caption: Morphologies

  morphologies/morphology-toc

.. toctree::
  :maxdepth: 2
  :caption: Placement

  placement/placement-toc

.. toctree::
  :maxdepth: 2
  :caption: Simulation

  simulation/simulation-toc

.. toctree::
   :maxdepth: 2
   :caption: References

   bsb/modules
   genindex
   py-modindex

.. _all-guides:

.. toctree::
  :maxdepth: 1
  :caption: User Guides

  guides/components
  guides/connectivity
  guides/connection-strategies
  guides/packaging
  examples/toc

.. toctree::
  :maxdepth: 2
  :caption: Developer Guides:

  dev/installation
  dev/documentation
  dev/plugins
  dev/hooks
  dev/reference
