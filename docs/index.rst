.. DBBS Cerebellum Scaffold documentation master file, created by
   sphinx-quickstart on Tue Oct 29 12:24:53 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Brain Scaffold Builder
==========================

The BSB is a **black box component framework** for multiparadigm neural modelling: we
provide structure, architecture and organization, and you provide the use-case specific
parts of your model. In our framework, your model is described in a code-free
configuration of **components** with parameters.

For the framework to reliably use components, and make them work together in a complex
workflow, it asks a fixed set of questions per component type: e.g. a connection component
will be asked how to connect cells. These contracts of cooperation between you and the
framework are called **interfaces**. The framework executes a transparently
parallelized workflow, and calls your components to fulfill their role.

This way, by *implementing our component interfaces* and declaring them in a
configuration file, most models end up being code-free, well-parametrized, self-contained,
human-readable, multi-scale models!

(PS: If we missed any hyped-up hyphenated adjectives, let us know! |:heart:|)

----

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

    .. grid-item-card:: :octicon:`mark-github;1em;sd-text-warning` Contributing
	    :link: https://github.com/dbbs-lab/bsb

	    Help out the project by contributing code.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   usage/installation
   usage/top-level-guide
   usage/getting-started
   usage/projects
   guides/toc
   examples/toc

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
  :caption: Topology

  topology/topology-toc

.. toctree::
  :maxdepth: 2
  :caption: Cell types

  cells/cells-toc

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
  :caption: Connectivity

  connectivity/connectivity-toc

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

.. toctree::
  :maxdepth: 2
  :caption: Developer Guides:

  dev/installation
  dev/documentation
  dev/services
  dev/plugins
  dev/hooks
  dev/reference
