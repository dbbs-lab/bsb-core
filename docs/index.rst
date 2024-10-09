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

.. grid:: 1 1 4 2
    :gutter: 1

    .. grid-item-card:: :octicon:`desktop-download;1em;sd-text-warning` Installation
      :link: installation-guide
      :link-type: ref

      How to install the code.

    .. grid-item-card:: :octicon:`flame;1em;sd-text-warning` Get started
      :link: get-started
      :link-type: ref

      Get started with your first project!

    .. grid-item-card:: :octicon:`device-camera-video;1em;sd-text-warning` Examples
       :link: examples
       :link-type: ref

       View examples explained step by step

    .. grid-item-card:: :octicon:`repo-clone;1em;sd-text-warning` BSB - CLI
       :link: cli-guide
       :link-type: ref

       Learn how to use the CLI.

    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Contributing
       :link: development-section
       :link-type: ref

       Help out the project by contributing code.

    .. grid-item-card:: :octicon:`gear;1em;sd-text-warning` Components
       :link: main-components
       :link-type: ref

       Explore more about the main components.

    .. grid-item-card:: :octicon:`briefcase;1em;sd-text-warning` Python API
       :link: whole-api
       :link-type: ref


    .. grid-item-card:: :octicon:`info;1em;sd-text-warning` FAQ
       :link: faq
       :link-type: ref



.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/installation
   getting-started/toc
   guides/toc
   examples/toc

.. toctree::
  :maxdepth: 2
  :caption: CLI

  cli/cli-toc

.. toctree::
   :maxdepth: 2
   :caption: Components

   components/intro
   config/configuration-toc
   topology/topology-toc
   cells/cells-toc
   morphologies/morphology-toc
   placement/placement-toc
   connectivity/connectivity-toc
   simulation/simulation-toc


.. toctree::
   :maxdepth: 2
   :caption: References

   bsb/modules
   genindex
   py-modindex

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   dev/faq

.. toctree::
  :maxdepth: 2
  :caption: Developer Guides:

  dev/installation
  dev/documentation
  dev/guidelines
  dev/services
  dev/plugins
  dev/hooks
  dev/reference
