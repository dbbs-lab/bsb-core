=================
Cell Connectivity
=================

Cell connections are made as the second step of compilation. Each connection
type configures one :class:`.connectivity.ConnectionStrategy` and can override
the ``connect`` method to connect cells to eachother. Use the scaffold instance's
:func:.core.Scaffold.connect_cells` to connect cells to eachother.

See the :doc:`/guides/connection-strategies`.

*************
Configuration
*************

Each ConnectionStrategy is a ConfigurableClass, meaning that the attributes from
the configuration files will be copied and validated onto the connection object.

****************
Connecting cells
****************

The connection matrices use a 2 column, 2 dimensional ndarray where the columns
are the from and to id respectively. For morphologically detailed connections
additional identifiers can be passed into the function to denote the specific
compartments and morphologies that were used.
