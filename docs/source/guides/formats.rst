==============
Output Formats
==============

.. _format_nc_list:

Nearly-continuous list
======================

This format is used to store lists that are almost always just a sequence of continuous
numbers. It will always contain pairs that describe a continuous chain of numbers as a
start and length.

For example this sequence::

    [15, 3, 30, 4]

Describes 3 numbers starting from 15 and 4 numbers starting from 30::

    [15, 16, 17, 30, 31, 32, 33]

See :func:`.helpers.continuity_list` for the implementation.

.. note::

    The scaffold generates continuous IDs, but this assumption does not hold true in many edge
    cases like manually placing cells, using custom placement strategies or after
    postprocessing the placed cells.
