"""
    Module for the Layer configuration node and its dependencies.
"""

from .. import config


@config.node
class Layer:
    name = config.attr(key=True)
    thickness = config.attr(type=float, required=True)
