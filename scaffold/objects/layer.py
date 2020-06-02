"""
    Module for the Layer configuration node and its dependencies.
"""

from .. import config


@config.node
class Layer:
    thickness = config.attr(type=float, required=True)
