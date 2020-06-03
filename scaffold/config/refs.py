"""
    This module contains shorthand ``reference`` definitions. References are used in the
    configuration module to point to other locations in the Configuration object.

    Minimally a reference is a function that takes the configuration root and the current
    node as arguments, and returns another node in the configuration object::

      def some_reference(root, here):
          return root.other.place

    More advanced usage of references will include custom reference errors.
"""


class Reference:
    def __call__(self, root, here):
        return here


class CellTypeReference(Reference):
    def __call__(self, root, here):
        return root.cell_types


class LayerReference(Reference):
    def __call__(self, root, here):
        return root.layers


layer_ref = LayerReference()
cell_type_ref = CellTypeReference()

__all__ = [k for k in vars().keys() if k.endswith("_ref") or k.endswith("__")]
