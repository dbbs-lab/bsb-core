import numpy as np
from .morphologies import Morphology as BaseMorphology
from .helpers import ConfigurableClass, dimensions, origin

class CellType:

    def __init__(self, name, placement=None):
        self.name = name
        self.placement = placement

    def validate(self):
        '''
            Check whether this CellType is valid to be used in the simulation.
        '''
        pass

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance
        self.id = scaffoldInstance.configuration.cell_type_map.index(self.name)
        self.validate()

    def set_morphology(self, morphology):
        '''
            Set the Morphology class for this cell type.

            :param morphology: Defines the geometrical constraints for the axon and dendrites of the cell type.
            :type morphology: Instance of a subclass of scaffold.morphologies.Morphology
        '''
        if not issubclass(type(morphology), BaseMorphology):
            raise Exception("Only subclasses of scaffold.morphologies.Morphology can be used as cell morphologies.")
        self.morphology = morphology

    def set_placement(self, placement):
        '''
            Set the placement strategy for this cell type.
        '''
        self.placement = placement

    def is_after_satisfied(self, cell_types):
        '''
            Determine whether the `after` specification of this cell type is met.
            Any cell types appearing in `self.after` need to occur before this cell type,
            so that this cell type appears "after" all these cell types.
        '''
        if not hasattr(self.placement, "after"): # No after?
            # Condition without constraints always True.
            return True
        is_met = False
        # Determine whether this cell type is out of order.
        for type in cell_types:
            if is_met and type.name in self.placement.after:
                # After conditions not met if we have we seen ourself and
                # find something that's supposed to be in front of us.
                return False
            elif type == self: # Is this us?
                # From this point on, nothing that appears in the after array is allowed to be encountered
                is_met = True
        # We didn't meet anything behind us that was supposed to be in front of us
        # => Condition met.
        return True

    def satisfy_after(self, cell_types):
        '''
            Given an array of cell types, place this cell type after all of the cell types
            specified in `self.after`. If cell types in `self.after` are missing from the given array
            this cell type is placed at the end of the array.
        '''
        before_types = self.placement.after.copy()
        i = 0
        place_after = False
        while len(before_types) > 0 and i < len(cell_types):
            if cell_types[i].name in before_types:
                before_types.remove(cell_types[i].name)
            if cell_types[i] == self:
                cell_types.remove(self)
                place_after = True
            else:
                i += 1
        if place_after:
            cell_types.insert(i, self)

    @staticmethod
    def resolve_placement_order(cell_types):
        '''
            Sort the given array based on placement count, and rearrange them so that all cell types
            appear after any other cell types specified in `cell_type.after`
        '''
        # The default approach is to sort on ascending placement count.
        sorting_cell_types = sorted(cell_types.values(), key=lambda x: x.placement.get_placement_count(x))
        # Afterwards cell types can be specified that need to be placed after other types.
        after_specifications = list(filter(lambda c: hasattr(c.placement, "after"), cell_types.values()))
        j = 0
        # Keep rearranging as long as any cell type's after condition isn't satisfied.
        while any(map(lambda c: not c.is_after_satisfied(sorting_cell_types), after_specifications)):
            j += 1
            # Rearrange each element that is out of place.
            for after_type in after_specifications:
                if not after_type.is_after_satisfied(sorting_cell_types):
                    after_type.satisfy_after(sorting_cell_types)
            # If we have had to rearrange all elements more than there are elements, the
            # conditions cannot be met, and a circular dependency is at play.
            if j > len(cell_types):
                raise Exception("Couldn't resolve placement order, probably a circular dependency including: {}".format(
                    ", ".join(list(map(lambda c: c.name, filter(lambda c: not c.is_after_satisfied(sorting_cell_types), after_specifications))))))
        # Return the sorted array.
        return sorting_cell_types

class Layer(dimensions, origin):

    def __init__(self, name, origin, dimensions, scaling=True):
        # Name of the layer
        self.name = name
        # The XYZ coordinates of the point at the center of the bottom plane of the layer.
        self.origin = np.array(origin)
        # Dimensions in the XYZ axes.
        self.dimensions = np.array(dimensions)
        self.volumeOccupied = 0.
        # Should this layer scale when the simulation volume is resized?
        self.scaling = scaling

    @property
    def available_volume(self):
        return self.volume - self.volumeOccupied

    @property
    def thickness(self):
        return self.dimensions[1]

    def allocateVolume(volume):
        self.volumeOccupied += volume

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance
