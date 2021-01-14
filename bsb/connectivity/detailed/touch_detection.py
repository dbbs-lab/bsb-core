import numpy as np
from ..strategy import ConnectionStrategy
from .shared import MorphologyStrategy
from ...helpers import (
    DistributionConfiguration,
    assert_attr_in,
)
from ...reporting import report, warn
from random import sample as sample_elements


class TouchInformation:
    def __init__(
        self, from_cell_type, from_cell_compartments, to_cell_type, to_cell_compartments
    ):
        self.from_cell_type = from_cell_type
        self.from_cell_compartments = from_cell_compartments
        self.to_cell_type = to_cell_type
        self.to_cell_compartments = to_cell_compartments


class TouchDetector(ConnectionStrategy, MorphologyStrategy):
    """
    Connectivity based on intersection of detailed morphologies
    """

    casts = {
        "compartment_intersection_radius": float,
        "cell_intersection_radius": float,
        "synapses": DistributionConfiguration.cast,
        "allow_zero_synapses": bool,
    }

    defaults = {
        "cell_intersection_plane": "xyz",
        "compartment_intersection_plane": "xyz",
        "compartment_intersection_radius": 5.0,
        "synapses": DistributionConfiguration.cast(1),
        "allow_zero_synapses": False,
    }

    required = [
        "cell_intersection_plane",
        "compartment_intersection_plane",
        "compartment_intersection_radius",
    ]

    def validate(self):
        planes = ["xyz", "xy", "xz", "yz", "x", "y", "z"]
        assert_attr_in(
            self.__dict__,
            "cell_intersection_plane",
            planes,
            "connection_types.{}".format(self.name),
        )
        assert_attr_in(
            self.__dict__,
            "compartment_intersection_plane",
            planes,
            "connection_types.{}".format(self.name),
        )

    def connect(self):
        # Create a dictionary to cache loaded morphologies.
        self.morphology_cache = {}

        for from_cell_type_index in range(len(self.from_cell_types)):
            from_cell_type = self.from_cell_types[from_cell_type_index]
            from_cell_compartments = self.from_cell_compartments[from_cell_type_index]
            for to_cell_type_index in range(len(self.to_cell_types)):
                to_cell_type = self.to_cell_types[to_cell_type_index]
                to_cell_compartments = self.to_cell_compartments[to_cell_type_index]
                touch_info = TouchInformation(
                    from_cell_type,
                    from_cell_compartments,
                    to_cell_type,
                    to_cell_compartments,
                )
                touch_info.from_placement = self.scaffold.get_placement_set(
                    from_cell_type
                )
                touch_info.from_positions = list(touch_info.from_placement.positions)
                touch_info.from_identifiers = list(touch_info.from_placement.identifiers)
                touch_info.to_placement = self.scaffold.get_placement_set(to_cell_type)
                touch_info.to_identifiers = list(touch_info.to_placement.identifiers)
                touch_info.to_positions = list(touch_info.to_placement.positions)
                # Intersect cells on the widest possible search radius.
                candidates = self.intersect_cells(touch_info)
                # Intersect cell compartments between matched cells.
                connections, morphology_names, compartments = self.intersect_compartments(
                    touch_info, candidates
                )
                # Connect the cells and store the morphologies and selected compartments that connect them.
                self.scaffold.connect_cells(
                    self,
                    connections,
                    morphologies=morphology_names,
                    compartments=compartments,
                )
        # Remove the morphology cache
        self.morphology_cache = None

    def intersect_cells(self, touch_info):
        from_cell_type = touch_info.from_cell_type
        to_cell_type = touch_info.to_cell_type
        cell_plane = self.cell_intersection_plane
        from_cell_tree = self.scaffold.trees.cells.get_planar_tree(
            from_cell_type.name, plane=cell_plane
        )
        to_cell_tree = self.scaffold.trees.cells.get_planar_tree(
            to_cell_type.name, plane=cell_plane
        )
        from_count = self.scaffold.get_placed_count(from_cell_type.name)
        to_count = self.scaffold.get_placed_count(to_cell_type.name)
        if hasattr(self, "cell_intersection_radius"):
            radius = self.cell_intersection_radius
        else:
            radius = self.get_search_radius(from_cell_type) + self.get_search_radius(
                to_cell_type
            )
        # TODO: Profile whether the reverse lookup with the smaller tree and then reversing the matches array
        # gains us any speed.
        if from_count < to_count:
            return to_cell_tree.query_radius(from_cell_tree.get_arrays()[0], radius)
        else:
            reversed_matches = from_cell_tree.query_radius(
                to_cell_tree.get_arrays()[0], radius
            )
            matches = [[] for _ in range(len(from_cell_tree.get_arrays()[0]))]
            for i in range(len(reversed_matches)):
                for match in reversed_matches[i]:
                    matches[match].append(i)
            return matches

    def intersect_compartments(self, touch_info, candidate_map):
        connected_cells = []
        morphology_names = []
        connected_compartments = []
        c_check = 0
        touching_cells = 0
        plots = 0
        for i in range(len(candidate_map)):
            from_id = touch_info.from_identifiers[i]
            touch_info.from_morphology = self.get_random_morphology(
                touch_info.from_cell_type
            )
            for j in candidate_map[i]:
                c_check += 1
                to_id = touch_info.to_identifiers[j]
                touch_info.to_morphology = self.get_random_morphology(
                    touch_info.to_cell_type
                )
                intersections = self.get_compartment_intersections(
                    touch_info, touch_info.from_positions[i], touch_info.to_positions[j]
                )
                if len(intersections) > 0:
                    touching_cells += 1
                    number_of_synapses = max(
                        min(int(self.synapses.sample()), len(intersections)),
                        int(not self.allow_zero_synapses),
                    )
                    cell_connections = [
                        [from_id, to_id] for _ in range(number_of_synapses)
                    ]
                    compartment_connections = sample_elements(
                        intersections, k=number_of_synapses
                    )
                    connected_cells.extend(cell_connections)
                    connected_compartments.extend(compartment_connections)
                    # Pad the morphology names with the right names for the amount of compartment connections made
                    morphology_names.extend(
                        [
                            [
                                touch_info.from_morphology.morphology_name,
                                touch_info.to_morphology.morphology_name,
                            ]
                            for _ in range(len(compartment_connections))
                        ]
                    )
        report(
            "Checked {} candidate cell pairs from {} to {}".format(
                c_check, touch_info.from_cell_type.name, touch_info.to_cell_type.name
            ),
            level=2,
        )
        report(
            "Touch connection results: \n* Touching pairs: {} \n* Synapses: {}".format(
                touching_cells, len(connected_compartments)
            ),
            level=2,
        )

        return (
            np.array(connected_cells, dtype=int),
            np.array(morphology_names, dtype=np.string_),
            np.array(connected_compartments, dtype=int),
        )

    def get_compartment_intersections(self, touch_info, from_pos, to_pos):
        from_morpho = touch_info.from_morphology
        to_morpho = touch_info.to_morphology
        query_points = (
            to_morpho.get_compartment_positions(touch_info.to_cell_compartments)
            + to_pos
            - from_pos
        )
        from_tree = from_morpho.get_compartment_tree(touch_info.from_cell_compartments)
        compartment_hits = from_tree.query_radius(
            query_points, self.compartment_intersection_radius
        )
        from_map = from_morpho.get_compartment_submask(touch_info.from_cell_compartments)
        to_map = to_morpho.get_compartment_submask(touch_info.to_cell_compartments)
        intersections = []
        for i in range(len(compartment_hits)):
            hits = compartment_hits[i]
            if len(hits) > 0:
                for j in range(len(hits)):
                    intersections.append([from_map[hits[j]], to_map[i]])
        return intersections

    def get_search_radius(self, cell_type):
        morphologies = self.get_all_morphologies(cell_type)
        max_radius = 0.0
        for morphology in morphologies:
            max_radius = max(
                max_radius,
                np.max(
                    np.sqrt(
                        np.sum(
                            np.power(morphology.compartment_tree.get_arrays()[0], 2),
                            axis=1,
                        )
                    )
                ),
            )
        return max_radius
