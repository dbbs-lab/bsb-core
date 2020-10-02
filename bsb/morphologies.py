import abc, numpy as np, pickle, h5py, math, itertools
from .helpers import ConfigurableClass
from .voxels import VoxelCloud, detect_box_compartments, Box
from sklearn.neighbors import KDTree
from .exceptions import *
from .reporting import report


class Compartment:
    """
        Compartments are line segments with a radius. They are the building block of
        :class:`Morphologies <.morphologies.Morphology>`.
    """

    def __init__(
        self,
        start,
        end,
        radius,
        id=None,
        type=None,
        parent=None,
        parent_id=None,
        section_id=None,
        morphology=None,
    ):
        self.id = id
        self.start = start
        self.end = end
        self.radius = radius
        self.type = type
        self.parent_id = parent_id
        self.parent = parent
        self.section_id = section_id
        self.morphology = morphology

    @classmethod
    def from_record(cls, morphology, repo_record):
        """
            Create a compartment from repository data.
        """
        # Transfer the record data onto a Compartment object.
        c = cls(
            id=repo_record[0],
            type=repo_record[1],
            start=np.array(repo_record[2:5]),
            end=np.array(repo_record[5:8]),
            radius=repo_record[8],
            parent_id=repo_record[9],
            morphology=morphology,
        )
        # Check if there's section id information on the record.
        if len(repo_record) == 11:
            c.section_id = int(repo_record[10])
        return c

    @property
    def midpoint(self):
        # Calculate midpoint of the compartment
        if not hasattr(self, "_midpoint"):
            self._midpoint = (self.end - self.start) / 2 + self.start
        return self._midpoint

    @property
    def spherical(self):
        # Calculate the radius of the outer sphere of this compartment
        if not hasattr(self, "_spherical"):
            self._spherical = np.sqrt((c.start[:] - c.end[:]) ** 2) / 2
        return self._spherical

    @classmethod
    def from_template(cls, template, **kwargs):
        """
            Create a compartment based on  a template compartment. Accepts any keyword
            argument to overwrite or add attributes.
        """
        c = cls(
            id=template.id,
            start=template.start,
            end=template.end,
            radius=template.radius,
            type=template.type,
            parent=template.parent,
            parent_id=template.parent_id,
            section_id=template.section_id,
            morphology=template.morphology,
        )
        for k, v in kwargs.items():
            c.__dict__[k] = v
        return c

    def to_record(self):
        """
            Return an array that can be used to store this compartment in an HDF5 dataset,
            or to construct a new Compartment.
        """
        record = [self.id, self.type, *self.start, *self.end, self.radius, self.parent_id]
        if hasattr(self, "section_id"):
            record.append(self.section_id)
        return record


def branch_iter(branch):
    """
        Iterate over a branch and all of its children depth first.
    """
    yield branch
    for child in branch._children:
        yield from branch_iter(child)


def _validate_branch_args(args):
    vec = Branch.vectors
    if len(args) > len(vec):
        raise TypeError(
            f"__init__ takes {len(vec) + 1} arguments but {len(args) + 1} given."
        )
    if len(args) < len(vec):
        n = len(vec) - len(args)
        w = "argument" if n == 1 else "arguments"
        missing = ", ".join(map("'{}'".format, vec[-n:]))
        raise TypeError(f"__init__ is missing {n} required {w}: {missing}.")


class Branch:
    vectors = ["x", "y", "z", "radii"]

    def __init__(self, *args):
        _validate_branch_args(args)
        self._children = []
        self._parent = None
        for v, vector in enumerate(self.__class__.vectors):
            self.__dict__[vector] = args[v]

    def attach_child(self, branch):
        self._children.append(branch)
        branch._parent = self

    def remove_child(self, branch):
        self._children.remove(branch)
        branch._parent = None

    def to_compartments(self, start_id=0, last_parent=None):
        comp_id = start_id

        def to_comp(data):
            nonlocal comp_id, last_parent
            comp = Compartment(comp_id, *data, parent=last_parent)
            comp_id += 1
            last_parent = comp
            return comp

        compartments = [to_comp(data) for data in self.walk()]

    def walk(self):
        return zip(*(self.__dict__[v] for v in self.__class__.vectors))


class Morphology:
    """
        A multicompartmental spatial representation of a cell based on connected 3D
        compartments.

        :todo: Uncouple from the MorphologyRepository and merge with TrueMorphology.
    """

    def __init__(self, scaffold, roots):
        self.scaffold = scaffold
        self.cloud = None
        self.has_morphology = True
        self.has_voxels = False
        self.roots = roots
        self._compartments = None

    @property
    def compartments(self):
        if self._compartments is None:
            self._compartments = self.to_compartments()
        return self._compartments

    @property
    def branches(self):
        """
            Return a depth-first flattened array of all branches.
        """
        return [*itertools.chain(*(branch_iter(root) for root in self.roots))]

    def to_compartments(self):
        """
            Return a flattened array of compartments
        """
        comp_counter = 0

        def treat_branch(branch, last_parent=None):
            comps = branch.to_compartments(comp_counter, last_parent)
            comp_counter += len(comps)
            parent_comp = comps[-1]
            child_iters = (treat_branch(b, parent_comp) for b in self._children)
            return itertools.chain(comps, *child_iters)

        return [*itertools.chain(treat_branch(root) for root in self.roots)]

    def flatten(self, vectors=None, matrix=False):
        """
            Return the flattened vectors of the morphology

            :param vectors: List of vectors to return such as ['x', 'y', 'z'] to get the
              positional vectors.
            :type vectors: list of str
            :returns: Tuple of the vectors in the given order, if `matrix` is True a
              matrix composed of the vectors is returned instead.
            :rtype: tuple of ndarrays (`matrix=False`) or matrix (`matrix=True`)
        """
        if vectors is None:
            vectors = Branch.vectors
        branches = self.branches
        if not branches:
            if matrix:
                return np.empty((0, len(vectors)))
            return tuple(np.empty(0) for _ in vectors)
        t = tuple(np.concatenate(*(getattr(b, v) for b in branches)) for v in vectors)
        return np.hstack(t) if matrix else t

    def init_morphology(self, repo_data, repo_meta):
        """
            Initialize this Morphology with detailed morphology data from a MorphologyRepository.
        """
        # Initialise as a true morphology
        self.compartments = []
        self.morphology_name = repo_meta["name"]
        self.has_morphology = True
        # Iterate over the data to create compartment objects
        for i in range(len(repo_data)):
            repo_record = repo_data[i, :]
            compartment = Compartment.from_record(self, repo_record)
            self.compartments.append(compartment)
        # Fortify the id-linked compartments' bond by referencing their parent object.
        for c in self.compartments:
            if c.parent_id is not None and c.parent_id != -1:
                c.parent = self.compartments[int(c.parent_id)]
            else:
                c.parent = None
        # Create a tree from the compartment object list
        self.update_compartment_tree()
        if (
            hasattr(self, "scaffold") and self.scaffold
        ):  # Is the scaffold ready at this point?
            self.store_compartment_tree()

    def store_compartment_tree(self):
        morphology_trees = self.scaffold.trees.morphologies
        morphology_trees.add_tree(self.morphology_name, self.compartment_tree)
        morphology_trees.save()

    def update_compartment_tree(self):
        self.compartment_tree = KDTree(
            np.array(list(map(lambda c: c.end, self.compartments)))
        )

    def init_voxel_cloud(self, voxel_data, voxel_meta, voxel_map):
        """
            Initialize this Morphology with a voxel cloud from a MorphologyRepository.
        """
        bounds = voxel_meta["bounds"]
        grid_size = voxel_meta["grid_size"]
        # Initialise as a true morphology
        self.cloud = VoxelCloud(bounds, voxel_data, grid_size, voxel_map)

    def voxelize(self, N, compartments=None):
        self.cloud = VoxelCloud.create(self, N, compartments=compartments)

    def create_compartment_map(self, tree, boxes, voxels, box_size):
        compartment_map = []
        box_positions = np.column_stack(boxes[:, voxels])
        for i in range(box_positions.shape[0]):
            box_origin = box_positions[i, :]
            compartment_map.append(detect_box_compartments(tree, box_origin, box_size))
        return compartment_map

    def get_bounding_box(self, compartments=None, centered=True):
        # Use the compartment tree to get a quick array of the compartments positions
        compartment_positions = np.array(
            list(map(lambda c: c.midpoint, compartments or self.compartments))
        )
        # Determine the amount of dimensions of the morphology. Let's hope 3 ;)
        n_dimensions = range(compartment_positions.shape[1])
        # Create a bounding box
        outer_box = Box()
        # The outer box dimensions are equal to the maximum distance between compartments in each of n dimensions
        outer_box.dimensions = np.array(
            [
                np.max(compartment_positions[:, i]) - np.min(compartment_positions[:, i])
                for i in n_dimensions
            ]
        )
        # The outer box origin should be in the middle of the outer bounds if 'centered' is True. (So lowermost point + sometimes half of dimensions)
        outer_box.origin = np.array(
            [
                np.min(compartment_positions[:, i])
                + (outer_box.dimensions[i] / 2) * int(centered)
                for i in n_dimensions
            ]
        )
        return outer_box

    def get_search_radius(self, plane="xyz"):
        pos = np.array(self.compartment_tree.get_arrays()[0])
        dimensions = ["x", "y", "z"]
        try:
            max_dists = np.max(
                np.abs(np.array([pos[:, dimensions.index(d)] for d in plane])), axis=1
            )
        except ValueError as e:
            raise ValueError("Unknown dimensions in dimension string '{}'".format(plane))
        return np.sqrt(np.sum(max_dists ** 2))

    def get_compartment_network(self):
        compartments = self.compartments
        node_list = [set([]) for c in compartments]
        # Add child nodes to their parent's adjacency set
        for node in compartments[1:]:
            if int(node.parent_id) == -1:
                continue
            node_list[int(node.parent_id)].add(int(node.id))
        return node_list

    def get_compartment_positions(self, types=None):
        if types is None:
            return self.compartment_tree.get_arrays()[0]
        type_ids = TrueMorphology.get_compartment_type_ids(types)
        return list(
            map(lambda c: c.end, filter(lambda c: c.type in type_ids, self.compartments))
        )

    def get_plot_range(self, offset=[0.0, 0.0, 0.0]):
        compartments = self.compartment_tree.get_arrays()[0]
        n_dimensions = range(compartments.shape[1])
        mins = np.array([np.min(compartments[:, i]) + offset[i] for i in n_dimensions])
        max = np.max(
            np.array(
                [np.max(compartments[:, i]) - mins[i] + offset[i] for i in n_dimensions]
            )
        )
        return list(zip(mins.tolist(), (mins + max).tolist()))

    def _comp_tree_factory(self, types):
        type_map = TrueMorphology.get_compartment_type_ids(types)

        def _comp_tree_product(_):
            return np.array(
                list(
                    map(
                        lambda c: c.end,
                        filter(lambda c: c.type in type_map, self.compartments),
                    )
                )
            )

        return _comp_tree_product

    def get_compartment_tree(self, compartment_types=None):
        if compartment_types is not None:
            if len(compartment_types) == 1:
                return self.scaffold.trees.morphologies.get_sub_tree(
                    self.morphology_name,
                    "+".join(compartment_types),
                    factory=self._comp_tree_factory(compartment_types),
                )
            else:
                raise NotImplementedError(
                    "Multicompartmental touch detection not implemented yet."
                )
        return self.compartment_tree

    def get_compartment_submask(self, compartment_types):
        i = 0
        type_ids = TrueMorphology.get_compartment_type_ids(compartment_types)
        mask = []
        for comp in self.compartments:
            if comp.type in type_ids:
                # mask[n] = original id
                # Where n is the index of the compartment in the filtered collection
                mask.append(comp.id)
        return mask

    def get_compartments(self, compartment_types=None):
        if compartment_types is None:
            return self.compartments.copy()
        i = 0
        try:
            type_ids = TrueMorphology.get_compartment_type_ids(compartment_types)
        except Exception as e:
            raise CompartmentError("Unknown compartment types encountered")
        return list(filter(lambda c: c.type in type_ids, self.compartments))

    @classmethod
    def get_compartment_type_ids(cls, types):
        ids = []
        for t in types:
            ids.append(cls.compartment_types[t])
            if t in cls.compartment_alias:
                ids.extend(cls.get_compartment_type_ids(cls.compartment_alias[t]))
        return ids

    def rotate(self, v0, v):
        """

            Rotate a morphology to be oriented as vector v, supposing to start from orientation v0.
            norm(v) = norm(v0) = 1
            Rotation matrix R, representing a rotation of angle alpha around vector k

        """
        R = get_rotation_matrix(v0, v)

        for c in range(len(self.compartments)):
            self.compartments[c].start = R.dot(self.compartments[c].start)
            self.compartments[c].end = R.dot(self.compartments[c].end)

        self.update_compartment_tree()


class GranuleCellGeometry(Morphology):
    casts = {
        "dendrite_length": float,
        "pf_height": float,
        "pf_height_sd": float,
    }
    required = ["dendrite_length", "pf_height", "pf_height_sd"]

    def validate(self):
        pass


class PurkinjeCellGeometry(Morphology):
    def validate(self):
        pass


class GolgiCellGeometry(Morphology):
    casts = {
        "dendrite_radius": float,
        "axon_x": float,
        "axon_y": float,
        "axon_z": float,
    }

    required = ["dendrite_radius"]

    def validate(self):
        pass


class RadialGeometry(Morphology):
    casts = {
        "dendrite_radius": float,
    }

    required = ["dendrite_radius"]

    def validate(self):
        pass


class NoGeometry(Morphology):
    def validate(self):
        pass


def get_rotation_matrix(v0, v):
    I = np.identity(3)
    # Reduce 1-size dimensions
    v0 = np.array(v0).squeeze()
    v = np.array(v).squeeze()
    # Normalize orientation vectors
    v0 = v0 / np.linalg.norm(v0)
    v = v / np.linalg.norm(v0)
    alpha = np.arccos(np.dot(v0, v))

    if math.isclose(alpha, 0.0, rel_tol=1e-4):
        report(
            "Rotating morphology between parallel orientation vectors, {} and {}!".format(
                v0, v
            ),
            level=3,
        )
        # We will not rotate the morphology, thus R = I
        return I
    elif math.isclose(alpha, np.pi, rel_tol=1e-4):
        report(
            "Rotating morphology between antiparallel orientation vectors, {} and {}!".format(
                v0, v
            ),
            level=3,
        )
        # We will rotate the morphology of 180° around a vector orthogonal to
        # the starting vector v0 (the same would be if we take the ending vector
        # v). We set the first and third components to 1; the second one is
        # obtained to have the scalar product with v0 equal to 0
        kx = 1
        kz = 1
        ky = -(v0[0] + v0[2]) / v0[1]
        k = np.array([kx, ky, kz])
    else:
        k = (np.cross(v0, v)) / math.sin(alpha)
        k = k / np.linalg.norm(k)

    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    # Compute and return the rotation matrix using Rodrigues' formula
    return I + math.sin(alpha) * K + (1 - math.cos(alpha)) * np.linalg.matrix_power(K, 2)
