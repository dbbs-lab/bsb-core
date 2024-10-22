from __future__ import annotations

import abc
from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from ... import config
from ...config import types


def _reshape_vectors(rot_pts, x, y, z):
    xrot = rot_pts[:, 0].reshape(x.shape)
    yrot = rot_pts[:, 1].reshape(y.shape)
    zrot = rot_pts[:, 2].reshape(z.shape)

    return xrot, yrot, zrot


def rotate_3d_mesh_by_vec(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rot_versor: np.ndarray, angle: float
):
    """
    Rotate meshgrid points according to a rotation versor and angle.

    :param numpy.ndarray[numpy.ndarray[float]] x: x coordinate points of the meshgrid
    :param numpy.ndarray[numpy.ndarray[float]] y: y coordinate points of the meshgrid
    :param numpy.ndarray[numpy.ndarray[float]] z: z coordinate points of the meshgrid
    :param numpy.ndarray[float] rot_versor: vector representing rotation versor
    :param float angle: rotation angle in radian
    :return: Rotated x, y, z coordinate points
    :rtype: Tuple[numpy.ndarray[numpy.ndarray[float]]
    """

    # Arrange point coordinates in shape (N, 3) for vectorized processing
    pts = np.array([x.ravel(), y.ravel(), z.ravel()]).transpose()

    # Create and apply rotation
    rot = R.from_rotvec(rot_versor * angle)
    rot_pts = rot.apply(pts)

    # return to original shape of meshgrid
    return _reshape_vectors(rot_pts, x, y, z)


def translate_3d_mesh_by_vec(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, t_vec: np.ndarray
):
    """
    Translate meshgrid points according to a 3d vector.

    :param numpy.ndarray[numpy.ndarray[float]] x: x coordinate points of the meshgrid
    :param numpy.ndarray[numpy.ndarray[float]] y: y coordinate points of the meshgrid
    :param numpy.ndarray[numpy.ndarray[float]] z: z coordinate points of the meshgrid
    :param numpy.ndarray[float] t_vec: translation vector
    :return: Translated x, y, z coordinate points
    :rtype: Tuple[numpy.ndarray[numpy.ndarray[float]]
    """

    # Arrange point coordinates in shape (N, 3) for vectorized processing
    pts = np.array([x.ravel(), y.ravel(), z.ravel()]).transpose()
    pts = pts + t_vec
    # return to original shape of meshgrid
    return _reshape_vectors(pts, x, y, z)


def rotate_3d_mesh_by_rot_mat(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rot_mat: np.ndarray
):
    """
    Rotate meshgrid points according to a rotation matrix.

    :param numpy.ndarray[numpy.ndarray[float]] x: x coordinate points of the meshgrid
    :param numpy.ndarray[numpy.ndarray[float]] y: y coordinate points of the meshgrid
    :param numpy.ndarray[numpy.ndarray[float]] z: z coordinate points of the meshgrid
    :param numpy.ndarray[numpy.ndarray[float]] rot_mat: rotation matrix, shape (3,3)
    :return: Rotated x, y, z coordinate points
    :rtype: Tuple[numpy.ndarray[numpy.ndarray[float]]
    """

    # Arrange point coordinates in shape (N, 3) for vectorized processing
    pts = np.array([x.ravel(), y.ravel(), z.ravel()]).transpose()

    # Create and apply rotation
    rot = R.from_matrix(rot_mat)
    rot_pts = rot.apply(pts)

    # return to original shape of meshgrid
    return _reshape_vectors(rot_pts, x, y, z)


def _surface_resampling(
    surface_function,
    theta_min=0,
    theta_max=2 * np.pi,
    phi_min=0,
    phi_max=np.pi,
    precision=25,
):
    # first sampling to estimate surface distribution
    theta, phi = np.meshgrid(
        np.linspace(theta_min, theta_max, precision),
        np.linspace(phi_min, phi_max, precision),
    )
    coords = surface_function(theta, phi)

    # estimate surfaces, decomposing it into parallelograms along theta and phi
    delta_t_temp = np.diff(coords, axis=2)
    delta_u_temp = np.diff(coords, axis=1)
    delta_t = np.zeros(coords.shape)
    delta_u = np.zeros(coords.shape)
    delta_t[: coords.shape[0], : coords.shape[1], 1 : coords.shape[2]] = delta_t_temp
    delta_u[: coords.shape[0], 1 : coords.shape[1], : coords.shape[2]] = delta_u_temp
    delta_S = np.linalg.norm(np.cross(delta_t, delta_u, 0, 0), axis=2)
    cum_S_t = np.cumsum(delta_S.sum(axis=0))
    cum_S_u = np.cumsum(delta_S.sum(axis=1))
    return theta, phi, cum_S_t, cum_S_u


def uniform_surface_sampling(
    n_points,
    surface_function,
    theta_min=0,
    theta_max=2 * np.pi,
    phi_min=0,
    phi_max=np.pi,
    precision=25,
):
    """
    Uniform-like random sampling of polar coordinates based on surface estimation.
    This sampling is useful on elliptic surfaces (e.g. sphere).
    Algorithm based on https://github.com/maxkapur/param_tools

    :param int n_points: number of points to sample
    :param Callable[..., numpy.ndarray[float]] surface_function: function converting polar
        coordinates into cartesian coordinates
    :param int precision: size of grid used to estimate function surface
    """

    theta, phi, cum_S_t, cum_S_u = _surface_resampling(
        surface_function, theta_min, theta_max, phi_min, phi_max, precision
    )
    # resample along the cumulative surface to uniformize point distribution
    # equivalent to a multinomial sampling
    sampled_t = np.random.rand(n_points) * cum_S_t[-1]
    sampled_u = np.random.rand(n_points) * cum_S_u[-1]
    sampled_t = interp1d(cum_S_t, theta[0, :])(sampled_t)
    sampled_u = interp1d(cum_S_u, phi[:, 0])(sampled_u)

    return surface_function(sampled_t, sampled_u)


def uniform_surface_wireframe(
    n_points_1,
    n_points_2,
    surface_function,
    theta_min=0,
    theta_max=2 * np.pi,
    phi_min=0,
    phi_max=np.pi,
    precision=25,
):
    """
    Uniform-like meshgrid of size (n_point_1, n_points_2) of polar coordinates based on surface
    estimation.
    This meshgrid is useful on elliptic surfaces (e.g. sphere).
    Algorithm based on https://github.com/maxkapur/param_tools

    :param Callable[..., numpy.ndarray[float]] surface_function: function converting polar
        coordinates into cartesian coordinates
    :param int precision: size of grid used to estimate function surface
    """

    theta, phi, cum_S_t, cum_S_u = _surface_resampling(
        surface_function, theta_min, theta_max, phi_min, phi_max, precision
    )
    sampled_t = np.linspace(0, cum_S_t[-1], n_points_1)
    sampled_u = np.linspace(0, cum_S_u[-1], n_points_2)
    sampled_t = interp1d(cum_S_t, theta[0, :])(sampled_t)
    sampled_u = interp1d(cum_S_u, phi[:, 0])(sampled_u)
    sampled_t, sampled_u = np.meshgrid(sampled_t, sampled_u)
    return surface_function(sampled_t, sampled_u)


def _get_prod_angle_vector(hv, z_versor=np.array([0, 0, 1])):
    """
    Calculate the cross product and the arc cosines angle between two vectors.

    :param numpy.ndarray hv: vector to rotate
    :param numpy.ndarray z_versor: reference vector
    :return: cross product and arc cosines angle
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """

    return np.cross(z_versor, hv), np.arccos(np.dot(hv, z_versor))


def _get_rotation_vector(hv, z_versor=np.array([0, 0, 1]), positive_angle=True):
    """
    Calculate the rotation vector between two vectors.

    :param numpy.ndarray hv: vector to rotate
    :param numpy.ndarray z_versor: reference vector
    :param bool positive_angle: if False, the angle is inverted
    :return: rotation vector
    :rtype: scipy.spatial.transform.Rotation
    """

    perp, angle = _get_prod_angle_vector(hv, z_versor)
    angle = angle if positive_angle else -angle
    rot = R.from_rotvec(perp * angle)
    return rot


def _rotate_by_coord(x, y, z, hv, origin, test_hv=False):
    perp, angle = _get_prod_angle_vector(hv / np.linalg.norm(hv))

    x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
    if test_hv and hv[2] < 0:
        z = -z
    return translate_3d_mesh_by_vec(x, y, z, origin)


def _get_extrema_after_rot(extrema, origin, top_center):
    height = np.linalg.norm(top_center - origin)
    rot = _get_rotation_vector((top_center - origin) / height)

    for i, pt in enumerate(extrema):
        extrema[i] = rot.apply(pt)

    return np.min(extrema, axis=0) + origin, np.max(extrema, axis=0) + origin


def inside_mbox(
    points: np.ndarray[float],
    mbb_min: np.ndarray[float],
    mbb_max: np.ndarray[float],
):
    """
    Check if the points given in input are inside the minimal bounding box.

    :param numpy.ndarray points: An array of 3D points.
    :param numpy.ndarray mbb_min: 3D point representing the lowest coordinate of the
        minimal bounding box.
    :param numpy.ndarray mbb_max: 3D point representing the highest coordinate of the
        minimal bounding box.
    :return: A bool np.ndarray specifying whether each point of the input array is inside the
        minimal bounding box or not.
    :rtype: numpy.ndarray[bool]
    """
    inside = (
        (points[:, 0] > mbb_min[0])
        & (points[:, 0] < mbb_max[0])
        & (points[:, 1] > mbb_min[1])
        & (points[:, 1] < mbb_max[1])
        & (points[:, 2] > mbb_min[2])
        & (points[:, 2] < mbb_max[2])
    )

    return inside


@config.dynamic(attr_name="type", default="shape", auto_classmap=True)
class GeometricShape(abc.ABC):
    """
    Base class for geometric shapes
    """

    epsilon = config.attr(type=float, required=False, default=1.0e-3)
    """Tolerance value to compare coordinates."""

    def __init__(self, **kwargs):
        self.mbb_min, self.mbb_max = self.find_mbb()

    @abc.abstractmethod
    def find_mbb(self):  # pragma: no cover
        """
        Compute the minimum bounding box surrounding the shape.
        """
        pass

    def check_mbox(self, points: np.ndarray[float]):
        """
        Check if the points given in input are inside the minimal bounding box.

        :param numpy.ndarray points: A cloud of points.
        :return: A bool np.ndarray specifying whether each point of the input array is inside the
            minimal bounding box or not.
        :rtype: numpy.ndarray
        """
        return inside_mbox(points, self.mbb_min, self.mbb_max)

    @abc.abstractmethod
    def get_volume(self):  # pragma: no cover
        """
        Get the volume of the geometric shape.
        :return: The volume of the geometric shape.
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def translate(self, t_vector: np.ndarray[float]):  # pragma: no cover
        """
        Translate the geometric shape by the vector t_vector.

        :param numpy.ndarray t_vector: The displacement vector
        """
        pass

    @abc.abstractmethod
    def rotate(self, r_versor: np.ndarray[float], angle: float):  # pragma: no cover
        """
        Rotate all the shapes around r_versor, which is a versor passing through the origin,
        by the specified angle.

        :param r_versor: A versor specifying the rotation axis.
        :type r_versor: numpy.ndarray[float]
        :param float angle: the rotation angle, in radians.
        """
        pass

    @abc.abstractmethod
    def generate_point_cloud(self, npoints: int):  # pragma: no cover
        """
        Generate a point cloud made by npoints points.

        :param int npoints: The number of points to generate.
        :return: a (npoints x 3) numpy array.
        :rtype: numpy.ndarray
        """
        pass

    @abc.abstractmethod
    def check_inside(
        self, points: np.ndarray[float]
    ) -> np.ndarray[bool]:  # pragma: no cover
        """
        Check if the points given in input are inside the geometric shape.

        :param numpy.ndarray points: A cloud of points.
        :return: A bool array with same length as points, containing whether the -ith point is
            inside the geometric shape or not.
        :rtype: numpy.ndarray
        """
        pass

    @abc.abstractmethod
    def wireframe_points(self, nb_points_1=30, nb_points_2=30):  # pragma: no cover
        """
        Generate a wireframe to plot the geometric shape.
        If a sampling of points is needed (e.g. for sphere), the wireframe is based on a grid
        of shape (nb_points_1, nb_points_2).

        :param int nb_points_1: number of points sampled along the first dimension
        :param int nb_points_2: number of points sampled along the second dimension
        :return: Coordinate components of the wireframe
        :rtype: Tuple[numpy.ndarray[numpy.ndarray[float]]
        """
        pass


@config.node
class ShapesComposition:
    """
    A collection of geometric shapes, which can be labelled to distinguish different parts of a
    neuron.
    """

    shapes = config.list(
        type=GeometricShape,
        required=types.same_size("shapes", "labels", required=True),
        hint=[{"type": "sphere", "radius": 40.0, "center": [0.0, 0.0, 0.0]}],
    )
    """List of GeometricShape that make up the neuron."""
    labels = config.list(
        type=types.list(),
        required=types.same_size("shapes", "labels", required=True),
        hint=[["soma", "dendrites", "axon"]],
    )
    """List of lists of labels associated to each geometric shape."""
    voxel_size = config.attr(type=float, required=False, default=1.0)
    """Dimension of the side of a voxel, used to determine how many points must be generated
    to represent the geometric shape."""

    def __init__(self, **kwargs):
        # The two corners individuating the minimal bounding box.
        self._mbb_min = np.array([0.0, 0.0, 0.0])
        self._mbb_max = np.array([0.0, 0.0, 0.0])

        self.find_mbb()

    def add_shape(self, shape: GeometricShape, labels: List[str]):
        """
        Add a geometric shape to the collection

        :param GeometricShape shape: A GeometricShape to add to the collection.
        :param List[str] labels: A list of labels for the geometric shape to add.
        """
        # Update mbb
        if len(self._shapes) == 0:
            self._mbb_min = np.copy(shape.mbb_min)
            self._mbb_max = np.copy(shape.mbb_max)
        else:
            self._mbb_min = np.minimum(self._mbb_min, shape.mbb_min)
            self._mbb_max = np.maximum(self._mbb_max, shape.mbb_max)
        self._shapes.append(shape)
        self._labels.append(labels)

    def filter_by_labels(self, labels: List[str]) -> ShapesComposition:
        """
        Filter the collection of shapes, returning only the ones corresponding the given labels.

        :param List[str] labels: A list of labels.
        :return: A new ShapesComposition object containing only the shapes labelled as specified.
        :rtype: ShapesComposition
        """
        result = ShapesComposition(dict(voxel_size=self.voxel_size, labels=[], shapes=[]))
        selected_id = np.where(np.isin(labels, self._labels))[0]
        result._shapes = [self._shapes[i].__copy__() for i in selected_id]
        result._labels = [self._labels[i].copy() for i in selected_id]
        result.mbb_min, result.mbb_max = result.find_mbb()
        return result

    def translate(self, t_vec: np.ndarray[float]):
        """
        Translate all the shapes in the collection by the vector t_vec. It also automatically
        translate the minimal bounding box.

        :param numpy.ndarray t_vec: The displacement vector.
        """
        for shape in self._shapes:
            shape.translate(t_vec)
        self._mbb_min += t_vec
        self._mbb_max += t_vec

    def get_volumes(self) -> List[float]:
        """
        Compute the volumes of all the shapes.

        :rtype: List[float]
        """
        return [shape.get_volume() for shape in self._shapes]

    def get_mbb_min(self):
        """
        Returns the bottom corner of the minimum bounding box containing the collection of shapes.

        :return: The bottom corner individuating the minimal bounding box of the shapes collection.
        :rtype: numpy.ndarray[float]
        """
        return self._mbb_min

    def get_mbb_max(self):
        """
        Returns the top corner of the minimum bounding box containing the collection of shapes.

        :return: The top corner individuating the minimal bounding box of the shapes collection.
        :rtype: numpy.ndarray[float]
        """
        return self._mbb_max

    def find_mbb(self) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Compute the minimal bounding box containing the collection of shapes.

        :return: The two corners individuating the minimal bounding box of the shapes collection.
        :rtype: Tuple(numpy.ndarray[float], numpy.ndarray[float])
        """
        mins = np.empty([len(self._shapes), 3])
        maxs = np.empty([len(self._shapes), 3])
        for i, shape in enumerate(self._shapes):
            mins[i, :] = shape.mbb_min
            maxs[i, :] = shape.mbb_max
        self._mbb_min = np.min(mins, axis=0) if len(self._shapes) > 0 else np.zeros(3)
        self._mbb_max = np.max(maxs, axis=0) if len(self._shapes) > 0 else np.zeros(3)
        return self._mbb_min, self._mbb_max

    def compute_n_points(self) -> List[int]:
        """
        Compute the number of points to generate in a point cloud, using the dimension of the voxel
        specified in self._voxel_size.

        :return: The number of points to generate.
        :rtype: numpy.ndarray[int]
        """
        return [int(shape.get_volume() // self.voxel_size**3) for shape in self._shapes]

    def generate_point_cloud(self) -> np.ndarray[float] | None:
        """
        Generate a point cloud. The number of points to generate is determined automatically using
        the voxel size.

        :return: A numpy.ndarray containing the 3D points of the cloud. If there are no shapes in
            the collection, it returns None.
        :rtype: numpy.ndarray[float] | None
        """
        if len(self._shapes) != 0:
            return np.concatenate(
                [
                    shape.generate_point_cloud(numpts)
                    for shape, numpts in zip(self._shapes, self.compute_n_points())
                ]
            )
        else:
            return None

    def generate_wireframe(
        self,
        nb_points_1=30,
        nb_points_2=30,
    ) -> Tuple[List, List, List] | None:
        """
        Generate the wireframes of a collection of shapes.
        If a sampling of points is needed for certain shapes (e.g. for sphere), their wireframe
        is based on a grid of shape (nb_points_1, nb_points_2).

        :param int nb_points_1: number of points sampled along the first dimension
        :param int nb_points_2: number of points sampled along the second dimension
        :return: The x,y,z coordinates of the wireframe of each shape.
        :rtype: Tuple[List[numpy.ndarray[numpy.ndarray[float]]]] | None
        """
        if len(self._shapes) != 0:
            x = []
            y = []
            z = []
            for shape in self._shapes:
                # For each shape, the shape of the wireframe is different, so we need to append them
                # manually
                xt, yt, zt = shape.wireframe_points(
                    nb_points_1=nb_points_1, nb_points_2=nb_points_2
                )
                x.append(xt)
                y.append(yt)
                z.append(zt)
            return x, y, z
        return None

    def inside_mbox(self, points: np.ndarray[float]) -> np.ndarray[bool]:
        """
        Check if the points given in input are inside the minimal bounding box of the collection.

        :param numpy.ndarray points: An array of 3D points.
        :return: A bool np.ndarray specifying whether each point of the input array is inside the
            minimal bounding box of the collection.
        :rtype: numpy.ndarray[bool]
        """
        return inside_mbox(points, self._mbb_min, self._mbb_max)

    def inside_shapes(self, points: np.ndarray[float]) -> np.ndarray[bool] | None:
        """
        Check if the points given in input are inside at least in one of the shapes of the
        collection.

        :param numpy.ndarray points: An array of 3D points.
        :return: A bool numpy.ndarray specifying whether each point of the input array is inside the
            collection of shapes or not.
        :rtype: numpy.ndarray[bool]
        """
        if len(self._shapes) != 0:
            is_inside = np.full(len(points), 0, dtype=bool)
            for shape in self._shapes:
                tmp = shape.check_mbox(points)
                if np.any(tmp):
                    is_inside = is_inside | shape.check_inside(points)
            return is_inside
        else:
            return None


@config.node
class Ellipsoid(GeometricShape, classmap_entry="ellipsoid"):
    """
    An ellipsoid, described in cartesian coordinates.
    """

    origin = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 0.0, 0.0]
    )
    """The coordinates of the center of the ellipsoid."""
    lambdas = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[1.0, 0.5, 2.0]
    )
    """The length of the three semi-axes."""

    @config.property(type=types.ndarray(shape=(3,)), required=True)
    def v0(self):
        """The versor on which the first semi-axis lies."""
        return self._v0

    @v0.setter
    def v0(self, value):
        self._v0 = np.copy(value) / np.linalg.norm(value)

    @config.property(type=types.ndarray(shape=(3,)), required=True)
    def v1(self):
        """The versor on which the second semi-axis lies."""
        return self._v1

    @v1.setter
    def v1(self, value):
        self._v1 = np.copy(value) / np.linalg.norm(value)

    @config.property(type=types.ndarray(shape=(3,)), required=True)
    def v2(self):
        """The versor on which the third semi-axis lies."""
        return self._v2

    @v2.setter
    def v2(self, value):
        self._v2 = np.copy(value) / np.linalg.norm(value)

    def find_mbb(self):
        # Find the minimum bounding box, to avoid computing it every time
        extrema = (
            np.array(
                [
                    self.lambdas[0] * self.v0,
                    -self.lambdas[0] * self.v0,
                    self.lambdas[1] * self.v1,
                    -self.lambdas[1] * self.v1,
                    self.lambdas[2] * self.v2,
                    -self.lambdas[2] * self.v2,
                ]
            )
            + self.origin
        )
        mbb_min = np.min(extrema, axis=0)
        mbb_max = np.max(extrema, axis=0)
        return mbb_min, mbb_max

    def get_volume(self):
        return np.pi * self.lambdas[0] * self.lambdas[1] * self.lambdas[2]

    def translate(self, t_vector: np.ndarray):
        self.origin += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: np.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.v0 = rot.apply(self.v0)
        self.v1 = rot.apply(self.v1)
        self.v2 = rot.apply(self.v2)

    def surface_point(self, theta, phi):
        """
        Convert polar coordinates into their 3D location on the ellipsoid surface.

        :param float|numpy.ndarray[float] theta: first polar coordinate in [0; 2*np.pi]
        :param float|numpy.ndarray[float] phi: second polar coordinate in [0; np.pi]
        :return: surface coordinates
        :rtype: float|numpy.ndarray[float]
        """
        return np.array(
            [
                self.lambdas[0] * np.cos(theta) * np.sin(phi),
                self.lambdas[1] * np.sin(theta) * np.sin(phi),
                self.lambdas[2] * np.cos(phi),
            ]
        )

    def generate_point_cloud(self, npoints: int):
        sampling = uniform_surface_sampling(npoints, self.surface_point)
        sampling = sampling.T * np.random.rand(npoints, 3)  # sample within the shape

        # Rotate the ellipse
        rmat = np.array([self.v0, self.v1, self.v2]).T
        sampling = sampling.dot(rmat)
        sampling = sampling + self.origin
        return sampling

    def check_inside(self, points: np.ndarray[float]):
        # Check if the quadratic form associated to the ellipse is less than 1 at a point
        diff = points - self.origin
        vmat = np.array([self.v0, self.v1, self.v2])
        diag = np.diag(1 / self.lambdas**2)
        qmat = vmat.dot(diag).dot(vmat)
        quad_prod = np.diagonal(diff.dot(qmat.dot(diff.T)))

        # Check if the points are inside the ellipsoid
        inside_points = quad_prod < 1
        return inside_points

    def wireframe_points(self, nb_points_1=30, nb_points_2=30):
        # Generate an ellipse orientated along x,y,z
        x, y, z = uniform_surface_wireframe(nb_points_1, nb_points_2, self.surface_point)
        # Rotate the ellipse
        rmat = np.array([self.v0, self.v1, self.v2]).T
        x, y, z = rotate_3d_mesh_by_rot_mat(x, y, z, rmat)
        return translate_3d_mesh_by_vec(x, y, z, self.origin)


@config.node
class Cone(GeometricShape, classmap_entry="cone"):
    """
    A cone, described in cartesian coordinates.
    """

    apex = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 1.0, 0.0]
    )
    """The coordinates of the apex of the cone."""
    origin = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 0.0, 0.0]
    )
    """The coordinates of the center of the cone's base."""
    radius = config.attr(type=float, required=False, default=1.0)
    """The radius of the base circle."""

    def find_mbb(self):
        # Vectors identifying half of the sides of the base rectangle in xy
        u = np.array([self.radius, 0, 0])
        v = np.array([0, self.radius, 0])

        # Find the rotation angle and axis
        hv = self.origin - self.apex
        rot = _get_rotation_vector(hv / np.linalg.norm(hv))

        # Rotated vectors of the box
        v1 = rot.apply(u)
        v2 = rot.apply(v)
        v3 = self.origin - self.apex

        # Coordinates identifying the minimal bounding box
        minima = np.min([v1, v2, v3, -v1, -v2], axis=0)
        maxima = np.max([v1, v2, v3, -v1, -v2], axis=0)
        return minima, maxima

    def get_volume(self):
        h = np.linalg.norm(self.apex - self.origin)
        b = np.pi * self.radius * self.radius
        return b * h / 3

    def translate(self, t_vector):
        self.origin += t_vector
        self.apex += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: np.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.apex = rot.apply(self.apex)

    def generate_point_cloud(self, npoints: int):
        theta = np.pi * 2.0 * np.random.rand(npoints)
        rand_a = np.random.rand(npoints)
        rand_b = np.random.rand(npoints)

        # Height vector
        hv = self.origin - self.apex
        cloud = np.full((npoints, 3), 0, dtype=float)

        # Generate a cone with the apex in the origin and the origin at (0,0,1)
        cloud[:, 0] = (self.radius * rand_a * np.cos(theta)) * rand_b
        cloud[:, 1] = self.radius * rand_a * np.sin(theta)
        cloud[:, 2] = rand_a * np.linalg.norm(hv)

        # Rotate the cone: Find the axis of rotation and compute the angle
        perp, angle = _get_prod_angle_vector(hv / np.linalg.norm(hv))

        if hv[2] < 0:
            cloud[:, 2] = -cloud[:, 2]
        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(cloud)

        # Translate the cone
        cloud = cloud + self.apex
        return cloud

    def check_inside(self, points: np.ndarray[float]):
        # Find the vector w of the height.
        h_vector = self.origin - self.apex
        height = np.linalg.norm(h_vector)
        h_vector /= height
        # Center the points
        pts = points - self.apex

        # Rotate back to xyz
        rot = _get_rotation_vector(h_vector)
        rot_pts = rot.apply(pts)

        # Find the angle between the points and the apex
        apex_angles = np.arccos(
            np.dot((rot_pts / np.linalg.norm(rot_pts, axis=1)[..., np.newaxis]), h_vector)
        )
        # Compute the cone angle
        cone_angle = np.arctan(self.radius / height)

        # Select the points inside the cone
        inside_points = (
            (apex_angles < cone_angle + self.epsilon)
            & (rot_pts[:, 2] > np.min([self.origin[2], self.apex[2]]) - self.epsilon)
            & (rot_pts[:, 2] < np.max([self.origin[2], self.apex[2]]) + self.epsilon)
        )
        return inside_points

    def wireframe_points(self, nb_points_1=30, nb_points_2=30):
        # Set up the grid in polar coordinates
        theta = np.linspace(0, 2 * np.pi, nb_points_1)
        r = np.linspace(0, self.radius, nb_points_2)
        theta, r = np.meshgrid(theta, r)

        # Height vector
        hv = np.array(self.origin) - np.array(self.apex)
        height = np.linalg.norm(hv)
        # angle = np.arctan(height/self.radius)

        # Generate a cone with the apex in the origin and the center at (0,0,1)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = r * height / self.radius

        # Rotate the cone
        return _rotate_by_coord(x, y, z, hv, self.apex, test_hv=True)


@config.node
class Cylinder(GeometricShape, classmap_entry="cylinder"):
    """
    A cylinder, described in cartesian coordinates.
    """

    origin = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 0.0, 0.0]
    )
    """The coordinates of the center of the bottom circle of the cylinder."""
    top_center = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 2.0, 0.0]
    )
    """The coordinates of the center of the top circle of the cylinder."""
    radius = config.attr(type=float, required=False, default=1.0)
    """The radius of the base circle."""

    def find_mbb(self):
        height = np.linalg.norm(self.top_center - self.origin)
        # Extrema of the xyz standard cyl
        extrema = [
            np.array([-self.radius, -self.radius, 0.0]),
            np.array([-self.radius, self.radius, 0.0]),
            np.array([self.radius, -self.radius, 0.0]),
            np.array([self.radius, self.radius, 0.0]),
            np.array([self.radius, self.radius, height]),
            np.array([-self.radius, self.radius, height]),
            np.array([self.radius, -self.radius, height]),
            np.array([-self.radius, -self.radius, height]),
        ]

        # Rotate the cylinder

        return _get_extrema_after_rot(extrema, self.origin, self.top_center)

    def get_volume(self):
        h = np.linalg.norm(self.top_center - self.origin)
        b = np.pi * self.radius * self.radius
        return b * h

    def translate(self, t_vector: np.ndarray[float]):
        self.origin += t_vector
        self.top_center += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: np.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        # rotation according to bottom center
        self.top_center = rot.apply(self.top_center)

    def generate_point_cloud(self, npoints: int):
        # Generate an ellipse orientated along x,y,z
        cloud = np.full((npoints, 3), 0, dtype=float)
        theta = np.pi * 2.0 * np.random.rand(npoints)
        rand = np.random.rand(npoints, 3)
        height = np.linalg.norm(self.top_center - self.origin)

        # Generate an ellipsoid centered at the origin, with the semiaxes on x,y,z
        cloud[:, 0] = self.radius * np.cos(theta)
        cloud[:, 1] = self.radius * np.sin(theta)
        cloud[:, 2] = height
        cloud = cloud * rand

        # Rotate the cylinder
        hv = (self.top_center - self.origin) / height
        rot = _get_rotation_vector(hv / np.linalg.norm(hv))
        cloud = rot.apply(cloud)
        cloud = cloud + self.origin
        return cloud

    def check_inside(self, points: np.ndarray[float]):
        # Translate back to origin
        pts = points - self.origin

        # Rotate back to xyz
        height = np.linalg.norm(self.top_center - self.origin)
        rot = _get_rotation_vector(
            (self.top_center - self.origin) / height, positive_angle=False
        )
        rot_pts = rot.apply(pts)
        # Check for intersections
        inside_points = (
            (rot_pts[:, 2] < height + self.epsilon)
            & (rot_pts[:, 2] > -self.epsilon)
            & (
                rot_pts[:, 0] * rot_pts[:, 0] + rot_pts[:, 1] * rot_pts[:, 1]
                < self.radius**2 + self.epsilon
            )
        )
        return inside_points

    def wireframe_points(self, nb_points_1=30, nb_points_2=30):
        # Set up the grid in polar coordinates
        theta = np.linspace(0, 2 * np.pi, nb_points_1)

        # Height vector
        hv = np.array(self.origin) - np.array(self.top_center)
        height = np.linalg.norm(hv)

        h = np.linspace(0, height, nb_points_2)
        theta, h = np.meshgrid(theta, h)

        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = h

        # Rotate the cylinder
        hv = (self.top_center - self.origin) / height
        return _rotate_by_coord(x, y, z, hv, self.origin)


@config.node
class Sphere(GeometricShape, classmap_entry="sphere"):
    """
    A sphere, described in cartesian coordinates.
    """

    origin = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 0.0, 0.0]
    )
    """The coordinates of the center of the sphere."""
    radius = config.attr(type=float, required=False, default=1.0)
    """The radius of the sphere."""

    def find_mbb(self):
        # Find the minimum bounding box, to avoid computing it every time
        mbb_min = np.array([-self.radius, -self.radius, -self.radius]) + self.origin
        mbb_max = np.array([self.radius, self.radius, self.radius]) + self.origin
        return mbb_min, mbb_max

    def get_volume(self):
        return np.pi * 4.0 / 3.0 * np.power(self.radius, 3)

    def translate(self, t_vector: np.ndarray[float]):
        self.origin += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: np.ndarray[float], angle: float):  # pragma: no cover
        # It's a sphere, it's invariant under rotation!
        pass

    def surface_function(self, theta, phi):
        """
        Convert polar coordinates into their 3D location on the sphere surface.

        :param float|numpy.ndarray[float] theta: first polar coordinate in [0; 2*np.pi]
        :param float|numpy.ndarray[float] phi: second polar coordinate in [0; np.pi]
        :return: surface coordinates
        :rtype: float|numpy.ndarray[float]
        """
        return np.array(
            [
                self.radius * np.cos(theta) * np.sin(phi),
                self.radius * np.sin(theta) * np.sin(phi),
                self.radius * np.cos(phi),
            ]
        )

    def generate_point_cloud(self, npoints: int):
        # Generate a sphere centered at the origin.
        cloud = uniform_surface_sampling(npoints, self.surface_function)
        cloud = cloud.T * np.random.rand(npoints, 3)  # sample within the shape

        cloud = cloud + self.origin

        return cloud

    def check_inside(self, points: np.ndarray[float]):
        # Translate the points, bringing the origin to the center of the sphere,
        # then check the inequality defining the sphere
        pts_centered = points - self.origin
        lhs = np.linalg.norm(pts_centered, axis=1)
        inside_points = lhs < self.radius + self.epsilon
        return inside_points

    def wireframe_points(self, nb_points_1=30, nb_points_2=30):
        x, y, z = uniform_surface_wireframe(
            nb_points_1, nb_points_2, self.surface_function
        )
        return translate_3d_mesh_by_vec(x, y, z, self.origin)


@config.node
class Cuboid(GeometricShape, classmap_entry="cuboid"):
    """
    A rectangular parallelepiped, described in cartesian coordinates.
    """

    origin = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 0.0, 0.0]
    )
    """The coordinates of the center of the barycenter of the bottom rectangle."""
    top_center = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 1.0, 0.0]
    )
    """The coordinates of the center of the barycenter of the top rectangle."""
    side_length_1 = config.attr(type=float, required=False, default=1.0)
    """Length of one side of the base rectangle."""
    side_length_2 = config.attr(type=float, required=False, default=1.0)
    """Length of the other side of the base rectangle."""

    def find_mbb(self):
        # Extrema of the cuboid centered at the origin
        extrema = [
            np.array([-self.side_length_1 / 2.0, -self.side_length_2 / 2.0, 0.0]),
            np.array([self.side_length_1 / 2.0, self.side_length_2 / 2.0, 0.0]),
            np.array([-self.side_length_1 / 2.0, self.side_length_2 / 2.0, 0.0]),
            np.array([self.side_length_1 / 2.0, -self.side_length_2 / 2.0, 0.0]),
            np.array(
                [
                    self.side_length_1 / 2.0 + self.top_center[0],
                    self.side_length_2 / 2.0 + self.top_center[1],
                    self.top_center[2],
                ]
            ),
            np.array(
                [
                    -self.side_length_1 / 2.0 + self.top_center[0],
                    self.side_length_2 / 2.0 + self.top_center[1],
                    self.top_center[2],
                ]
            ),
            np.array(
                [
                    -self.side_length_1 / 2.0 + self.top_center[0],
                    -self.side_length_2 / 2.0 + self.top_center[1],
                    self.top_center[2],
                ]
            ),
            np.array(
                [
                    self.side_length_1 / 2.0 + self.top_center[0],
                    -self.side_length_2 / 2.0 + self.top_center[1],
                    self.top_center[2],
                ]
            ),
        ]

        # Rotate the cuboid
        return _get_extrema_after_rot(extrema, self.origin, self.top_center)

    def get_volume(self):
        h = np.linalg.norm(self.top_center - self.origin)
        return h * self.side_length_1 * self.side_length_2

    def translate(self, t_vector: np.ndarray[float]):
        self.origin += t_vector
        self.top_center += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: np.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.top_center = rot.apply(self.top_center)

    def generate_point_cloud(self, npoints: int):
        # Generate a unit cuboid whose base rectangle has the barycenter in the origin
        rand = np.random.rand(npoints, 3)
        rand[:, 0] = rand[:, 0] - 0.5
        rand[:, 1] = rand[:, 1] - 0.5

        # Scale the sides of the cuboid
        height = np.linalg.norm(self.top_center - self.origin)
        rand[:, 0] = rand[:, 0] * self.side_length_1 / 2.0
        rand[:, 1] = rand[:, 1] * self.side_length_2 / 2.0
        rand[:, 2] = rand[:, 2] * height

        # Rotate the cuboid
        rot = _get_rotation_vector((self.top_center - self.origin) / height)
        cloud = rot.apply(rand)

        # Translate the cuboid
        cloud = cloud + self.origin
        return cloud

    def check_inside(self, points: np.ndarray[float]):
        # Translate back to origin
        pts = points - self.origin

        # Rotate back to xyz
        height = np.linalg.norm(self.top_center - self.origin)
        rot = _get_rotation_vector(
            (self.top_center - self.origin) / height, positive_angle=False
        )
        rot_pts = rot.apply(pts)

        # Check for intersections
        inside_points = (
            (rot_pts[:, 2] < height)
            & (rot_pts[:, 2] > 0.0)
            & (rot_pts[:, 0] < self.side_length_1)
            & (rot_pts[:, 0] > -self.side_length_1)
            & (rot_pts[:, 1] < self.side_length_2)
            & (rot_pts[:, 1] > -self.side_length_2)
        )
        return inside_points

    def wireframe_points(self, **kwargs):
        a = self.side_length_1 / 2.0
        b = self.side_length_2 / 2.0
        c = np.linalg.norm(self.top_center - self.origin)

        x = np.array(
            [
                [-a, a, a, -a],  # x coordinate of points in bottom surface
                [-a, a, a, -a],  # x coordinate of points in upper surface
                [-a, a, -a, a],  # x coordinate of points in outside surface
                [-a, a, -a, a],
            ]
        )  # x coordinate of points in inside surface
        y = np.array(
            [
                [-b, -b, b, b],  # y coordinate of points in bottom surface
                [-b, -b, b, b],  # y coordinate of points in upper surface
                [-b, -b, -b, -b],  # y coordinate of points in outside surface
                [b, b, b, b],
            ]
        )  # y coordinate of points in inside surface
        z = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],  # z coordinate of points in bottom surface
                [c, c, c, c],  # z coordinate of points in upper surface
                [0.0, 0.0, c, c],  # z coordinate of points in outside surface
                [0.0, 0.0, c, c],
            ]
        )  # z coordinate of points in inside surface

        # Rotate the cuboid
        hv = (self.top_center - self.origin) / c
        return _rotate_by_coord(x, y, z, hv, self.origin)


@config.node
class Parallelepiped(GeometricShape, classmap_entry="parallelepiped"):
    """
    A generic parallelepiped, described by the vectors (following the right-hand orientation) of the
    sides in cartesian coordinates
    """

    origin = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 0.0, 0.0]
    )
    """The coordinates of the left-bottom edge."""
    side_vector_1 = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[1.0, 0.0, 0.0]
    )
    """The first vector identifying the parallelepiped (using the right-hand orientation: the 
        thumb)."""
    side_vector_2 = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 1.0, 0.0]
    )
    """The second vector identifying the parallelepiped (using the right-hand orientation: the 
        index)."""
    side_vector_3 = config.attr(
        type=types.ndarray(shape=(3,), dtype=float), required=True, hint=[0.0, 0.0, 1.0]
    )
    """The third vector identifying the parallelepiped (using the right-hand orientation: the 
        middle finger)."""

    def find_mbb(self):
        extrema = np.vstack(
            [
                np.array([0.0, 0.0, 0.0]),
                self.side_vector_1 + self.side_vector_2 + self.side_vector_3,
            ]
        )
        maxima = np.max(extrema, axis=0) + self.origin
        minima = np.min(extrema, axis=0) + self.origin
        return minima, maxima

    def get_volume(self):
        vol = np.dot(self.side_vector_3, np.cross(self.side_vector_1, self.side_vector_2))
        return vol

    def translate(self, t_vector: np.ndarray[float]):
        self.origin += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: np.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        # self.center = rot.apply(self.center)
        self.side_vector_1 = rot.apply(self.side_vector_1)
        self.side_vector_2 = rot.apply(self.side_vector_2)
        self.side_vector_3 = rot.apply(self.side_vector_3)

    def generate_point_cloud(self, npoints: int):
        # Generate a linear combination of points in the volume
        cloud = np.full((npoints, 3), 0, dtype=float)
        rand = np.random.rand(npoints, 3)
        for i in range(npoints):
            cloud[i] = (
                rand[i, 0] * self.side_vector_1
                + rand[i, 1] * self.side_vector_2
                + rand[i, 2] * self.side_vector_3
            )
        cloud += self.origin
        return cloud

    def check_inside(self, points: np.ndarray[float]):
        # Translate back to origin
        pts = points - self.origin

        # Rotate back to xyz
        height = np.linalg.norm(self.side_vector_3)
        rot = _get_rotation_vector(hv=self.side_vector_3 / height, positive_angle=True)
        rot_pts = rot.apply(pts)

        # Compute the Fourier components wrt to the vectors identifying the parallelepiped

        v1_norm = np.linalg.norm(self.side_vector_1)
        comp1 = rot_pts.dot(self.side_vector_1) / v1_norm
        v2_norm = np.linalg.norm(self.side_vector_2)
        comp2 = rot_pts.dot(self.side_vector_2) / v2_norm
        v3_norm = np.linalg.norm(self.side_vector_3)
        comp3 = rot_pts.dot(self.side_vector_3) / v3_norm

        # The points are inside the parallelepiped if and only if all the Fourier components
        # are between 0 and the norm of sides of the parallelepiped
        inside_points = (
            (comp1 > 0.0)
            & (comp1 < v1_norm)
            & (comp2 > 0.0)
            & (comp2 < v2_norm)
            & (comp3 > 0.0)
            & (comp3 < v3_norm)
        )
        return inside_points

    def wireframe_points(self, **kwargs):
        va = self.side_vector_1
        vb = self.side_vector_2
        vc = self.side_vector_3

        a = va
        b = va + vb
        c = vb
        d = np.array([0.0, 0.0, 0.0])
        e = va + vc
        f = va + vb + vc
        g = vb + vc
        h = vc

        x = np.array(
            [
                [a[0], b[0], c[0], d[0]],
                [e[0], f[0], g[0], h[0]],
                [a[0], b[0], f[0], e[0]],
                [d[0], c[0], g[0], h[0]],
            ]
        )
        y = np.array(
            [
                [a[1], b[1], c[1], d[1]],
                [e[1], f[1], g[1], h[1]],
                [a[1], b[1], f[1], e[1]],
                [d[1], c[1], g[1], h[1]],
            ]
        )
        z = np.array(
            [
                [a[2], b[2], c[2], d[2]],
                [e[2], f[2], g[2], h[2]],
                [a[2], b[2], f[2], e[2]],
                [d[2], c[2], g[2], h[2]],
            ]
        )

        return x + self.origin[0], y + self.origin[1], z + self.origin[2]
