from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import abc
import copy
from typing import List, Tuple
import numpy
from bsb import config
from bsb.config import types

from .cloud_mesh_utils import (
    rotate_3d_mesh_by_vec,
    translate_3d_mesh_by_vec,
    rotate_3d_mesh_by_rot_mat,
)


def inside_mbox(
    points: numpy.ndarray[float],
    mbb_min: numpy.ndarray[float],
    mbb_max: numpy.ndarray[float],
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
    def find_mbb(self):
        """
        Compute the minimum bounding box surrounding the shape.
        """
        pass

    def check_mbox(self, points: numpy.ndarray[float]):
        """
        Check if the points given in input are inside the minimal bounding box.

        :param numpy.ndarray points: A cloud of points.
        :return: A bool np.ndarray specifying whether each point of the input array is inside the
            minimal bounding box or not.
        :rtype: numpy.ndarray
        """
        return inside_mbox(points, self.mbb_min, self.mbb_max)

    @abc.abstractmethod
    def get_volume(self):
        """
        Get the volume of the geometric shape.
        :return: The volume of the geometric shape.
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def translate(self, t_vector: numpy.ndarray[float]):
        """
        Translate the geometric shape by the vector t_vector.

        :param numpy.ndarray t_vector: The displacement vector
        """
        pass

    @abc.abstractmethod
    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        """
        Rotate all the shapes around r_versor, which is a versor passing through the origin,
        by the specified angle.

        :param r_versor: A versor specifying the rotation axis.
        :type r_versor: numpy.ndarray[float]
        :param float angle: the rotation angle, in radians.
        """
        pass

    @abc.abstractmethod
    def generate_point_cloud(self, npoints: int):
        """
        Generate a point cloud made by npoints points.

        :param int npoints: The number of points to generate.
        :return: a (npoints x 3) numpy array.
        :rtype: numpy.ndarray
        """
        pass

    @abc.abstractmethod
    def check_inside(self, points: numpy.ndarray[float]) -> numpy.ndarray[bool]:
        """
        Check if the points given in input are inside the geometric shape.

        :param numpy.ndarray points: A cloud of points.
        :return: A bool array of length npoints, containing whether the -ith point is inside the
            geometric shape or not.
        :rtype: numpy.ndarray
        """
        pass

    @abc.abstractmethod
    def wireframe_points(self):
        """
        Generate a wireframe to plot the geometric shape.

        :return: An array of 3D points
        :rtype: numpy.ndarray
        """
        pass

    def clone(self):
        # TODO: find a cleaner way to copy
        return type(self)(
            {k: copy.deepcopy(self.__getattribute__(k)) for k in self._config_attr_order}
        )


@config.node
class ShapesComposition:
    """
    A collection of geometric shapes, which can be labelled to distinguish different parts of a
    neuron.
    """

    shapes = config.list(
        type=GeometricShape, required=types.same_size("shapes", "labels", required=True)
    )
    """List of GeometricShape that make up the neuron."""
    labels = config.list(
        type=list, required=types.same_size("shapes", "labels", required=True)
    )
    """List of lists of labels associated to each geometric shape."""
    voxel_size = config.attr(type=float, required=False, default=1.0)
    """Dimension of the side of a voxel, used to determine how many points must be generated
    in a point cloud."""

    def __init__(self, **kwargs):
        # The two corners individuating the minimal bounding box.
        self._mbb_min = np.array([0.0, 0.0, 0.0])
        self._mbb_max = np.array([0.0, 0.0, 0.0])

        self.find_mbb()

    def copy(self) -> ShapesComposition:
        """
        Return a copy of this object.

        :return: A copy of this object.
        :rtype: ShapesComposition
        """
        result = type(self)(dict(voxel_size=self.voxel_size, labels=[], shapes=[]))
        for shape, label in zip(self._shapes, self._labels):
            result._shapes.append(shape.clone())
            result._labels.append(label.copy())
        result._mbb_max = np.copy(self._mbb_max)
        result._mbb_min = np.copy(self._mbb_min)
        return result

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
        result = ShapesComposition(
            dict(voxel_size=self._voxel_size, labels=[], shapes=[])
        )
        selected_id = np.where(np.isin(labels, self._labels))[0]
        result._shapes = [self._shapes[i].clone() for i in selected_id]
        result._labels = [self._labels[i].copy() for i in selected_id]
        result.mbb_min, result.mbb_max = result.find_mbb()
        return result

    def translate(self, t_vec: numpy.ndarray[float]):
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

    def find_mbb(self) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
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
        return [
            int(shape.get_volume() // self._voxel_size**3) for shape in self._shapes
        ]

    def set_voxel_size(self, voxel_size: float):
        """
        Set the size of the side of a voxel.

        :param float voxel_size: Dimension of the side of a voxel.
        """
        self._voxel_size = voxel_size

    def generate_point_cloud(self) -> numpy.ndarray[float]:
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
    ) -> Tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float]]:
        """
        Generate a wireframe to plot the collection of shapes.

        :return: The x,y,z coordinates of the wireframe
        :rtype: Tuple(numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float]) | None
        """
        if len(self._shapes) != 0:
            return tuple(
                np.array([shape.wireframe_points() for shape in self._shapes]).T.tolist()
            )
        return None

    def inside_mbox(self, points: numpy.ndarray[float]) -> numpy.ndarray[bool]:
        """
        Check if the points given in input are inside the minimal bounding box of the collection.

        :param numpy.ndarray points: An array of 3D points.
        :return: A bool np.ndarray specifying whether each point of the input array is inside the
            minimal bounding box of the collection.
        :rtype: numpy.ndarray[bool]
        """
        return inside_mbox(points, self._mbb_min, self._mbb_max)

    def inside_shapes(self, points: numpy.ndarray[float]) -> numpy.ndarray[bool]:
        """
        Check if the points given in input are inside at least in one of the shapes of the
        collection.

        :param numpy.ndarray points: An array of 3D points.
        :return: A bool numpy.ndarray specifying whether each point of the input array is inside the
            collection of shapes or not.
        :rtype: numpy.ndarray[bool]
        """
        if len(self._shapes) != 0:
            cloud = np.full(len(points), 0, dtype=bool)
            for shape in self._shapes:
                tmp = shape.check_mbox(points)
                if np.any(tmp):
                    cloud = cloud | shape.check_inside(points)
            return cloud
        else:
            return None


@config.node
class Ellipsoid(GeometricShape, classmap_entry="ellipsoid"):
    """
    An ellipsoid, described in cartesian coordinates.
    """

    center = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the center of the ellipsoid."""
    lambdas = config.attr(type=types.ndarray(dtype=float), required=True)
    """The length of the three semi-axes."""

    @config.property(type=types.ndarray(), required=True)
    def v0(self):
        """The versor on which the first semi-axis lies."""
        return self._v0

    @v0.setter
    def v0(self, value):
        self._v0 = np.copy(value) / np.linalg.norm(value)

    @config.property(type=types.ndarray(), required=True)
    def v1(self):
        """The versor on which the second semi-axis lies."""
        return self._v1

    @v1.setter
    def v1(self, value):
        self._v1 = np.copy(value) / np.linalg.norm(value)

    @config.property(type=types.ndarray(), required=True)
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
            + self.center
        )
        mbb_min = np.min(extrema, axis=0)
        mbb_max = np.max(extrema, axis=0)
        return mbb_min, mbb_max

    def get_volume(self):
        return np.pi * self.lambdas[0] * self.lambdas[1] * self.lambdas[2]

    def translate(self, t_vector: np.ndarray):
        self.center += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.v0 = rot.apply(self.v0)
        self.v1 = rot.apply(self.v1)
        self.v2 = rot.apply(self.v2)

    def generate_point_cloud(self, npoints: int):
        # Generate an ellipse orientated along x,y,z
        cloud = np.full((npoints, 3), 0, dtype=float)
        theta = np.pi * 2.0 * np.random.rand(npoints)
        phi = 1.0 * np.pi * np.random.rand(npoints)
        rand = np.random.rand(npoints, 3)

        # Generate an ellipsoid centered at the origin, with the semiaxes on x,y,z
        cloud[:, 0] = self.lambdas[0] * np.cos(theta) * np.sin(phi)
        cloud[:, 1] = self.lambdas[1] * np.sin(theta) * np.sin(phi)
        cloud[:, 2] = self.lambdas[2] * np.cos(phi)
        cloud = cloud * rand

        # Rotate the ellipse
        rmat = np.array([self.v0, self.v1, self.v2]).T
        cloud = cloud.dot(rmat)
        cloud = cloud + self.center

        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Check if the quadratic form associated to the ellipse is less than 1 at a point
        diff = points - self.center
        vmat = np.array([self.v0, self.v1, self.v2])
        diag = np.diag(1 / self.lambdas**2)
        qmat = vmat.dot(diag).dot(vmat)
        quad_prod = np.diagonal(diff.dot(qmat.dot(diff.T)))

        # Check if the points are inside the ellipsoid
        inside_points = quad_prod < 1
        return inside_points

    def wireframe_points(self):
        # Generate an ellipse orientated along x,y,z
        theta = np.linspace(0, 2 * np.pi, 90)
        phi = np.linspace(0, np.pi, 90)

        theta, phi = np.meshgrid(theta, phi)

        # Generate an ellipsoid centered at the origin, with the semiaxes on x,y,z
        x = self.lambdas[0] * np.cos(theta) * np.sin(phi)
        y = self.lambdas[1] * np.sin(theta) * np.sin(phi)
        z = self.lambdas[2] * np.cos(phi)

        # Rotate the ellipse
        rmat = np.array([self.v0, self.v1, self.v2]).T
        x, y, z = rotate_3d_mesh_by_rot_mat(x, y, z, rmat)
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self.center)

        return x, y, z


@config.node
class Cone(GeometricShape, classmap_entry="cone"):
    """
    A cone, described in cartesian coordinates.
    """

    apex = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the apex of the cone."""
    center = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the center of the cone."""
    radius = config.attr(type=float, required=False, default=1.0e-3)
    """The radius of the base circle."""

    def find_mbb(self):
        # Vectors identifying half of the sides of the base rectangle in xy
        u = np.array([self.radius, 0, 0])
        v = np.array([0, self.radius, 0])

        # Find the rotation angle and axis
        hv = self.center - self.apex
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)

        # Rotated vectors of the box
        v1 = rot.apply(u)
        v2 = rot.apply(v)
        v3 = self.center - self.apex

        # Coordinates identifying the minimal bounding box
        minima = np.min([v1, v2, v3, -v1, -v2], axis=0)
        maxima = np.max([v1, v2, v3, -v1, -v2], axis=0)
        return minima, maxima

    def get_volume(self):
        h = np.linalg.norm(self.apex - self.center)
        b = np.pi * self.radius * self.radius
        return b * h / 3

    def translate(self, t_vector):
        self.center += t_vector
        self.apex += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.apex = rot.apply(self.apex)

    def generate_point_cloud(self, npoints: int):
        theta = np.pi * 2.0 * np.random.rand(npoints)
        rand_a = np.random.rand(npoints)
        rand_b = np.random.rand(npoints)

        # Height vector
        hv = self.center - self.apex
        cloud = np.full((npoints, 3), 0, dtype=float)

        # Generate a cone with the apex in the origin and the center at (0,0,1)
        cloud[:, 0] = (self.radius * rand_a * np.cos(theta)) * rand_b
        cloud[:, 1] = self.radius * rand_a * np.sin(theta)
        cloud[:, 2] = rand_a * np.linalg.norm(hv)

        # Rotate the cone: Find the axis of rotation and compute the angle
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        if hv[2] < 0:
            cloud[:, 2] = -cloud[:, 2]

        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(cloud)

        # Translate the cone
        cloud = cloud + self.apex
        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Find the vector w of the height.
        h_vector = self.center - self.apex
        height = np.linalg.norm(h_vector)
        hv = h_vector / height

        # Center the points
        pts = points - self.apex

        # Rotate back to xyz
        zvers = np.array([0, 0, 1], dtype=np.float64)
        perp = np.cross(zvers, hv)
        angle = -np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        rot_pts = rot.apply(pts)

        # Find the angle between the points and the apex
        apex_angles = np.arccos(
            np.dot((rot_pts / np.linalg.norm(rot_pts, axis=1)[..., np.newaxis]), hv)
        )
        # Compute the cone angle
        cone_angle = np.arctan(self.radius / height)

        # Select the points inside the cone
        inside_points = (
            (apex_angles < cone_angle + self.epsilon)
            & (rot_pts[:, 2] > np.min([self.center[2], self.apex[2]]) - self.epsilon)
            & (rot_pts[:, 2] < np.max([self.center[2], self.apex[2]]) + self.epsilon)
        )
        return inside_points

    def wireframe_points(self):
        # Set up the grid in polar coordinates
        theta = np.linspace(0, 2 * np.pi, 90)
        r = np.linspace(0, self.radius, 150)
        theta, r = np.meshgrid(theta, r)

        # Height vector
        hv = np.array(self.center) - np.array(self.apex)
        height = np.linalg.norm(hv)
        # angle = np.arctan(height/self.radius)

        # Generate a cone with the apex in the origin and the center at (0,0,1)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = r * height / self.radius

        # Rotate the cone
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, -1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        if hv[2] < 0:
            z = -z
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self.apex)
        return x, y, z


@config.node
class Cylinder(GeometricShape, classmap_entry="cylinder"):
    """
    A cylinder, described in cartesian coordinates.
    """

    bottom_center = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the center of the bottom circle of the cylinder."""
    top_center = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the center of the top circle of the cylinder."""
    radius = config.attr(type=float, required=False, default=1.0e-3)
    """The radius of the base circle."""

    def find_mbb(self):
        height = np.linalg.norm(self.top_center - self.bottom_center)
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
        hv = (self.top_center - self.bottom_center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)

        for i, pt in enumerate(extrema):
            extrema[i] = rot.apply(pt)

        maxima = np.max(extrema, axis=0) + self.bottom_center
        minima = np.min(extrema, axis=0) + self.bottom_center
        return minima, maxima

    def get_volume(self):
        h = np.linalg.norm(self.top_center - self.bottom_center)
        b = np.pi * self.radius * self.radius
        return b * h

    def translate(self, t_vector: numpy.ndarray[float]):
        self.bottom_center += t_vector
        self.top_center += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.top_center = rot.apply(self.top_center)

    def generate_point_cloud(self, npoints: int):
        # Generate an ellipse orientated along x,y,z
        cloud = np.full((npoints, 3), 0, dtype=float)
        theta = np.pi * 2.0 * np.random.rand(npoints)
        rand = np.random.rand(npoints, 3)
        height = np.linalg.norm(self.top_center - self.bottom_center)

        # Generate an ellipsoid centered at the origin, with the semiaxes on x,y,z
        cloud[:, 0] = self.radius * np.cos(theta)
        cloud[:, 1] = self.radius * np.sin(theta)
        cloud[:, 2] = height
        cloud = cloud * rand

        # Rotate the cylinder
        hv = (self.top_center - self.bottom_center) / height
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(cloud)
        cloud = cloud + self.bottom_center
        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Translate back to origin
        pts = points - self.bottom_center

        # Rotate back to xyz
        height = np.linalg.norm(self.top_center - self.bottom_center)
        hv = (self.top_center - self.bottom_center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = -np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
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

    def wireframe_points(self):
        # Set up the grid in polar coordinates
        theta = np.linspace(0, 2 * np.pi, 90)

        # Height vector
        hv = np.array(self.bottom_center) - np.array(self.top_center)
        height = np.linalg.norm(hv)

        h = np.linspace(0, height, 150)
        theta, h = np.meshgrid(theta, h)

        # Generate a cone with the apex in the origin and the center at (0,0,height)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = h

        # Rotate the cylinder
        hv = (self.top_center - self.bottom_center) / height
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self.bottom_center)

        return x, y, z


@config.node
class Sphere(GeometricShape, classmap_entry="sphere"):
    """
    A sphere, described in cartesian coordinates.
    """

    center = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the center of the sphere."""
    radius = config.attr(type=float, required=False, default=1.0e-3)
    """The radius of the sphere."""

    def find_mbb(self):
        # Find the minimum bounding box, to avoid computing it every time
        mbb_min = np.array([-self.radius, -self.radius, -self.radius]) + self.center
        mbb_max = np.array([self.radius, self.radius, self.radius]) + self.center
        return mbb_min, mbb_max

    def get_volume(self):
        return np.pi * 4.0 / 3.0 * np.power(self.radius, 3)

    def translate(self, t_vector: numpy.ndarray[float]):
        self.center += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        # It's a sphere, it's invariant under rotation!
        pass

    def generate_point_cloud(self, npoints: int):
        # Generate a sphere centered at the origin.
        cloud = np.full((npoints, 3), 0, dtype=float)
        theta = np.pi * 2.0 * np.random.rand(npoints)
        phi = -1.0 * np.pi * np.random.rand(npoints)
        rand = np.random.rand(npoints, 3)
        cloud[:, 0] = self.radius * np.cos(theta) * np.sin(phi)
        cloud[:, 1] = self.radius * np.sin(theta) * np.sin(phi)
        cloud[:, 2] = self.radius * np.cos(phi)
        cloud = cloud * rand

        cloud = cloud + self.center

        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Translate the points, bringing the origin to the center of the sphere,
        # then check the inequality defining the sphere
        pts_centered = points - self.center
        lhs = np.linalg.norm(pts_centered, axis=1)
        inside_points = lhs < self.radius + self.epsilon
        return inside_points

    def wireframe_points(self):
        # Generate a sphere centered at the origin
        theta = np.linspace(0, 2 * np.pi, 90)
        phi = np.linspace(0, np.pi, 90)
        theta, phi = np.meshgrid(theta, phi)
        x = self.radius * np.cos(theta) * np.sin(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(phi)

        # Translate the sphere
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self.center)

        return x, y, z


@config.node
class Cuboid(GeometricShape, classmap_entry="cuboid"):
    """
    A rectangular parallelepiped, described in cartesian coordinates.
    """

    bottom_center = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the center of the barycenter of the bottom rectangle."""
    top_center = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the center of the barycenter of the top rectangle."""
    side_length_1 = config.attr(type=float, required=False, default=1.0e-3)
    """Length of one side of the base rectangle."""
    side_length_2 = config.attr(type=float, required=False, default=1.0e-3)
    """Length of the other side of the base rectangle."""

    def find_mbb(self):
        height = np.linalg.norm(self.top_center - self.bottom_center)
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
        hv = (self.top_center - self.bottom_center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)

        for i, pt in enumerate(extrema):
            extrema[i] = rot.apply(pt)

        maxima = np.max(extrema, axis=0) + self.bottom_center
        minima = np.min(extrema, axis=0) + self.bottom_center
        return minima, maxima

    def get_volume(self):
        h = np.linalg.norm(self.top_center - self.bottom_center)
        return h * self.side_length_1 * self.side_length_2

    def translate(self, t_vector: numpy.ndarray[float]):
        self.bottom_center += t_vector
        self.top_center += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.top_center = rot.apply(self.top_center)

    def generate_point_cloud(self, npoints: int):
        # Generate a unit cuboid whose base rectangle has the barycenter in the origin
        rand = np.random.rand(npoints, 3)
        rand[:, 0] = rand[:, 0] - 0.5
        rand[:, 1] = rand[:, 1] - 0.5

        # Scale the sides of the cuboid
        height = np.linalg.norm(self.top_center - self.bottom_center)
        rand[:, 0] = rand[:, 0] * self.side_length_1 / 2.0
        rand[:, 1] = rand[:, 1] * self.side_length_2 / 2.0
        rand[:, 2] = rand[:, 2] * height

        # Rotate the cuboid
        hv = (self.top_center - self.bottom_center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(rand)

        # Translate the cuboid
        cloud = cloud + self.bottom_center
        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Translate back to origin
        pts = points - self.bottom_center

        # Rotate back to xyz
        height = np.linalg.norm(self.top_center - self.bottom_center)
        hv = (self.top_center - self.bottom_center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = -np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
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

    def wireframe_points(self):
        a = self.side_length_1 / 2.0
        b = self.side_length_2 / 2.0
        c = np.linalg.norm(self.top_center - self.bottom_center)

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
        hv = (self.top_center - self.bottom_center) / c
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self.bottom_center)

        return x, y, z

    # -----------------------------------------------------


@config.node
class Parallelepiped(GeometricShape, classmap_entry="parallelepiped"):
    """
    A generic parallelepiped, described by the vectors (following the right-hand orientation) of the
    sides in cartesian coordinates
    """

    center = config.attr(type=types.ndarray(dtype=float), required=True)
    """The coordinates of the left-bottom edge."""
    side_vector_1 = config.attr(type=types.ndarray(dtype=float), required=True)
    """The first vector identifying the parallelepiped (using the right-hand orientation: the 
        thumb)."""
    side_vector_2 = config.attr(type=types.ndarray(dtype=float), required=True)
    """The second vector identifying the parallelepiped (using the right-hand orientation: the 
        index)."""
    side_vector_3 = config.attr(type=types.ndarray(dtype=float), required=True)
    """The third vector identifying the parallelepiped (using the right-hand orientation: the 
        middle finger)."""

    def find_mbb(self):
        extrema = np.vstack(
            [
                np.array([0.0, 0.0, 0.0]),
                self.side_vector_1 + self.side_vector_2 + self.side_vector_3,
            ]
        )
        maxima = np.max(extrema, axis=0) + self.center
        minima = np.min(extrema, axis=0) + self.center
        return minima, maxima

    def get_volume(self):
        vol = np.dot(self.side_vector_3, np.cross(self.side_vector_1, self.side_vector_2))
        return vol

    def translate(self, t_vector: numpy.ndarray[float]):
        self.center += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
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
        cloud = cloud + self.center
        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Translate back to origin
        pts = points - self.center

        # Rotate back to xyz
        height = np.linalg.norm(self.side_vector_3)
        hv = self.side_vector_3 / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = -np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
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

    def wireframe_points(self):
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

        return x, y, z
