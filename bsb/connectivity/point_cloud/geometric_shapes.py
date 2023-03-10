from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import abc
import copy
from typing import List, Tuple
import numpy

from .cloud_mesh_utils import (
    rotate_3d_mesh_by_vec,
    translate_3d_mesh_by_vec,
    rotate_3d_mesh_by_rot_mat,
)


class GeometricShape(abc.ABC):
    """
    Base class for geometric shapes
    """

    def __init__(self):
        self._mbb_min, self._mbb_max = self.find_mbb()

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
        :return: A bool array of length npoints, containing whether the -ith point is inside the
            minimal bounding box or not.
        :rtype: numpy.ndarray
        """
        inside = (
            (points[:, 0] > self._mbb_min[0])
            & (points[:, 0] < self._mbb_max[0])
            & (points[:, 1] > self._mbb_min[1])
            & (points[:, 1] < self._mbb_max[1])
            & (points[:, 2] > self._mbb_min[2])
            & (points[:, 2] < self._mbb_max[2])
        )

        return inside

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


class ShapesComposition:
    """
    A collection of geometric shapes, which can be labelled to distinguish different parts of a
    neuron.
    """

    def __init__(self, voxel_size=1.0):
        # A list of GeometricShape.
        self._shapes = []
        # Dimension of the side of a voxel, used to determine how many points must be generated
        # in a point cloud.
        self._voxel_size = voxel_size
        # The two corners individuating the minimal bounding box.
        self._mbb_min = np.array([0.0, 0.0, 0.0])
        self._mbb_max = np.array([0.0, 0.0, 0.0])
        # The labels of each geometric shape in the collection.
        self._labels = []

    def copy(self) -> ShapesComposition:
        """
        Return a copy of this object.

        :return: A copy of this object.
        :rtype: ShapesComposition
        """
        return copy.deepcopy(self)

    def add_shape(self, shape: GeometricShape, labels: List[str]):
        """
        Add a geometric shape to the collection

        :param GeometricShape shape: A GeometricShape to add to the collection.
        :param List[str] labels: A list of labels for the geometric shape to add.
        """
        self._shapes.append(shape)
        self._labels.append(labels)

    def filter_by_labels(self, labels: List[str]) -> ShapesComposition:
        """
        Filter the collection of shapes, returning only the ones corresponding the the given labels.

        :param List[str] labels: A list of labels.
        :return: A new ShapesComposition object containing only the shapes labelled as specified.
        :rtype: ShapesComposition
        """
        result = copy.deepcopy(self)
        selected_id = []
        for i, lb_list in enumerate(self._labels):
            for to_select in labels:
                if to_select in lb_list:
                    selected_id.append(i)
        selected_id = set(selected_id)
        not_selected_id = []
        for i in range(len(self._shapes)):
            if i not in selected_id:
                not_selected_id.append(i)
        result.shapes = []
        result.labels = []
        for nn, i in enumerate(selected_id):
            result.shapes.append(copy.deepcopy(self._shapes[i]))
            result.labels.append(copy.deepcopy(self._labels[i]))
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
        volumes = []
        if len(self.shape) != 0:
            for shape in self._shapes:
                volumes.append(shape.get_volume())
        return volumes

    def find_mbb(self) -> Tuple(numpy.ndarray[float], numpy.ndarray[float]):
        """
        Compute the minimal bounding box containing the collection of shapes.

        :return: The two corners individuating a the minimal bounding box
        :rtype: Tuple(numpy.ndarray[float], numpy.ndarray[float])
        """
        mins = np.empty([len(self._shapes), 3])
        maxs = np.empty([len(self._shapes), 3])
        for i, shape in enumerate(self._shapes):
            mins[i, :] = shape._mbb_min
            maxs[i, :] = shape._mbb_max

        box_min = np.min(mins, axis=0)
        box_max = np.max(maxs, axis=0)
        return box_min, box_max

    def compute_n_points(self) -> List[int]:
        """
        Compute the number of points to generate in a point cloud, using the dimension of the voxel
        specified in self._voxel_size.

        :return: The number of points to generate.
        :rtype: List[int]
        """
        npoints = []
        if len(self._shapes) != 0:
            for shape in self._shapes:
                npoints.append(int(shape.get_volume() // (self._voxel_size) ** 3))
        return npoints

    def set_voxel_size(self, voxel_size: float):
        """
        Set the size of the side of a voxel.

        :param numpy.ndarray t_vec: The displacement vector.
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
            cloud = np.empty([3], dtype=float)
            npoints = self.compute_n_points()
            for shape, numpts in zip(self._shapes, npoints):
                tmp = shape.generate_point_cloud(numpts)
                cloud = np.vstack((cloud, tmp))
            return cloud[1:]
        else:
            return None

    def generate_wireframe(
        self,
    ) -> Tuple(numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float]):
        """
        Generate a wireframe to plot the collection of shapes.

        :return: The x,y,z coordinates of the wireframe
        :rtype: Tuple(numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float]) | None
        """
        if len(self._shapes) != 0:
            cloud = np.empty([3])
            npoints = self.compute_n_points()
            x = []
            y = []
            z = []
            for shape in self._shapes:
                xt, yt, zt = shape.wireframe_points()
                x.append(xt)
                y.append(yt)
                z.append(zt)
            return x, y, z
        else:
            return None

    def inside_mbox(self, points: numpy.ndarray[float]) -> numpy.ndarray[bool]:
        """
        Check if the points given in input are inside the minimal bounding box of the collection.

        :param numpy.ndarray points: An array of 3D points.
        :return: A bool np.ndarray specifying whether each point of the input array is inside the
            minimal bounding box of the collection.
        :rtype: numpy.ndarray[bool]
        """
        inside = (
            (points[:, 0] > self._mbb_min[0])
            & (points[:, 0] < self._mbb_max[0])
            & (points[:, 1] > self._mbb_min[1])
            & (points[:, 1] < self._mbb_max[1])
            & (points[:, 2] > self._mbb_min[2])
            & (points[:, 2] < self._mbb_max[2])
        )

        return inside

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

    def save_to_file(self, filename: str):
        """
        Save the collection of shapes to file.

        :param str filename: The name of the output file.
        """
        with open(filename, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_file(self, filename: str):
        """
        Load a collection of shapes from a file.

        :param str filename: The name of the output file.
        """
        tmp = None
        with open(filename, "rb") as handle:
            tmp = pickle.load(handle)
        self._shapes = tmp._shapes
        self._voxel_size = tmp._voxel_size
        self._mbb_min, self._mbb_max = self.find_mbb()
        self._labels = tmp._labels


class Ellipsoid(GeometricShape):
    """
    An ellipsoid, described in cartesian coordinates.
    """

    def __init__(
        self,
        center: numpy.ndarray[float],
        lambdas: numpy.ndarray[float],
        v0: numpy.ndarray[float],
        v1: numpy.ndarray[float],
        v2: numpy.ndarray[float],
        epsilon=1.0e-3,
    ):
        """
        :param numpy.ndarray[float] center: The coordinates of the center of the ellipsoid.
        :param numpy.ndarray[float] lambdas: The length of the three semi-axes.
        :param numpy.ndarray[float] center: The versor on which the first semi-axis lies.
        :param numpy.ndarray[float] center: The versor on which the second semi-axis lies.
        :param numpy.ndarray[float] center: The versor on which the third semi-axis lies.
        :param float epsilon: Tolerance value to compare coordinates.
        """

        self._center = copy.deepcopy(center)
        self.lambdas = copy.deepcopy(lambdas)
        self.v0 = copy.deepcopy(v0) / np.linalg.norm(v0)
        self.v1 = copy.deepcopy(v1) / np.linalg.norm(v1)
        self.v2 = copy.deepcopy(v2) / np.linalg.norm(v2)
        self._epsilon = epsilon
        super().__init__()

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
            + self._center
        )
        mbb_min = np.min(extrema, axis=0)
        mbb_max = np.max(extrema, axis=0)
        return mbb_min, mbb_max

    def get_volume(self):
        return np.pi * self.lambdas[0] * self.lambdas[1] * self.lambdas[2]

    def translate(self, t_vector: np.ndarray):
        self._center += t_vector
        self._mbb_min += t_vector
        self._mbb_max += t_vector

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
        cloud = cloud + self._center

        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Check if the quadratic form associated to the ellipse is less than 1 at a point
        diff = points - self._center
        vmat = np.array([self.v0, self.v1, self.v2])
        diag = np.diag(1 / self.lambdas**2)
        qmat = (vmat).dot(diag).dot(vmat)
        quad_prod = np.full((len(points)), 0, dtype=float)

        # TODO: Find a way to vectorize these computations
        for i, p in enumerate(diff):
            quad_prod[i] = p.dot(qmat.dot(p.T))

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
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self._center)

        return x, y, z


class Cone(GeometricShape):
    """
    A cone, described in cartesian coordinates.
    """

    def __init__(
        self,
        apex: numpy.ndarray[float],
        center: numpy.ndarray[float],
        radius: float,
        epsilon=1.0e-3,
    ):
        """
        :param numpy.ndarray[float] apex: The coordinates of the apex of the cone.
        :param numpy.ndarray[float] center: The coordinates of the center of the base circle.
        :param float radius: The radius of the base circle.
        :param float epsilon: Tolerance value to compare coordinates.
        """

        self._center = copy.deepcopy(center)
        self._radius = radius
        self._apex = copy.deepcopy(apex)
        self._epsilon = epsilon

        super().__init__()

    def find_mbb(self):
        # Vectors identifying half of the sides of the base rectangle in xy
        u = np.array([self._radius, 0, 0])
        v = np.array([0, self._radius, 0])

        # Find the rotation angle and axis
        hv = self._center - self._apex
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)

        # Rotated vectors of the box
        v1 = rot.apply(u)
        v2 = rot.apply(v)
        v3 = self._center - self._apex

        # Coordinates identifying the minimal bounding box
        minima = np.min([v1, v2, v3, -v1, -v2], axis=0)
        maxima = np.max([v1, v2, v3, -v1, -v2], axis=0)
        return minima, maxima

    def get_volume(self):
        h = np.linalg.norm(self._apex - self._center)
        b = np.pi * self._radius * self._radius
        return b * h / 3

    def translate(self, t_vector):
        self._center += t_vector
        self._apex += t_vector
        self._mbb_min += t_vector
        self._mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        # self._center = rot.apply(self._center)
        self._apex = rot.apply(self._apex)

    def generate_point_cloud(self, npoints: int):
        theta = np.pi * 2.0 * np.random.rand(npoints)
        rand_a = np.random.rand(npoints)
        rand_b = np.random.rand(npoints)

        # Height vector
        hv = self._center - self._apex
        cloud = np.full((npoints, 3), 0, dtype=float)

        # Generate a cone with the apex in the origin and the center at (0,0,1)
        cloud[:, 0] = (self._radius * rand_a * np.cos(theta)) * rand_b
        cloud[:, 1] = self._radius * rand_a * np.sin(theta)
        cloud[:, 2] = rand_a * np.linalg.norm(hv)

        # Rotate the cone: Find the axis of rotation and compute the angle
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        # print(angle)

        if hv[2] < 0:
            cloud[:, 2] = -cloud[:, 2]

        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(cloud)

        # Translate the cone
        cloud = cloud + self._apex
        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Find the vector w of the height.
        h_vector = self._center - self._apex
        height = np.linalg.norm(h_vector)
        hv = h_vector / height

        # Center the points
        pts = points - self._apex

        # Rotate back to xyz
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = -np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        rot_pts = rot.apply(points)

        # Find the angle between the points and the apex
        diff = rot_pts - np.array([0, 0, height])
        apex_angles = np.full((len(points)), 0, dtype=float)

        # TODO: Find a way to vectorize these computation
        for i, p in enumerate(diff):
            apex_angles[i] = np.arccos(np.dot(p / np.linalg.norm(p), hv))

        # Compute the cone angle
        cone_angle = np.arctan(self._radius / height)

        # Select the points inside the cone
        inside_points = (
            (apex_angles < cone_angle + self._epsilon)
            & (rot_pts[:, 2] > np.min([self._center[2], self._apex[2]]) - self._epsilon)
            & (rot_pts[:, 2] < np.max([self._center[2], self._apex[2]]) + self._epsilon)
        )
        return inside_points

    def wireframe_points(self):
        # Set up the grid in polar coordinates
        theta = np.linspace(0, 2 * np.pi, 90)
        r = np.linspace(0, self._radius, 150)
        theta, r = np.meshgrid(theta, r)

        # Height vector
        hv = np.array(self._center) - np.array(self._apex)
        height = np.linalg.norm(hv)
        # angle = np.arctan(height/self._radius)

        # Generate a cone with the apex in the origin and the center at (0,0,1)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = r * height / self._radius

        # Rotate the cone
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, -1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        if hv[2] < 0:
            z = -z
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self._apex)

        return x, y, z


class Cylinder(GeometricShape):
    """
    A cylinder, described in cartesian coordinates.
    """

    def __init__(
        self,
        center: numpy.ndarray[float],
        radius: float,
        height_vector: numpy.ndarray[float],
        epsilon=1e-3,
    ):
        """
        :param numpy.ndarray[float] center: The coordinates of the center of the bottom circle of
            the cylinder.
        :param float radius: The radius of the circle.
        :param numpy.ndarray[float] height_vector: The coordinates of the center of the top circle
            of the cylinder.
        :param float epsilon: Tolerance value to compare coordinates.
        """

        self._center = copy.deepcopy(center)
        self._radius = radius
        # Position of the apex
        self._height_vector = copy.deepcopy(height_vector)
        self._epsilon = epsilon

        super().__init__()

    def find_mbb(self):
        height = np.linalg.norm(self._height_vector - self._center)
        # Extrema of the xyz standard cyl
        extrema = [
            np.array([-self._radius, -self._radius, 0.0]),
            np.array([-self._radius, self._radius, 0.0]),
            np.array([self._radius, -self._radius, 0.0]),
            np.array([self._radius, self._radius, 0.0]),
            np.array([self._radius, self._radius, height]),
            np.array([-self._radius, self._radius, height]),
            np.array([self._radius, -self._radius, height]),
            np.array([-self._radius, -self._radius, height]),
        ]

        # Rotate the cylinder
        hv = (self._height_vector - self._center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)

        for i, pt in enumerate(extrema):
            extrema[i] = rot.apply(pt)

        maxima = np.max(extrema, axis=0) + self._center
        minima = np.min(extrema, axis=0) + self._center
        return minima, maxima

    def get_volume(self):
        h = np.linalg.norm(self._height_vector - self._center)
        b = np.pi * self._radius * self._radius
        return b * h

    def translate(self, t_vector: numpy.ndarray[float]):
        self._center += t_vector
        self._height_vector += t_vector
        self._mbb_min += t_vector
        self._mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        # self._center = rot.apply(self._center)
        self._height_vector = rot.apply(self._height_vector)

    def generate_point_cloud(self, npoints: int):
        # Generate an ellipse orientated along x,y,z
        cloud = np.full((npoints, 3), 0, dtype=float)
        theta = np.pi * 2.0 * np.random.rand(npoints)
        rand = np.random.rand(npoints, 3)
        height = np.linalg.norm(self._height_vector - self._center)

        # Generate an ellipsoid centered at the origin, with the semiaxes on x,y,z
        cloud[:, 0] = self._radius * np.cos(theta)
        cloud[:, 1] = self._radius * np.sin(theta)
        cloud[:, 2] = height
        cloud = cloud * rand

        # Rotate the cylinder
        hv = (self._height_vector - self._center) / height
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(cloud)
        cloud = cloud + self._center
        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Translate back to origin
        pts = points - self._center

        # Rotate back to xyz
        height = np.linalg.norm(self._height_vector - self._center)
        hv = (self._height_vector - self._center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = -np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        rot_pts = rot.apply(pts)
        # print(rot_pts[0])
        # Check for intersections
        inside_points = (
            (rot_pts[:, 2] < height + self._epsilon)
            & (rot_pts[:, 2] > -self._epsilon)
            & (
                rot_pts[:, 0] * rot_pts[:, 0] + rot_pts[:, 1] * rot_pts[:, 1]
                < self._radius**2 + self._epsilon
            )
        )
        return inside_points

    def wireframe_points(self):
        # Set up the grid in polar coordinates
        theta = np.linspace(0, 2 * np.pi, 90)

        # Height vector
        hv = np.array(self._center) - np.array(self._height_vector)
        height = np.linalg.norm(hv)

        h = np.linspace(0, height, 150)
        theta, h = np.meshgrid(theta, h)

        # Generate a cone with the apex in the origin and the center at (0,0,height)
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)
        z = h

        # Rotate the cylinder
        hv = (self._height_vector - self._center) / height
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self._center)

        return x, y, z


class Sphere(GeometricShape):
    """
    A sphere, described in cartesian coordinates.
    """

    def __init__(self, center: numpy.ndarray[float], radius: float, epsilon=1e-3):
        """
        :param numpy.ndarray[float] center: The coordinates of the center of the sphere.
        :param float radius: The radius of the sphere.
        :param float epsilon: Tolerance value to compare coordinates.
        """
        self._center = copy.deepcopy(center)
        self._radius = radius
        self._epsilon = epsilon
        super().__init__()

    def find_mbb(self):
        # Find the minimum bounding box, to avoid computing it every time
        mbb_min = np.array([-self._radius, -self._radius, -self._radius]) + self._center
        mbb_max = np.array([self._radius, self._radius, self._radius]) + self._center
        return mbb_min, mbb_max

    def get_volume(self):
        return np.pi * 4.0 / 3.0 * np.power(self._radius, 3)

    def translate(self, t_vector: numpy.ndarray[float]):
        self._center += t_vector
        self._mbb_min += t_vector
        self._mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        # It's a sphere, it's invariant under rotation!
        pass

    def generate_point_cloud(self, npoints: int):
        # Generate a sphere centered at the origin.
        cloud = np.full((npoints, 3), 0, dtype=float)
        theta = np.pi * 2.0 * np.random.rand(npoints)
        phi = -1.0 * np.pi * np.random.rand(npoints)
        rand = np.random.rand(npoints, 3)
        cloud[:, 0] = self._radius * np.cos(theta) * np.sin(phi)
        cloud[:, 1] = self._radius * np.sin(theta) * np.sin(phi)
        cloud[:, 2] = self._radius * np.cos(phi)
        cloud = cloud * rand

        cloud = cloud + self._center

        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Translate the points, bringing the origin to the center of the sphere,
        # then check the inequality defining the sphere
        pts_centered = points - self._center
        lhs = np.linalg.norm(pts_centered, axis=1)
        inside_points = lhs < self._radius + self._epsilon
        return inside_points

    def wireframe_points(self):
        # Generate a sphere centered at the origin
        theta = np.linspace(0, 2 * np.pi, 90)
        phi = np.linspace(0, np.pi, 90)
        theta, phi = np.meshgrid(theta, phi)
        x = self._radius * np.cos(theta) * np.sin(phi)
        y = self._radius * np.sin(theta) * np.sin(phi)
        z = self._radius * np.cos(phi)

        # Translate the sphere
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self._center)

        return x, y, z


class Cuboid(GeometricShape):
    """
    A rectangular parallelepiped, described in cartesian coordinates.
    """

    def __init__(
        self,
        center: numpy.ndarray[float],
        side_length_1: float,
        side_length_2: float,
        height_vector: numpy.ndarray[float],
        epsilon=1e-3,
    ):
        """
        :param numpy.ndarray[float] center: The coordinates of the barycenter of the bottom
            rectangle.
        :param float side_length_1: Length of one side of the base rectangle.
        :param float side_length_2: Length of the other side of the base rectangle.
        :param numpy.ndarray[float] height_vector: The coordinates of the barycenter of the top
            rectangle.
        :param float epsilon: Tolerance value to compare coordinates.
        """

        self._center = copy.deepcopy(center)
        self._side_length_1 = side_length_1
        self._side_length_2 = side_length_2
        # Position of the apex
        self._height_vector = copy.deepcopy(height_vector)
        self._epsilon = epsilon

        super().__init__()

    def find_mbb(self):
        height = np.linalg.norm(self._height_vector - self._center)
        # Extrema of the cuboid centered at the origin
        extrema = [
            np.array([-self._side_length_1 / 2.0, -self._side_length_2 / 2.0, 0.0]),
            np.array([self._side_length_1 / 2.0, self._side_length_2 / 2.0, 0.0]),
            np.array([-self._side_length_1 / 2.0, self._side_length_2 / 2.0, 0.0]),
            np.array([self._side_length_1 / 2.0, -self._side_length_2 / 2.0, 0.0]),
            np.array(
                [
                    self._side_length_1 / 2.0 + self._height_vector[0],
                    self._side_length_2 / 2.0 + self._height_vector[1],
                    self._height_vector[2],
                ]
            ),
            np.array(
                [
                    -self._side_length_1 / 2.0 + self._height_vector[0],
                    self._side_length_2 / 2.0 + self._height_vector[1],
                    self._height_vector[2],
                ]
            ),
            np.array(
                [
                    -self._side_length_1 / 2.0 + self._height_vector[0],
                    -self._side_length_2 / 2.0 + self._height_vector[1],
                    self._height_vector[2],
                ]
            ),
            np.array(
                [
                    self._side_length_1 / 2.0 + self._height_vector[0],
                    -self._side_length_2 / 2.0 + self._height_vector[1],
                    self._height_vector[2],
                ]
            ),
        ]

        # Rotate the cuboid
        hv = (self._height_vector - self._center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)

        for i, pt in enumerate(extrema):
            extrema[i] = rot.apply(pt)

        maxima = np.max(extrema, axis=0) + self._center
        minima = np.min(extrema, axis=0) + self._center
        return minima, maxima

    def get_volume(self):
        h = np.linalg.norm(self._height_vector - self._center)
        return h * self._side_length_1 * self._side_length_2

    def translate(self, t_vector: numpy.ndarray[float]):
        self._center += t_vector
        self._height_vector += t_vector
        self._mbb_min += t_vector
        self._mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        # self._center = rot.apply(self._center)
        self._height_vector = rot.apply(self._height_vector)

    def generate_point_cloud(self, npoints: int):
        # Generate a unit cuboid whose base rectangle has the barycenter in the origin
        cloud = np.full((npoints, 3), 0, dtype=float)
        rand = np.random.rand(npoints, 3)
        rand[:, 0] = rand[:, 0] - 0.5
        rand[:, 1] = rand[:, 1] - 0.5

        # Scale the sides of the cuboid
        height = np.linalg.norm(self._height_vector - self._center)
        rand[:, 0] = rand[:, 0] * self._side_length_1 / 2.0
        rand[:, 1] = rand[:, 1] * self._side_length_2 / 2.0
        rand[:, 2] = rand[:, 2] * height

        # Rotate the cuboid
        hv = (self._height_vector - self._center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(rand)

        # Translate the cuboid
        cloud = cloud + self._center
        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Translate back to origin
        pts = points - self._center

        # Rotate back to xyz
        height = np.linalg.norm(self._height_vector - self._center)
        hv = (self._height_vector - self._center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = -np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        rot_pts = rot.apply(pts)

        # Check for intersections
        inside_points = (
            (rot_pts[:, 2] < height)
            & (rot_pts[:, 2] > 0.0)
            & (rot_pts[:, 0] < self._side_length_1)
            & (rot_pts[:, 0] > -self._side_length_1)
            & (rot_pts[:, 1] < self._side_length_2)
            & (rot_pts[:, 1] > -self._side_length_2)
        )
        return inside_points

    def wireframe_points(self):
        a = self._side_length_1 / 2.0
        b = self._side_length_2 / 2.0
        c = np.linalg.norm(self._height_vector - self._center)

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
        hv = (self._height_vector - self._center) / c
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self._center)

        return x, y, z

    # -----------------------------------------------------


class Parallelepiped(GeometricShape):
    """
    A generic parallelepiped, described by the vectors (following the right-hand orientation) of the
    sides in cartesian coordinates
    """

    def __init__(
        self,
        center: numpy.ndarray[float],
        side_vector_1: float,
        side_vector_2: float,
        side_vector_3: float,
        epsilon=1e-3,
    ):
        """
        :param numpy.ndarray[float] center: The coordinates of the left-bottom edge.
        :param float side_vector_1: The first vector identifying the parallelepiped (using the
            right-hand orientation: the thumb).
        :param float side_vector_2: The second vector identifying the parallelepiped (using the
            right-hand orientation: the index).
        :param float side_vector_3: The third vector identifying the parallelepiped (using the
            right-hand orientation: the middle finger).
        :param float epsilon: Tolerance value to compare coordinates.
        """

        self._center = copy.deepcopy(center)
        self._side_vector_1 = copy.deepcopy(side_vector_1)
        self._side_vector_2 = copy.deepcopy(side_vector_2)
        self._side_vector_3 = copy.deepcopy(side_vector_3)
        # Position of the apex
        self._epsilon = epsilon

        super().__init__()

    def find_mbb(self):
        extrema = np.vstack(
            [
                np.array([0.0, 0.0, 0.0]),
                self._side_vector_1 + self._side_vector_2 + self._side_vector_3,
            ]
        )
        print(extrema)
        maxima = np.max(extrema, axis=0) + self._center
        minima = np.min(extrema, axis=0) + self._center
        return minima, maxima

    def get_volume(self):
        vol = np.dot(
            self._side_vector_3, np.cross(self._side_vector_1, self._side_vector_2)
        )
        return vol

    def translate(self, t_vector: numpy.ndarray[float]):
        self._center += t_vector
        self._mbb_min += t_vector
        self._mbb_max += t_vector

    def rotate(self, r_versor: numpy.ndarray[float], angle: float):
        rot = R.from_rotvec(r_versor * angle)
        # self._center = rot.apply(self._center)
        self._side_vector_1 = rot.apply(self._side_vector_1)
        self._side_vector_2 = rot.apply(self._side_vector_2)
        self._side_vector_3 = rot.apply(self._side_vector_3)

    def generate_point_cloud(self, npoints: int):
        # Generate a linear combination of points in the volume
        cloud = np.full((npoints, 3), 0, dtype=float)
        rand = np.random.rand(npoints, 3)
        for i in range(npoints):
            cloud[i] = (
                rand[i, 0] * self._side_vector_1
                + rand[i, 1] * self._side_vector_2
                + rand[i, 2] * self._side_vector_3
            )
        cloud = cloud + self._center
        return cloud

    def check_inside(self, points: numpy.ndarray[float]):
        # Translate back to origin
        pts = points - self._center

        # Rotate back to xyz
        height = np.linalg.norm(self._height_vector - self._center)
        hv = (self._height_vector - self._center) / height
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = -np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        rot_pts = rot.apply(pts)

        # Compute the Fourier components wrt to the vectors identifying the parallelepiped

        comp1 = rot_pts.dot(self._side_vector_1)
        comp2 = rot_pts.dot(self._side_vector_2)
        comp3 = rot_pts.dot(self._side_vector_3)

        # The points are inside the parallelepiped if and only if all the Fourier components
        # are in (0,1)
        inside_points = (
            (comp1 > 0.0)
            & (comp1 < 1.0)
            & (comp2 > 0.0)
            & (comp2 < 1.0)
            & (comp3 > 0.0)
            & (comp3 < 1.0)
        )
        return inside_points

    def wireframe_points(self):
        va = self._side_vector_1
        vb = self._side_vector_2
        vc = self._side_vector_3

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

        """# Rotate the cuboid
        hv = (self._height_vector - self._center) / c
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
 
        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self._center)"""
        print(x)
        return x, y, z
