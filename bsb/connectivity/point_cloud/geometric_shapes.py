import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pickle
import abc
import plotly.graph_objects as go
import copy
from .cloud_mesh_utils import (
    rotate_3d_mesh_by_vec,
    translate_3d_mesh_by_vec,
    rotate_3d_mesh_by_rot_mat,
)

# Base class for geometric shapes
class GeometricShape(abc.ABC):
    @abc.abstractmethod
    def get_volume(self):
        pass

    @abc.abstractmethod
    def translate(self, t_vector: np.ndarray):
        pass

    @abc.abstractmethod
    def rotate(self, r_versor: np.ndarray, angle: float):
        pass

    @abc.abstractmethod
    def generate_point_cloud(self, npoints: int):
        pass

    @abc.abstractmethod
    def check_mbox(self, points: np.ndarray):
        pass

    @abc.abstractmethod
    def check_inside(self, points: np.ndarray):
        pass

    @abc.abstractmethod
    def wireframe_points(self):
        pass


# -----------------------------------------------------


class ShapesComposition:
    def __init__(self, voxel_size = 1):
        self.shapes = []
        self.voxel_size = voxel_size
        self.mbb_min = np.array([0.,0.,0.])
        self.mbb_max = np.array([0.,0.,0.])
        self.origin = np.array([0.,0.,0.])

    def copy(self):
        return copy.deepcopy(self)

    def add_shape(self, shape):
        self.shapes.append(shape)

    def translate(self, t_vec):
        for shape in self.shapes:
            shape.translate(t_vec)
            self.origin += t_vec
            self.mbb_min += t_vec
            self.mbb_max += t_vec

    def get_volumes(self):
        volumes = []
        if len(self.shape) != 0:
            for shape in self.shapes:
                volumes.append(shape.get_volume())
        return volumes

    def find_mbb(self):
        box_min = np.empty([3])
        box_max = np.empty([3])

        mins = np.empty([len(self.shapes),3])
        maxs = np.empty([len(self.shapes),3])
        print(len(self.shapes))
        for i,shape in enumerate(self.shapes):
            #min, max = shape.find_mbb()
            mins[i,:] = shape.mbb_min
            maxs[i,:] = shape.mbb_max

        box_min = np.min(mins, axis=0)
        box_max = np.max(maxs, axis=0)
        return box_min, box_max

    def compute_n_points(self):
        npoints = []
        if len(self.shapes) != 0:
            for shape in self.shapes:
                npoints.append(int(shape.get_volume() // self.voxel_size))
        return npoints

    def generate_point_cloud(self):
        if len(self.shapes) != 0:
            cloud = np.empty([3])
            npoints = self.compute_n_points()
            for shape, numpts in zip(self.shapes, npoints):
                tmp = shape.generate_point_cloud(numpts)
                cloud = np.vstack((cloud, tmp))
            return cloud[1:]
        else:
            return None

    def generate_wireframe(self):
        if len(self.shapes) != 0:
            cloud = np.empty([3])
            npoints = self.compute_n_points()
            x = []
            y = []
            z = []
            for shape in self.shapes:
                xt, yt, zt = shape.wireframe_points()
                x.append(xt)
                y.append(yt)
                z.append(zt)
            return x, y, z
        else:
            return None

    def inside_mbox(self, points):

        inside = (
            (points[:, 0] > self.mbb_min[0])
            & (points[:, 0] < self.mbb_max[0])
            & (points[:, 2] > self.mbb_min[1])
            & (points[:, 2] < self.mbb_max[1])
            & (points[:, 1] > self.mbb_min[2])
            & (points[:, 1] < self.mbb_max[2])
        )
        """print(points[0])
        print(self.mbb_min)
        print(self.mbb_max)
        print("-------")"""

        return inside

        
        """if len(self.shapes) != 0:
            cloud = np.full(len(points), 0, dtype=bool)
            for shape in self.shapes:
                cloud = cloud | shape.check_mbox(points)

            return cloud
        else:
            return None"""

    def inside_shapes(self, points):
        if len(self.shapes) != 0:
            cloud = np.full(len(points), 0, dtype=bool)
            for shape in self.shapes:
                cloud = cloud | shape.check_inside(points)
            return cloud
        else:
            return None

    def save_to_file(self, filename: str):
        with open(filename, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_file(self, filename: str):
        tmp = None
        with open(filename, "rb") as handle:
            tmp = pickle.load(handle)
        self.shapes = tmp.shapes
        self.voxel_size = tmp.voxel_size
        self.mbb_min, self.mbb_max = self.find_mbb()

    def plot_cloud(self, npoints):
        to_plot = np.empty([3])
        n_p = self.compute_n_points()
        for shape, n in zip(self.shapes, n_p):
            tmp = shape.generate_point_cloud(n)
            to_plot = np.vstack((to_plot, tmp))

        to_plot = to_plot[1:]
        ax = plt.figure().add_subplot(projection="3d")
        ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], color="red")
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        ax.set_zlabel("$Z$")
        plt.show()
        plt.clf()

    def plot_wireframe(self):
        ax = plt.figure().add_subplot(projection="3d")
        for shape in self.shapes:
            x, y, z = shape.wireframe_points()
            ax.plot_wireframe(x, y, z)
        plt.show()
        plt.clf()


# -----------------------------------------------------


class Ellipsoid(GeometricShape):
    def __init__(
        self,
        center: np.ndarray,
        lambdas: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        epsilon=1.0e-3,
    ):
        self.center = center
        self.lambdas = lambdas
        self.v0 = v0 / np.linalg.norm(v0)
        self.v1 = v1 / np.linalg.norm(v1)
        self.v2 = v2 / np.linalg.norm(v2)
        self.epsilon = epsilon
        self.mbb_min, self.mbb_max = self.find_mbb()

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

    def rotate(self, r_versor: np.ndarray, angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.center = rot.apply(self.center)
        self.v0 = rot.apply(self.v0)
        self.v1 = rot.apply(self.v1)
        self.v2 = rot.apply(self.v2)

    def generate_point_cloud(self, npoints: int):

        # Generate an ellipse orientated along x,y,z
        cloud = np.full((npoints, 3), 0, dtype=float)
        theta = np.pi * 2.0 * np.random.rand(npoints)
        phi = -1.0 * np.pi * np.random.rand(npoints)
        rand = np.random.rand(npoints, 3)

        # Generate an ellipsoid centered at the origin, with the semiaxes on x,y,z
        cloud[:, 0] = self.lambdas[0] * np.cos(theta) * np.sin(phi)
        cloud[:, 1] = self.lambdas[1] * np.sin(theta) * np.sin(phi)
        cloud[:, 2] = self.lambdas[2] * np.cos(phi)
        cloud = cloud * rand

        # Rotate the ellipse
        rmat = np.array([self.v0, self.v1, self.v2]).T
        # rmat = rmat/np.linalg.det(rmat)
        cloud = cloud.dot(rmat)
        cloud = cloud + self.center

        return cloud

    def check_mbox(self, points: np.ndarray):

        # Check for intersections with mbb
        inside = (
            (points[:, 0] > self.mbb_min[0])
            & (points[:, 0] < self.mbb_max[0])
            & (points[:, 1] > self.mbb_min[1])
            & (points[:, 1] < self.mbb_max[1])
            & (points[:, 2] > self.mbb_min[2])
            & (points[:, 2] < self.mbb_max[2])
        )
        return inside

    def check_inside(self, points: np.ndarray):
        # Check if the quadratic form associated to the ellipse is less than 1 at a point
        diff = points - self.center
        vmat = np.array([self.v0, self.v1, self.v2])
        diag = np.diag(1 / self.lambdas**2)
        qmat = (vmat).dot(diag).dot(vmat)
        #print(qmat)
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
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self.center)

        return x, y, z


# -----------------------------------------------------


class Cone(GeometricShape):
    def __init__(
        self, apex: np.ndarray, center: np.ndarray, radius: float, epsilon=1.0e-3
    ):
        self.center = center
        self.radius = radius
        self.apex = apex
        self.epsilon = epsilon

        # Find the minimum bounding box, to avoid computing it every time
        self.mbb_min, self.mbb_max = self.find_mbb()

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

    def rotate(self, r_versor: np.ndarray, angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.center = rot.apply(self.center)
        self.apex = rot.apply(self.apex)

    def generate_point_cloud(self, npoints: int):
        theta = np.pi * 2.0 * np.random.rand(npoints)
        rand_a = np.random.rand(npoints)
        rand_b = np.random.rand(npoints)

        # Height vector
        hv = np.array(self.center) - np.array(self.apex)
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
        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(cloud)

        # Translate the cone
        cloud = cloud + self.apex
        return cloud

    def check_mbox(self, points: np.ndarray):

        """# Center the points to the origin
        pts = points - self.apex

        # Vectors identifying half of the sides of the square in xzy
        u = np.array([self.radius, 0, 0])
        v = np.array([0, self.radius, 0])

        # Find the rotation angle and axis
        hv = self.center - self.apex
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)"""

        """inside = (
            (points[:, 2] > self.mbb_min[2])
            & (points[:, 2] < self.mbb_max[2]))
        if np.any(inside):
            inside = inside & (points[:, 1] > self.mbb_min[1]) & (points[:, 1] < self.mbb_max[1]) & (points[:, 0] > self.mbb_min[0]) & (points[:, 0] < self.mbb_max[0])"""

        #Look for points inside the mbb.
        inside = (
            (points[:, 0] > self.mbb_min[0])
            & (points[:, 0] < self.mbb_max[0])
            & (points[:, 1] > self.mbb_min[1])
            & (points[:, 1] < self.mbb_max[1])
            & (points[:, 2] > self.mbb_min[2])
            & (points[:, 2] < self.mbb_max[2])
            )
        
        return inside

    def check_inside(self, points: np.ndarray):

        # Find the vector w of the height.
        h_vector = self.center - self.apex
        height = np.linalg.norm(h_vector)
        hv = h_vector / height

        # Center the points
        pts = points - self.apex

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
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self.apex)

        return x, y, z


# -----------------------------------------------------


class Cylinder(GeometricShape):
    def __init__(
        self, center: np.ndarray, radius: float, height_vector: np.ndarray, epsilon=1e-3
    ):
        self.center = center
        self.radius = radius
        # Position of the apex
        self.height_vector = height_vector
        self.epsilon = epsilon

        # Find the minimum bounding box, to avoid computing it every time
        self.mbb_min, self.mbb_max = self.find_mbb()

    def find_mbb(self):

        # Find extrema of the mbb
        extrema = np.array(
            [
                self.center + self.radius,
                self.center - self.radius,
                self.height_vector + self.center + self.radius,
                self.height_vector + self.center - self.radius,
            ]
        )
        minima = np.min(extrema, axis=0)
        maxima = np.max(extrema, axis=0)
        return minima, maxima

    def get_volume(self):
        h = np.linalg.norm(self.height_vector)
        b = np.pi * self.radius * self.radius
        return b * h

    def translate(self, t_vector):
        self.center += t_vector
        self.height_vector += t_vector
        self.mbb_min += t_vector
        self.mbb_max += t_vector

    def rotate(self, r_versor: np.ndarray, angle: float):
        rot = R.from_rotvec(r_versor * angle)
        self.center = rot.apply(self.center)
        self.height_vector = rot.apply(self.height_vector)

    def generate_point_cloud(self, npoints: int):
        # Generate an ellipse orientated along x,y,z
        cloud = np.full((npoints, 3), 0, dtype=float)
        theta = np.pi * 2.0 * np.random.rand(npoints)
        rand = np.random.rand(npoints, 3)
        height = np.linalg.norm(self.height_vector - self.center)

        # Generate an ellipsoid centered at the origin, with the semiaxes on x,y,z
        cloud[:, 0] = self.radius * np.cos(theta)
        cloud[:, 1] = self.radius * np.sin(theta)
        cloud[:, 2] = height
        cloud = cloud * rand

        # Rotate the cylinder
        hv = (self.height_vector - self.center) / height
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))
        rot = R.from_rotvec(perp * angle)
        cloud = rot.apply(cloud)
        cloud = cloud + self.center
        return cloud

    def check_mbox(self, points: np.ndarray):

        # Check for intersections with mbb
        
        inside = (
            (points[:, 2] > self.mbb_min[2])
            & (points[:, 2] < self.mbb_max[2]))
        if np.any(inside):
            inside = inside & (points[:, 1] > self.mbb_min[1]) & (points[:, 1] < self.mbb_max[1]) & (points[:, 0] > self.mbb_min[0]) & (points[:, 0] < self.mbb_max[0])
        
                 
        """inside = (
            (points[:, 0] > self.mbb_min[0])
            & (points[:, 0] < self.mbb_max[0])
            & (points[:, 1] > self.mbb_min[1])
            & (points[:, 1] < self.mbb_max[1])
            & (points[:, 2] > self.mbb_min[2])
            & (points[:, 2] < self.mbb_max[2])
        )"""
        return inside

    def check_inside(self, points: np.ndarray):
        # Translate back to origin
        pts = points - self.center

        # Rotate back to xyz
        height = np.linalg.norm(self.height_vector - self.center)
        hv = (self.height_vector - self.center) / height
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
        hv = np.array(self.center) - np.array(self.height_vector)
        height = np.linalg.norm(hv)

        h = np.linspace(0, height, 150)
        theta, h = np.meshgrid(theta, h)

        # Generate a cone with the apex in the origin and the center at (0,0,height)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = h

        # Rotate the cylinder
        hv = (self.height_vector - self.center) / height
        hv = hv / np.linalg.norm(hv)
        zvers = np.array([0, 0, 1])
        perp = np.cross(zvers, hv)
        angle = np.arccos(np.dot(hv, zvers))

        x, y, z = rotate_3d_mesh_by_vec(x, y, z, perp, angle)
        x, y, z = translate_3d_mesh_by_vec(x, y, z, self.center)

        return x, y, z


# -----------------------------------------------------


class Sphere(GeometricShape):
    def __init__(self, center: np.ndarray, radius: float, epsilon=1e-3):
        self.center = center
        self.radius = radius
        self.mbb_min, self.mbb_max = self.find_mbb()
        self.epsilon = epsilon

    def find_mbb(self):
        # Find the minimum bounding box, to avoid computing it every time
        mbb_min = np.array([-self.radius, -self.radius, -self.radius]) + self.center
        mbb_max = np.array([self.radius, self.radius, self.radius]) + self.center
        return mbb_min, mbb_max

    def get_volume(self):
        return np.pi * 4.0 / 3.0 * np.power(self.radius, 3)

    def translate(self, t_vector: np.ndarray):
        self.center += t_vector

    def rotate(self, r_versor: np.ndarray, angle: float):
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

    def check_mbox(self, points: np.ndarray):

        # Check for intersections with mbb
        inside = (
            (points[:, 0] > self.mbb_min[0])
            & (points[:, 0] < self.mbb_max[0])
            & (points[:, 1] > self.mbb_min[1])
            & (points[:, 1] < self.mbb_max[1])
            & (points[:, 2] > self.mbb_min[2])
            & (points[:, 2] < self.mbb_max[2])
        )
        return inside

    def check_inside(self, points: np.ndarray):
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


# -----------------------------------------------------
