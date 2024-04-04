import unittest

import numpy as np
from bsb_test import NumpyTestCase

from bsb import (
    Cone,
    Cuboid,
    Cylinder,
    Ellipsoid,
    Parallelepiped,
    ShapesComposition,
    Sphere,
)
from bsb.exceptions import RequirementError


class TestGeometricShapes(unittest.TestCase, NumpyTestCase):
    def _check_points_inside(self, sc, volume, voxel_size):
        self.assertClose(np.sum(sc.get_volumes()), volume)
        expected_number_of_points = int(volume / voxel_size**3)
        point_cloud = sc.generate_point_cloud()
        npoints = len(point_cloud)

        # Check the number of points in the point cloud
        self.assertEqual(
            npoints,
            expected_number_of_points,
            "The number of point in the point cloud is not the expected one",
        )

        # Check if the point cloud is inside the sphere
        points_inside_sphere = sc.inside_shapes(point_cloud)
        all_points_inside = np.all(points_inside_sphere)
        self.assertEqual(
            all_points_inside,
            True,
            "The point cloud should be inside the ShapeComposition",
        )

    def _check_translation(self, sc, expected_mbb):
        # Check translation
        translation_vec = np.array([1.0, 10.0, 100.0])
        sc.translate(translation_vec)
        mbb = sc.find_mbb()
        expected_mbb += translation_vec
        self.assertClose(
            mbb[0],
            expected_mbb[0],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        self.assertClose(
            mbb[1],
            expected_mbb[1],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        sc.translate(-translation_vec)
        expected_mbb -= translation_vec

    # Create a sphere, add it to a ShapeComposition object and test the minimal bounding box, inside_mbox, inside_shapes and generate_point_cloud methods
    def test_sphere(self):
        # Create a ShapesComposition object; In this test the size of the voxel is not important.
        conf = dict(voxel_size=25, shapes=[], labels=[])
        sc = ShapesComposition(conf)
        self.assertEqual(None, sc.generate_point_cloud())
        self.assertEqual(None, sc.inside_shapes(np.array([[0.0, 0.0, 0.0]])))
        self.assertEqual(None, sc.generate_wireframe())
        self.assertEqual([], sc.get_volumes())

        # Add the sphere to the ShapesComposition object
        radius = 100.0
        origin = np.array([0, 0, 0], dtype=np.float64)
        configuration = dict(radius=radius, origin=origin)
        sc.add_shape(Sphere(configuration), ["sphere"])

        # Find the mmb
        mbb = sc.find_mbb()
        expected_mbb = np.array(
            [[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]], dtype=np.float64
        )

        # If the result is correct the mmb is the box individuated by
        # the opposite vertices [-100., -100.,   0.] and [100., 100., 100.].
        # The tuple must be in the correct order.
        self.assertClose(
            mbb[0],
            expected_mbb[0],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        self.assertClose(
            mbb[1],
            expected_mbb[1],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )

        # The point of coordinates (0,0,50) is inside the sphere, while (200,200,200) is not.
        # We test check_mbox, check_inside with these two points
        point_to_check = np.array([[0, 0, 50], [200, 200, 200]], dtype=np.float64)
        inside_mbox = sc.inside_mbox(point_to_check)
        inside_shape = sc.inside_shapes(point_to_check)

        # Check if the points are inside the mbb
        expected_inside_mbox = [True, False]
        self.assertEqual(
            inside_mbox[0],
            expected_inside_mbox[0],
            "The point (0,0,50) should be inside the minimal bounding box",
        )
        self.assertEqual(
            inside_mbox[1],
            expected_inside_mbox[1],
            "The point (200,200,200) should be outside the minimal bounding box",
        )

        # Check if the points are inside the sphere
        expected_inside_shape = [True, False]
        self.assertEqual(
            inside_shape[0],
            expected_inside_shape[0],
            "The point (0,0,50) should be inside the sphere",
        )
        self.assertEqual(
            inside_shape[1],
            expected_inside_shape[1],
            "The point (0,0,-50) should be outside the sphere",
        )

        # Test generate_point_cloud method.
        # The expected number of points is given by the volume of the sphere divided by the voxel side to the third
        # The points should be inside the sphere.
        volume = 4 * (np.pi * configuration["radius"] ** 3) / 3.0
        self._check_points_inside(sc, volume, conf["voxel_size"])
        self._check_translation(sc, expected_mbb)
        wireframe = np.array(sc.generate_wireframe())
        self.assertEqual((3, 1, 30, 30), wireframe.shape)
        self.assertTrue(
            np.allclose(np.linalg.norm(wireframe[:, 0, :, 0].T - origin, axis=1), radius)
        )

    # Create an ellipsoid, add it to a ShapeComposition object and test the minimal bounding box, inside_mbox, inside_shapes and generate_point_cloud methods
    def test_ellipsoid(self):
        # Create a ShapesComposition object; In this test the size of the voxel is not important.
        conf = dict(voxel_size=25, shapes=[], labels=[])
        sc = ShapesComposition(conf)

        # Add the ellipsoid to the ShapesComposition object
        configuration = dict(
            origin=np.array([0, 0, 0], dtype=np.float64),
            lambdas=np.array([50, 100, 10], dtype=np.float64),
            v0=np.array([1, 0, 0], dtype=np.float64),
            v1=np.array([0, 1, 0], dtype=np.float64),
            v2=np.array([0, 0, 1], dtype=np.float64),
        )
        ellipsoid = Ellipsoid(configuration)
        sc.add_shape(ellipsoid, ["ellipsoid"])

        # Find the mmb
        mbb = sc.find_mbb()
        expected_mbb = np.array(
            [[-50.0, -100.0, -10.0], [50.0, 100.0, 10.0]], dtype=np.float64
        )

        # If the result is correct the mmb is the box individuated by
        # the opposite vertices [-100., -100.,   -10.] and [100., 100., 10.].
        # The tuple must be in the correct order.
        self.assertClose(
            mbb[0],
            expected_mbb[0],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        self.assertClose(
            mbb[1],
            expected_mbb[1],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )

        # The point of coordinates (0,0,5) is inside the ellipsoid, while (20,20,20) is not.
        # We test check_mbox, check_inside with these two points
        point_to_check = np.array([[0, 0, 5], [20, 20, 20]], dtype=np.float64)
        inside_mbox = sc.inside_mbox(point_to_check)
        inside_shape = sc.inside_shapes(point_to_check)

        # Check if the points are inside the mbb
        expected_inside_mbox = [True, False]
        self.assertEqual(
            inside_mbox[0],
            expected_inside_mbox[0],
            "The point (0,0,50) should be inside the minimal bounding box",
        )
        self.assertEqual(
            inside_mbox[1],
            expected_inside_mbox[1],
            "The point (200,200,200) should be outside the minimal bounding box",
        )

        # Check if the points are inside the ellipsoid
        expected_inside_shape = [True, False]
        self.assertEqual(
            inside_shape[0],
            expected_inside_shape[0],
            "The point (0,0,50) should be inside the ellipsoid",
        )
        self.assertEqual(
            inside_shape[1],
            expected_inside_shape[1],
            "The point (0,0,-50) should be outside the ellipsoid",
        )

        # Test generate_point_cloud method.
        # The expected number of points is given by the volume of the ellipsoid divided by the voxel side to the third
        # The points should be inside the ellipsoid.
        volume = (
            np.pi
            * configuration["lambdas"][0]
            * configuration["lambdas"][1]
            * configuration["lambdas"][2]
        )
        self._check_points_inside(sc, volume, conf["voxel_size"])
        self._check_translation(sc, expected_mbb)

        # Check rotation
        ellipsoid.rotate(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        mbb = ellipsoid.find_mbb()
        self.assertClose(mbb[0], expected_mbb[0, [1, 0, 2]])
        self.assertClose(mbb[1], expected_mbb[1, [1, 0, 2]])

        wireframe = np.array(sc.generate_wireframe())
        self.assertEqual((3, 1, 30, 30), wireframe.shape)
        for coord in (wireframe[:, 0, :, 0] - ellipsoid.origin[..., np.newaxis]).T:
            self.assertTrue(-1e-5 <= coord[0] <= 1e5)
            self.assertTrue(-1e-5 <= coord[1] <= 50)
            self.assertTrue(-10 <= coord[2] <= 10)

    # Create a cylinder, add it to a ShapeComposition object and test the minimal bounding box, inside_mbox, inside_shapes and generate_point_cloud methods
    def test_cylinder(self):
        # Create a ShapesComposition object; In this test the size of the voxel is not important.
        conf = dict(voxel_size=25, shapes=[], labels=[])
        sc = ShapesComposition(conf)

        # Add the cylinder to the ShapesComposition object
        radius = 100.0
        origin = np.array([0, 0, 0], dtype=np.float64)
        top_center = np.array([0, 0, 10], dtype=np.float64)

        configuration = dict(
            radius=radius,
            origin=origin,
            top_center=top_center,
        )
        cylinder = Cylinder(configuration)
        sc.add_shape(cylinder, ["cylinder"])

        # Find the mmb
        mbb = sc.find_mbb()
        expected_mbb = np.array(
            [[-100.0, -100.0, 0.0], [100.0, 100.0, 10.0]], dtype=np.float64
        )

        # If the result is correct the mmb is the box individuated by
        # the opposite vertices [-100., -100.,   0.] and [100., 100., 100.].
        # The tuple must be in the correct order.
        self.assertClose(
            mbb[0],
            expected_mbb[0],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        self.assertClose(
            mbb[1],
            expected_mbb[1],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )

        # The point of coordinates (0,0,5) is inside the cylinder, while (200,200,200) is not.
        # We test check_mbox, check_inside with these two points
        point_to_check = np.array([[0, 0, 5], [200, 200, 200]], dtype=np.float64)
        inside_mbox = sc.inside_mbox(point_to_check)
        inside_shape = sc.inside_shapes(point_to_check)

        # Check if the points are inside the mbb
        expected_inside_mbox = [True, False]
        self.assertEqual(
            inside_mbox[0],
            expected_inside_mbox[0],
            "The point (0,0,50) should be inside the minimal bounding box",
        )
        self.assertEqual(
            inside_mbox[1],
            expected_inside_mbox[1],
            "The point (200,200,200) should be outside the minimal bounding box",
        )

        # Check if the points are inside the cylinder
        expected_inside_shape = [True, False]
        self.assertEqual(
            inside_shape[0],
            expected_inside_shape[0],
            "The point (0,0,50) should be inside the cylinder",
        )
        self.assertEqual(
            inside_shape[1],
            expected_inside_shape[1],
            "The point (0,0,-50) should be outside the cylinder",
        )

        # Test generate_point_cloud method.
        # The expected number of points is given by the volume of the cylinder divided by the voxel side to the third
        # The points should be inside the cylinder.
        height = np.linalg.norm(configuration["top_center"] - configuration["origin"])
        volume = np.pi * height * configuration["radius"] ** 2
        self._check_points_inside(sc, volume, conf["voxel_size"])
        self._check_translation(sc, expected_mbb)

        # Check rotation
        cylinder.rotate(np.array([1.0, 0.0, 0.0]), np.pi / 2)
        mbb = cylinder.find_mbb()
        self.assertClose(mbb[0], [-100.0, -10.0, -100.0])
        self.assertClose(mbb[1], [100.0, 0.0, 100.0])

        wireframe = np.array(sc.generate_wireframe())
        self.assertEqual((3, 1, 30, 30), wireframe.shape)
        self.assertTrue(
            np.allclose(
                np.linalg.norm(
                    (wireframe[:, 0, :, 0] - cylinder.origin[..., np.newaxis])[
                        np.array([0, 2])
                    ],
                    axis=0,
                ),
                radius,
            )
        )
        for p in wireframe[:, 0, :, 0].T:
            self.assertTrue(
                0
                <= np.absolute(p[1] - cylinder.origin[1])
                <= np.absolute(cylinder.top_center[1] - cylinder.origin[1])
            )

    # Create a parallelepiped, add it to a ShapeComposition object and test the minimal bounding box, inside_mbox, inside_shapes and generate_point_cloud methods
    def test_parallelepiped(self):
        # Create a ShapesComposition object; In this test the size of the voxel is not important.
        conf = dict(voxel_size=5, shapes=[], labels=[])
        sc = ShapesComposition(conf)

        # Add the parallelepiped to the ShapesComposition object
        configuration = dict(
            origin=np.array([-5, -5, -5], dtype=np.float64),
            side_vector_1=np.array([10, 0, 0], dtype=np.float64),
            side_vector_2=np.array([0, 100, 0], dtype=np.float64),
            side_vector_3=np.array([0, 0, 10], dtype=np.float64),
        )
        parallelepiped = Parallelepiped(configuration)
        sc.add_shape(
            parallelepiped,
            ["parallelepiped"],
        )

        # Find the mmb
        mbb = sc.find_mbb()
        expected_mbb = np.array([[-5.0, -5.0, -5.0], [5.0, 95.0, 5.0]], dtype=np.float64)

        # If the result is correct the mmb is the box individuated by
        # the opposite vertices [-5., -5.,   -5.] and [5., 5., 5.].
        # The tuple must be in the correct order.
        self.assertClose(
            mbb[0],
            expected_mbb[0],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        self.assertClose(
            mbb[1],
            expected_mbb[1],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )

        # The point of coordinates (0,0,0) is inside the parallelepiped, while (10,10,10) is not.
        # We test check_mbox, check_inside with these two points
        point_to_check = np.array([[0, 0, 0], [10, 100, 10]], dtype=np.float64)
        inside_mbox = sc.inside_mbox(point_to_check)
        inside_shape = sc.inside_shapes(point_to_check)

        # Check if the points are inside the mbb
        expected_inside_mbox = [True, False]
        self.assertEqual(
            inside_mbox[0],
            expected_inside_mbox[0],
            "The point (0,0,0) should be inside the minimal bounding box",
        )
        self.assertEqual(
            inside_mbox[1],
            expected_inside_mbox[1],
            "The point (10,10,10) should be outside the minimal bounding box",
        )

        # Check if the points are inside the parallelepiped
        expected_inside_shape = [True, False]
        self.assertEqual(
            inside_shape[0],
            expected_inside_shape[0],
            "The point (0,0,0) should be inside the parallelepiped",
        )
        self.assertEqual(
            inside_shape[1],
            expected_inside_shape[1],
            "The point (10,100,10) should be outside the parallelepiped",
        )

        # Test generate_point_cloud method.
        # The expected number of points is given by the volume of the parallelepiped divided by the voxel side to the third
        # The points should be inside the parallelepiped.
        volume = (
            np.linalg.norm(configuration["side_vector_1"])
            * np.linalg.norm(configuration["side_vector_2"])
            * np.linalg.norm(configuration["side_vector_3"])
        )
        self._check_points_inside(sc, volume, conf["voxel_size"])
        self._check_translation(sc, expected_mbb)

        # Check rotation
        parallelepiped.rotate(np.array([1.0, 0.0, 0.0]), np.pi / 2)
        mbb = parallelepiped.find_mbb()
        expected_mbb = np.array(
            [[-5.0, -15.0, -5.0], [5.0, -5.0, 95.0]], dtype=np.float64
        )
        self.assertClose(mbb[0], expected_mbb[0])
        self.assertClose(mbb[1], expected_mbb[1])
        wireframe = np.array(sc.generate_wireframe())
        self.assertEqual((3, 1, 4, 4), wireframe.shape)
        self.assertTrue(np.alltrue(wireframe.reshape(3, 16).T - expected_mbb[0] >= -1e-5))
        self.assertTrue(np.alltrue(wireframe.reshape(3, 16).T - expected_mbb[1] <= 1e-5))

    # Create a cuboid, add it to a ShapeComposition object and test the minimal bounding box, inside_mbox, inside_shapes and generate_point_cloud methods
    def test_cuboid(self):
        # Create a ShapesComposition object; In this test the size of the voxel is not important.
        conf = dict(voxel_size=25, shapes=[], labels=[])
        sc = ShapesComposition(conf)

        # Add the cuboid to the ShapesComposition object
        configuration = dict(
            origin=np.array([0, 0, 0], dtype=np.float64),
            side_length_1=5.0,
            side_length_2=10.0,
            top_center=np.array([0, 0, 20], dtype=np.float64),
        )
        cuboid = Cuboid(configuration)
        sc.add_shape(cuboid, ["cuboid"])

        # Find the mmb
        mbb = sc.find_mbb()
        expected_mbb = np.array([[-2.5, -5.0, 0.0], [2.5, 5.0, 20.0]], dtype=np.float64)

        # If the result is correct the mmb is the box individuated by
        # the opposite vertices [-5., -5.,   -5.] and [5., 5., 5.].
        # The tuple must be in the correct order.
        self.assertClose(
            mbb[0],
            expected_mbb[0],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        self.assertClose(
            mbb[1],
            expected_mbb[1],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )

        # The point of coordinates (1,1,1) is inside the cuboid, while (10,10,10) is not.
        # We test check_mbox, check_inside with these two points
        point_to_check = np.array([[1, 1, 1], [10, 10, 10]], dtype=np.float64)
        inside_mbox = sc.inside_mbox(point_to_check)
        inside_shape = sc.inside_shapes(point_to_check)

        # Check if the points are inside the mbb
        expected_inside_mbox = [True, False]
        self.assertEqual(
            inside_mbox[0],
            expected_inside_mbox[0],
            "The point (0,0,0) should be inside the minimal bounding box",
        )
        self.assertEqual(
            inside_mbox[1],
            expected_inside_mbox[1],
            "The point (10,10,10) should be outside the minimal bounding box",
        )

        # Check if the points are inside the cuboid
        expected_inside_shape = [True, False]
        self.assertEqual(
            inside_shape[0],
            expected_inside_shape[0],
            "The point (0,0,0) should be inside the cuboid",
        )
        self.assertEqual(
            inside_shape[1],
            expected_inside_shape[1],
            "The point (10,10,10) should be outside the cuboid",
        )

        # Test generate_point_cloud method.
        # The expected number of points is given by the volume of the cuboid divided by the voxel side to the third
        # The points should be inside the cuboid.
        volume = (
            configuration["side_length_1"]
            * configuration["side_length_2"]
            * np.linalg.norm(configuration["top_center"])
        )
        self._check_points_inside(sc, volume, conf["voxel_size"])
        self._check_translation(sc, expected_mbb)

        # Check rotation
        expected_mbb = np.array([[-2.5, -5.0, 0.0], [2.5, 5.0, 20.0]], dtype=np.float64)
        cuboid.rotate(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        mbb = cuboid.find_mbb()
        self.assertClose(mbb[0], expected_mbb[0])
        self.assertClose(mbb[1], expected_mbb[1])
        wireframe = np.array(sc.generate_wireframe())
        self.assertEqual((3, 1, 4, 4), wireframe.shape)
        self.assertTrue(np.alltrue(wireframe.reshape(3, 16).T - expected_mbb[0] >= -1e-5))
        self.assertTrue(np.alltrue(wireframe.reshape(3, 16).T - expected_mbb[1] <= 1e-5))

    # Create a cone, add it to a ShapeComposition object and test the minimal bounding box, inside_mbox, inside_shapes and generate_point_cloud methods
    def test_cone(self):
        # Create a ShapesComposition object; In this test the size of the voxel is not important.
        conf = dict(voxel_size=50, shapes=[], labels=[])
        sc = ShapesComposition(conf)

        # Add the cone to the ShapesComposition object
        radius = 100.0
        origin = np.array([0, 0, 100], dtype=np.float64)
        apex = np.array([0, 0, 0], dtype=np.float64)
        configuration = {
            "origin": origin,
            "radius": radius,
            "apex": apex,
        }
        cone = Cone(configuration)
        sc.add_shape(cone, ["cone"])

        # Find the mmb
        mbb = sc.find_mbb()
        expected_mbb = np.array(
            [[-100.0, -100.0, 0.0], [100.0, 100.0, 100.0]], dtype=np.float64
        )

        # If the result is correct the mmb is the box individuated by
        # the opposite vertices [-100., -100.,   0.] and [100., 100., 100.].
        # The tuple must be in the correct order.
        self.assertClose(
            mbb[0],
            expected_mbb[0],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        self.assertClose(
            mbb[1],
            expected_mbb[1],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )

        # The point of coordinates (0,0,50) is inside the cone, while (0,0,-50) is not.
        # We test check_mbox, check_inside with these two points
        point_to_check = np.array([[0, 0, 50], [0, 0, -50]], dtype=np.float64)
        inside_mbox = sc.inside_mbox(point_to_check)
        inside_shape = sc.inside_shapes(point_to_check)

        # Check if the points are inside the mbb
        expected_inside_mbox = [True, False]
        self.assertEqual(
            inside_mbox[0],
            expected_inside_mbox[0],
            "The point (0,0,50) should be inside the minimal bounding box",
        )
        self.assertEqual(
            inside_mbox[1],
            expected_inside_mbox[1],
            "The point (0,0,-50) should be outside the minimal bounding box",
        )

        # Check if the points are inside the cone
        expected_inside_shape = [True, False]
        self.assertEqual(
            inside_shape[0],
            expected_inside_shape[0],
            "The point (0,0,50) should be inside the cone",
        )
        self.assertEqual(
            inside_shape[1],
            expected_inside_shape[1],
            "The point (0,0,-50) should be outside the cone",
        )

        # Test generate_point_cloud method.
        # The expected number of points is given by the volume of the cone divided by the voxel side to the third
        # The points should be inside the cone.
        cone_height = np.linalg.norm(configuration["origin"] - configuration["apex"])
        volume = (np.pi * cone_height * configuration["radius"] ** 2) / 3.0
        self._check_points_inside(sc, volume, conf["voxel_size"])
        self._check_translation(sc, expected_mbb)

        # Check rotation
        expected_mbb = np.array(
            [[-100.0, -100.0, 0.0], [100.0, 100.0, 100.0]], dtype=np.float64
        )
        cone.rotate(np.array([0.0, 1.0, 0.0]), np.pi / 2)
        mbb = cone.find_mbb()
        self.assertClose(mbb[0], expected_mbb[0])
        self.assertClose(mbb[1], expected_mbb[1])

        wireframe = np.array(sc.generate_wireframe())
        self.assertEqual((3, 1, 30, 30), wireframe.shape)
        self.assertTrue(
            np.alltrue(
                np.linalg.norm(
                    (wireframe[:, 0, :, 0] - cone.origin[..., np.newaxis])[
                        np.array([0, 1])
                    ],
                    axis=0,
                )
                - radius
                <= 1e-5
            )
        )
        for p in wireframe[:, 0, :, 0].T:
            self.assertTrue(
                0
                <= np.absolute(p[2] - cone.origin[2])
                <= np.absolute(cone.apex[2] - cone.origin[2])
            )

    # Create ShapeComposition object, add a sphere and a cylinder and then test
    def test_shape_composition(self):
        config_sphere = dict(radius=10.0, origin=np.array([0, 0, 0], dtype=np.float64))
        config_cylinder = dict(
            top_center=np.array([0, 0, 0], dtype=np.float64),
            radius=25.0,
            origin=np.array([0, 0, -40], dtype=np.float64),
        )

        with self.assertRaises(RequirementError):
            ShapesComposition(
                dict(
                    shapes=[
                        dict(
                            type="sphere",
                            radius=10.0,
                            center=np.array([0, 0, 0], dtype=np.float64),
                        )
                    ],
                    labels=[["label1"], ["label2"]],
                )
            )

        with self.assertRaises(RequirementError):
            ShapesComposition(dict(shapes=[], labels=[["label1"], ["label2"]]))

        # Create a ShapesComposition object
        conf = dict(voxel_size=10, shapes=[], labels=[])
        sc = ShapesComposition(conf)

        # Build a sphere
        sc.add_shape(Sphere(config_sphere), ["sphere"])

        # Build a cylinder
        sc.add_shape(Cylinder(config_cylinder), ["cylinder"])

        # Check if shapes filtering by labels works
        filtered_shape = sc.filter_by_labels(["sphere"])
        self.assertIsInstance(filtered_shape._shapes[0], Sphere)

        # Test the mininimal bounding box of the composition
        # The expected mbb is [-25,-25,-40], [25,25,10]
        mbb = sc.find_mbb()
        expected_mbb = (
            np.array([-25.0, -25.0, -40.0], dtype=np.float64),
            np.array([25.0, 25.0, 10.0], dtype=np.float64),
        )

        self.assertClose(
            mbb[0],
            expected_mbb[0],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )
        self.assertClose(
            mbb[1],
            expected_mbb[1],
            "The minimal bounding box returned by find_mbb method is not the expected one",
        )

        # Test generate_point_cloud method for the composition of shapes.
        # The expected number of points is given by the sum of the volumes the voxel side to the third
        # The points should be inside the cone.
        sphere_volume = 4.0 / 3.0 * np.pi * config_sphere["radius"] ** 3
        cylinder_volume = (
            np.pi
            * np.linalg.norm(config_cylinder["origin"])
            * config_cylinder["radius"] ** 2
        )
        total_volume = sphere_volume + cylinder_volume
        self._check_points_inside(sc, total_volume, conf["voxel_size"])
        self._check_translation(sc, expected_mbb)
