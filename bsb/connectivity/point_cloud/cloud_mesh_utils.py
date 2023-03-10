import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_3d_mesh_by_vec(
    x: np.array, y: np.array, z: np.array, rot_versor: np.array, angle: float
):
    # Arrange point coordinates in shape (N, 3) for vectorized processing
    pts = np.array([x.ravel(), y.ravel(), z.ravel()]).transpose()

    # Create and apply rotation
    rot = R.from_rotvec(rot_versor * angle)
    rot_pts = rot.apply(pts)

    # return to original shape of meshgrid
    xrot = rot_pts[:, 0].reshape(x.shape)
    yrot = rot_pts[:, 1].reshape(y.shape)
    zrot = rot_pts[:, 2].reshape(z.shape)

    return xrot, yrot, zrot


def translate_3d_mesh_by_vec(x: np.array, y: np.array, z: np.array, t_vec: np.array):
    # Arrange point coordinates in shape (N, 3) for vectorized processing
    pts = np.array([x.ravel(), y.ravel(), z.ravel()]).transpose()

    pts = pts + t_vec

    # return to original shape of meshgrid
    xt = pts[:, 0].reshape(x.shape)
    yt = pts[:, 1].reshape(y.shape)
    zt = pts[:, 2].reshape(z.shape)

    return xt, yt, zt


def rotate_3d_mesh_by_rot_mat(x: np.array, y: np.array, z: np.array, rot_mat: np.array):
    # Arrange point coordinates in shape (N, 3) for vectorized processing
    pts = np.array([x.ravel(), y.ravel(), z.ravel()]).transpose()

    # Create and apply rotation
    rot = R.from_matrix(rot_mat)
    rot_pts = rot.apply(pts)

    # return to original shape of meshgrid
    xrot = rot_pts[:, 0].reshape(x.shape)
    yrot = rot_pts[:, 1].reshape(y.shape)
    zrot = rot_pts[:, 2].reshape(z.shape)

    return xrot, yrot, zrot
