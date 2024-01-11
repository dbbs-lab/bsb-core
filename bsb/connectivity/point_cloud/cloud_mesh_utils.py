import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


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
    :param Callable[..., np.ndarray[float]] surface_function: function converting polar coordinates
        into cartesian coordinates
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
    nb_points_2,
    surface_function,
    theta_min=0,
    theta_max=2 * np.pi,
    phi_min=0,
    phi_max=np.pi,
    precision=25,
):
    """
    Uniform-like meshgrid of polar coordinates based on surface estimation.
    This sampling is useful on elliptic surfaces (e.g. sphere).
    Algorithm based on https://github.com/maxkapur/param_tools

    :param int n_points: number of points to sample
    :param Callable[..., np.ndarray[float]] surface_function: function converting polar coordinates
        into cartesian coordinates
    :param int precision: size of grid used to estimate function surface
    """

    theta, phi, cum_S_t, cum_S_u = _surface_resampling(
        surface_function, theta_min, theta_max, phi_min, phi_max, precision
    )
    sampled_t = np.linspace(0, cum_S_t[-1], n_points_1)
    sampled_u = np.linspace(0, cum_S_u[-1], nb_points_2)
    sampled_t = interp1d(cum_S_t, theta[0, :])(sampled_t)
    sampled_u = interp1d(cum_S_u, phi[:, 0])(sampled_u)
    sampled_t, sampled_u = np.meshgrid(sampled_t, sampled_u)
    return surface_function(sampled_t, sampled_u)


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
