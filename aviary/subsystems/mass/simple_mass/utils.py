import jax.numpy as jnp
import openmdao.jax as omj
from scipy.interpolate import CubicSpline


def precompute_airfoil_geometry(num_sections):
    x_points = jnp.linspace(0, 1, num_sections)
    dx = 1 / (num_sections - 1)
    return x_points, dx


def airfoil_thickness(x, max_thickness):
    return (
        5
        * max_thickness
        * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    )


def airfoil_camber_line(x, camber, camber_location):
    camber_location = omj.ks_max(camber_location, 1e-9)  # Divide by zero check
    return jnp.where(
        x < camber_location,
        (camber / camber_location**2) * (2 * camber_location * x - x**2),
        (camber / (1 - camber_location) ** 2)
        * ((1 - 2 * camber_location) + 2 * camber_location * x - x**2),
    )


def extract_airfoil_features(x_coords, y_coords):
    """
    Extract camber, camber location, and max thickness from the given airfoil data.
    This method assumes x_coords are normalized (ranging from 0 to 1).
    """
    # Approximate the camber line and max thickness from the data
    # Assume the camber line is the line of symmetry between the upper and lower surfaces
    upper_surface = y_coords[: int(len(x_coords) // 2)]
    lower_surface = y_coords[int(len(x_coords) // 2) :]
    x_upper = x_coords[: int(len(x_coords) // 2)]
    x_lower = x_coords[int(len(x_coords) // 2) :]

    upper_spline = CubicSpline(x_upper, upper_surface, bc_type='natural')
    lower_spline = CubicSpline(x_lower, lower_surface, bc_type='natural')

    camber_line = (upper_spline(x_coords) + lower_spline(x_coords)) / 2.0

    thickness = upper_spline(x_coords) - lower_spline(x_coords)

    max_thickness_index = omj.ks_max(thickness)
    max_thickness_value = thickness[max_thickness_index]

    camber_slope = jnp.gradient(camber_line, x_coords)
    camber_location_index = omj.ks_max(omj.smooth_abs(camber_slope))
    camber_location = x_coords[camber_location_index]

    camber = camber_line[camber_location_index]

    return camber, camber_location, max_thickness_value, camber_line, thickness
