import os

import jax.numpy as jnp
import numpy as np
import openmdao.api as om
import openmdao.jax as omj
from scipy.interpolate import CubicSpline

from aviary.subsystems.mass.simple_mass.materials_database import materials
from aviary.utils.functions import add_aviary_input, add_aviary_output
from aviary.utils.named_values import get_keys
from aviary.variable_info.variables import Aircraft

try:
    from quadax import quadgk
except ImportError:
    raise ImportError(
        "quadax package not found. You can install it by running 'pip install quadax'."
    )


Debug = False


class WingMass(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', types=int, default=10)

        self.options.declare(
            'airfoil_type', types=str, default='2412'
        )  # use 2412 as example for default

        self.options.declare('material', default='Balsa', values=list(get_keys(materials)))

        self.options.declare(
            'airfoil_data_file', default=None, types=str
        )  # For user-provided airfoil data file

    def setup(self):
        self.options['use_jit'] = not (Debug)

        # Inputs
        add_aviary_input(self, Aircraft.Wing.SPAN, units='m')  # Full wingspan (adjustable)

        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='m')  # Root chord length

        self.add_input('tip_chord', val=1.0, units='m')  # Tip chord length -- no aviary input

        self.add_input(
            'twist', val=jnp.zeros(self.options['num_sections']), units='deg'
        )  # Twist angles -- no aviary input

        self.add_input(
            'thickness_dist',
            val=jnp.ones(self.options['num_sections']) * 0.1,
            shape=(self.options['num_sections'],),
            units='m',
        )  # Thickness distribution of the wing (height) -- no aviary input

        # Outputs
        add_aviary_output(self, Aircraft.Wing.MASS, units='kg')

        if airfoil_data_file and os.path.exists(airfoil_data_file):
            airfoil_data = np.loadtxt(airfoil_data_file)
            x_coords = airfoil_data[:, 0]
            y_coords = airfoil_data[:, 1]

            (
                self.camber,
                self.camber_location,
                self.max_thickness,
                self.thickness,
                self.camber_line,
            ) = self.extract_airfoil_features(x_coords, y_coords)
            len(x_coords)
        else:
            # Parse the NACA airfoil type (4-digit)
            self.camber = int(airfoil_type[0]) / 100.0  # Maximum camber
            self.camber_location = int(airfoil_type[1]) / 10.0  # Location of max camber
            self.max_thickness = int(airfoil_type[2:4]) / 100.0  # Max thickness
            self.options['num_sections']

    def get_self_statics(self):
        return (
            self.camber,
            self.camber_location,
            self.max_thickness,
            self.thickness,
            self.camber_line,
            self.options['material'],
        )

    def compute_primal(
        self, aircraft__wing__span, aircraft__wing__root_chord, tip_chord, twist, thickness_dist
    ):
        material = self.options['material']  # Material is taken from options
        airfoil_type = self.options['airfoil_type']  # NACA airfoil type
        airfoil_data_file = self.options['airfoil_data_file']

        # Get material density
        density = materials.get_val(material, 'kg/m**3')

        if airfoil_data_file and os.path.exists(airfoil_data_file):
            airfoil_data = np.loadtxt(airfoil_data_file)
            x_coords = airfoil_data[:, 0]
            y_coords = airfoil_data[:, 1]

            camber, camber_location, max_thickness, thickness, camber_line = (
                self.extract_airfoil_features(x_coords, y_coords)
            )
            thickness_dist = thickness
            num_sections = len(x_coords)
        else:
            # Parse the NACA airfoil type (4-digit)
            camber = int(airfoil_type[0]) / 100.0  # Maximum camber
            camber_location = int(airfoil_type[1]) / 10.0  # Location of max camber
            max_thickness = int(airfoil_type[2:4]) / 100.0  # Max thickness
            num_sections = self.options['num_sections']

        # Wing spanwise distribution
        jnp.linspace(0, aircraft__wing__span, num_sections)

        n_points = num_sections
        jnp.linspace(0, 1, n_points)
        1 / (n_points - 1)

        if airfoil_type:
            aircraft__wing__mass_first_part, _ = quadgk(
                lambda x: density
                * 2
                * jnp.atleast_1d(self.airfoil_thickness(x, max_thickness))
                * jnp.sqrt(
                    1 + ((camber / camber_location**2) * (2 * camber_location - 2 * x)) ** 2
                ),
                [0, camber_location],
                epsabs=1e-9,
                epsrel=1e-9,
            )
            aircraft__wing__mass_second_part, _ = quadgk(
                lambda x: density
                * 2
                * jnp.atleast_1d(self.airfoil_thickness(x, max_thickness))
                * jnp.sqrt(
                    1 + (camber / (1 - camber_location) ** 2 * (2 * camber_location - 2 * x)) ** 2
                ),
                [camber_location, 1],
                epsabs=1e-9,
                epsrel=1e-9,
            )

            aircraft__wing__mass = (
                aircraft__wing__mass_first_part + aircraft__wing__mass_second_part
            )

        elif airfoil_data_file is not None:
            aircraft__wing__mass, _ = quadgk(
                density * 2 * thickness_dist * jnp.sqrt(1 + jnp.gradient(camber_line) ** 2),
                [0, 1],
                epsabs=1e-9,
                epsrel=1e-9,
            )

        return aircraft__wing__mass

    def precompute_airfoil_geometry(self):
        num_sections = self.options['num_sections']
        n_points = num_sections
        x_points = jnp.linspace(0, 1, n_points)
        dx = 1 / (n_points - 1)
        return x_points, dx

    def airfoil_thickness(self, x, max_thickness):
        return (
            5
            * max_thickness
            * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        )

    def airfoil_camber_line(self, x, camber, camber_location):
        camber_location = omj.smooth_max(camber_location, 1e-9)  # Divide by zero check
        return jnp.where(
            x < camber_location,
            (camber / camber_location**2) * (2 * camber_location * x - x**2),
            (camber / (1 - camber_location) ** 2)
            * ((1 - 2 * camber_location) + 2 * camber_location * x - x**2),
        )

    def extract_airfoil_features(self, x_coords, y_coords):
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

        max_thickness_index = jnp.argmax(thickness)
        max_thickness_value = thickness[max_thickness_index]

        camber_slope = jnp.gradient(camber_line, x_coords)
        camber_location_index = jnp.argmax(omj.smooth_abs(camber_slope))
        camber_location = x_coords[camber_location_index]

        camber = camber_line[camber_location_index]

        return camber, camber_location, max_thickness_value, thickness, camber_line
