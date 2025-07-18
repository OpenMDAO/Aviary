import jax.numpy as jnp
import numpy as np
import openmdao.api as om
import openmdao.jax as omj
from scipy.interpolate import CubicSpline

from aviary.subsystems.mass.simple_mass.materials_database import materials
from aviary.utils.functions import get_path
from aviary.utils.named_values import get_keys
from aviary.variable_info.functions import add_aviary_output
from aviary.variable_info.variables import Aircraft

try:
    from quadax import quadgk
except ImportError:
    raise ImportError(
        "quadax package not found. You can install it by running 'pip install quadax'."
    )


# BUG this component is not currently working - mass is not being properly computed
class HorizontalTailMass(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare(
            'NACA_digits',
            default='2412',
            types=(str, int),
            desc='4 digit code for NACA airfoil of tail',
        )

        self.options.declare(
            'airfoil_file',
            default=None,
            desc='File path for airfoil coordinates (overrides NACA digits)',
        )

        self.options.declare(
            'material', default='Balsa', values=list(get_keys(materials)), desc='Material type'
        )

        self.options.declare('num_sections', default=10, desc='Number of sections for enumeration')

        self.camber = 0
        self.camber_location = 0
        self.max_thickness = 0
        self.camber_line = 0
        self.thickness = 0

    def setup(self):
        # Inputs
        # TODO unused?!?
        # add_aviary_input(self, Aircraft.HorizontalTail.SPAN, units='m', desc='Tail span')
        # add_aviary_input(
        #     self, Aircraft.HorizontalTail.ROOT_CHORD, units='m', desc='Root chord length'
        # )

        # self.add_input('tip_chord_tail', val=0.8, units='m', desc='Tip chord length')
        self.add_input(
            'thickness_ratio', val=0.12, desc='Max thickness to chord ratio for NACA airfoil'
        )
        # self.add_input('skin_thickness', val=0.002, units='m', desc='Skin panel thickness')
        # self.add_input(
        #     'twist_tail',
        #     val=jnp.zeros(self.options['num_sections']),
        #     units='deg',
        #     desc='Twist distribution',
        # )

        # Outputs
        add_aviary_output(
            self, Aircraft.HorizontalTail.MASS, units='kg', desc='Total mass of the tail'
        )

        # File check
        airfoil_file = self.options['airfoil_file']
        if airfoil_file is not None:
            airfoil_file = get_path(airfoil_file)
            # try:
            airfoil_data = np.loadtxt(airfoil_file, skiprows=1)  # Assume a header
            # except Exception as e:
            #     raise ValueError(f'Error reading airfoil file: {e}')
            x_coords, y_coords = airfoil_data[:, 0], airfoil_data[:, 1]

            (
                self.camber,
                self.camber_location,
                self.max_thickness,
                self.camber_line,
                self.thickness,
            ) = extract_airfoil_features(x_coords, y_coords)

        else:
            NACA_digits = str(self.options['NACA_digits'])
            # Parse the NACA airfoil type (4-digit)
            self.camber = int(NACA_digits[0]) / 100.0  # Maximum camber
            self.camber_location = int(NACA_digits[1]) / 10.0  # Location of max camber
            self.max_thickness = int(NACA_digits[2:4]) / 100.0  # Max thickness

    def get_self_statics(self):
        return (
            self.camber,
            self.camber_location,
            self.max_thickness,
            self.camber_line,
            self.thickness,
            self.options['material'],
            self.options['num_sections'],
            self.options['NACA_digits'],
        )

    def compute_primal(
        self,
        # aircraft__horizontal_tail__span,
        # aircraft__horizontal_tail__root_chord,
        # aircraft__vertical_tail__span,
        # aircraft__vertical_tail__root_chord,
        # tip_chord_tail,
        thickness_ratio,
        # skin_thickness,
        # twist_tail,
    ):
        material = self.options['material']
        density = materials.get_val(material, 'kg/m**3')
        airfoil_file = self.options['airfoil_file']
        # num_sections = self.options['num_sections']
        camber = self.camber
        camber_location = self.camber_location
        max_thickness = self.max_thickness
        camber_line = self.camber_line
        thickness = self.thickness

        # This is just so that the differentiation and unittest do not break.
        aircraft__horizontal_tail__mass = 0.0 * thickness_ratio

        # TODO unused??
        # span_locations = jnp.linspace(0, aircraft__horizontal_tail__span, num_sections)

        # Get x_points and dx for later
        # x_points, dx = precompute_airfoil_geometry()

        # Thickness distribution
        # thickness_dist = airfoil_thickness(x_points, max_thickness)

        if airfoil_file is not None:
            aircraft__horizontal_tail__mass, _ = quadgk(
                density * 2 * thickness * jnp.sqrt(1 + jnp.gradient(camber_line) ** 2),
                [0, 1],
                epsabs=1e-9,
                epsrel=1e-9,
            )
        else:
            total_mass_first_part, _ = quadgk(
                lambda x: density
                * 2
                * jnp.atleast_1d(airfoil_thickness(x, max_thickness))
                * jnp.sqrt(
                    1 + ((camber / camber_location**2) * (2 * camber_location - 2 * x)) ** 2
                ),
                [0, camber_location],
                epsabs=1e-9,
                epsrel=1e-9,
            )
            total_mass_second_part, _ = quadgk(
                lambda x: density
                * 2
                * jnp.atleast_1d(airfoil_thickness(x, max_thickness))
                * jnp.sqrt(
                    1 + (camber / (1 - camber_location) ** 2 * (2 * camber_location - 2 * x)) ** 2
                ),
                [camber_location, 1],
                epsabs=1e-9,
                epsrel=1e-9,
            )

            aircraft__horizontal_tail__mass = total_mass_first_part + total_mass_second_part

        return aircraft__horizontal_tail__mass


class VerticalTailMass(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare(
            'NACA_digits',
            default='2412',
            types=(str, int),
            desc='4 digit code for NACA airfoil of tail',
        )

        self.options.declare(
            'airfoil_file',
            default=None,
            desc='File path for airfoil coordinates (overrides NACA digits)',
        )

        self.options.declare(
            'material', default='Balsa', values=list(get_keys(materials)), desc='Material type'
        )

        self.options.declare('num_sections', default=10, desc='Number of sections for enumeration')

        self.camber = 0
        self.camber_location = 0
        self.max_thickness = 0
        self.camber_line = 0
        self.thickness = 0

    def setup(self):
        # Inputs
        # TODO unused?!?
        # add_aviary_input(self, Aircraft.VerticalTail.SPAN, units='m', desc='Tail span')
        # add_aviary_input(
        #     self, Aircraft.VerticalTail.ROOT_CHORD, units='m', desc='Root chord length'
        # )

        # self.add_input('tip_chord_tail', val=0.8, units='m', desc='Tip chord length')
        self.add_input(
            'thickness_ratio', val=0.12, desc='Max thickness to chord ratio for NACA airfoil'
        )
        # self.add_input('skin_thickness', val=0.002, units='m', desc='Skin panel thickness')
        # self.add_input(
        #     'twist_tail',
        #     val=jnp.zeros(self.options['num_sections']),
        #     units='deg',
        #     desc='Twist distribution',
        # )

        # Outputs
        add_aviary_output(
            self, Aircraft.VerticalTail.MASS, units='kg', desc='Total mass of the tail'
        )

        # File check
        airfoil_file = self.options['airfoil_file']
        if airfoil_file is not None:
            airfoil_file = get_path(airfoil_file)
            # try:
            airfoil_data = np.loadtxt(airfoil_file, skiprows=1)  # Assume a header
            # except Exception as e:
            #     raise ValueError(f'Error reading airfoil file: {e}')
            x_coords, y_coords = airfoil_data[:, 0], airfoil_data[:, 1]

            (
                self.camber,
                self.camber_location,
                self.max_thickness,
                self.camber_line,
                self.thickness,
            ) = extract_airfoil_features(x_coords, y_coords)

        else:
            NACA_digits = str(self.options['NACA_digits'])
            # Parse the NACA airfoil type (4-digit)
            self.camber = int(NACA_digits[0]) / 100.0  # Maximum camber
            self.camber_location = int(NACA_digits[1]) / 10.0  # Location of max camber
            self.max_thickness = int(NACA_digits[2:4]) / 100.0  # Max thickness

    def get_self_statics(self):
        return (
            self.camber,
            self.camber_location,
            self.max_thickness,
            self.camber_line,
            self.thickness,
            self.options['material'],
            self.options['num_sections'],
            self.options['NACA_digits'],
        )

    def compute_primal(
        self,
        # aircraft__vertical_tail__span,
        # aircraft__vertical_tail__root_chord,
        # aircraft__vertical_tail__span,
        # aircraft__vertical_tail__root_chord,
        # tip_chord_tail,
        thickness_ratio,
        # skin_thickness,
        # twist_tail,
    ):
        material = self.options['material']
        density = materials.get_val(material, 'kg/m**3')
        airfoil_file = self.options['airfoil_file']
        # num_sections = self.options['num_sections']
        camber = self.camber
        camber_location = self.camber_location
        max_thickness = self.max_thickness
        camber_line = self.camber_line
        thickness = self.thickness

        # This is just so that the differentiation and unittest do not break.
        aircraft__vertical_tail__mass = 0.0 * thickness_ratio

        # TODO unused??
        # span_locations = jnp.linspace(0, aircraft__vertical_tail__span, num_sections)

        # Get x_points and dx for later
        # x_points, dx = precompute_airfoil_geometry()

        # Thickness distribution
        # thickness_dist = airfoil_thickness(x_points, max_thickness)

        if airfoil_file is not None:
            aircraft__vertical_tail__mass, _ = quadgk(
                density * 2 * thickness * jnp.sqrt(1 + jnp.gradient(camber_line) ** 2),
                [0, 1],
                epsabs=1e-9,
                epsrel=1e-9,
            )
        else:
            total_mass_first_part, _ = quadgk(
                lambda x: density
                * 2
                * jnp.atleast_1d(airfoil_thickness(x, max_thickness))
                * jnp.sqrt(
                    1 + ((camber / camber_location**2) * (2 * camber_location - 2 * x)) ** 2
                ),
                [0, camber_location],
                epsabs=1e-9,
                epsrel=1e-9,
            )
            total_mass_second_part, _ = quadgk(
                lambda x: density
                * 2
                * jnp.atleast_1d(airfoil_thickness(x, max_thickness))
                * jnp.sqrt(
                    1 + (camber / (1 - camber_location) ** 2 * (2 * camber_location - 2 * x)) ** 2
                ),
                [camber_location, 1],
                epsabs=1e-9,
                epsrel=1e-9,
            )

            aircraft__vertical_tail__mass = total_mass_first_part + total_mass_second_part

        return aircraft__vertical_tail__mass


# def precompute_airfoil_geometry(self):
#     n_points = self.options['num_sections']
#     x_points = jnp.linspace(0, 1, n_points)
#     dx = 1 / (n_points - 1)
#     return x_points, dx


def airfoil_thickness(x, max_thickness):
    return (
        5
        * max_thickness
        * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    )


# def airfoil_camber_line(self, x, camber, camber_location):
#     camber_location = omj.ks_max(camber_location, 1e-9)  # Divide by zero check
#     return jnp.where(
#         x < camber_location,
#         (camber / camber_location**2) * (2 * camber_location * x - x**2),
#         (camber / (1 - camber_location) ** 2)
#         * ((1 - 2 * camber_location) + 2 * camber_location * x - x**2),
#     )


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
