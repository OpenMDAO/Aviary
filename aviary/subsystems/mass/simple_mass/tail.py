from pathlib import Path

import jax.numpy as jnp
import numpy as np
import openmdao.api as om

from aviary.subsystems.mass.simple_mass.materials_database import materials
from aviary.subsystems.mass.simple_mass.utils import (
    airfoil_camber_line,
    airfoil_thickness,
    extract_airfoil_features,
    precompute_airfoil_geometry,
)
from aviary.utils.functions import get_path
from aviary.utils.named_values import get_keys
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
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
            'airfoil_data_file',
            default=None,
            types=(str, Path),
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
        # TODO most of these are unused?!?
        add_aviary_input(self, Aircraft.HorizontalTail.SPAN, units='m', desc='Tail span')
        add_aviary_input(
            self, Aircraft.HorizontalTail.ROOT_CHORD, units='m', desc='Root chord length'
        )

        self.add_input('tip_chord_tail', val=0.8, units='m', desc='Tip chord length')
        self.add_input(
            'thickness_ratio', val=0.12, desc='Max thickness to chord ratio for NACA airfoil'
        )
        self.add_input('skin_thickness', val=0.002, units='m', desc='Skin panel thickness')
        self.add_input(
            'twist_tail',
            val=jnp.zeros(self.options['num_sections']),
            units='deg',
            desc='Twist distribution',
        )

        # Outputs
        add_aviary_output(
            self, Aircraft.HorizontalTail.MASS, units='kg', desc='Total mass of the tail'
        )

        # File check
        airfoil_file = self.options['airfoil_data_file']
        if airfoil_file is not None:
            airfoil_file = get_path(airfoil_file)
            airfoil_data = np.loadtxt(airfoil_file, skiprows=1)  # Assume a header

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
        aircraft__horizontal_tail__span,
        aircraft__horizontal_tail__root_chord,
        tip_chord_tail,
        thickness_ratio,
        skin_thickness,
        twist_tail,
    ):
        material = self.options['material']
        density = materials.get_val(material, 'kg/m**3')
        airfoil_file = self.options['airfoil_data_file']
        num_sections = self.options['num_sections']
        camber = self.camber
        camber_location = self.camber_location
        max_thickness = self.max_thickness
        camber_line = self.camber_line
        thickness = self.thickness

        # This is just so that the differentiation and unittest do not break.
        aircraft__horizontal_tail__mass = 0.0 * thickness_ratio

        # TODO unused??
        span_locations = jnp.linspace(0, aircraft__horizontal_tail__span, num_sections)

        # Get x_points and dx for later
        x_points, dx = precompute_airfoil_geometry(num_sections)

        # Thickness distribution
        thickness_dist = airfoil_thickness(x_points, max_thickness)

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
            'airfoil_data_file',
            default=None,
            types=(str, Path),
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
        # TODO most of these are unused?!?
        add_aviary_input(self, Aircraft.VerticalTail.SPAN, units='m', desc='Tail span')
        add_aviary_input(
            self, Aircraft.VerticalTail.ROOT_CHORD, units='m', desc='Root chord length'
        )

        self.add_input('tip_chord_tail', val=0.8, units='m', desc='Tip chord length')
        self.add_input(
            'thickness_ratio', val=0.12, desc='Max thickness to chord ratio for NACA airfoil'
        )
        self.add_input('skin_thickness', val=0.002, units='m', desc='Skin panel thickness')
        self.add_input(
            'twist_tail',
            val=jnp.zeros(self.options['num_sections']),
            units='deg',
            desc='Twist distribution',
        )

        # Outputs
        add_aviary_output(
            self, Aircraft.VerticalTail.MASS, units='kg', desc='Total mass of the tail'
        )

        # File check
        airfoil_file = self.options['airfoil_data_file']
        if airfoil_file is not None:
            airfoil_file = get_path(airfoil_file)
            airfoil_data = np.loadtxt(airfoil_file, skiprows=1)  # Assume a header

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
        aircraft__vertical_tail__span,
        aircraft__vertical_tail__root_chord,
        tip_chord_tail,
        thickness_ratio,
        skin_thickness,
        twist_tail,
    ):
        material = self.options['material']
        density = materials.get_val(material, 'kg/m**3')
        airfoil_file = self.options['airfoil_data_file']
        num_sections = self.options['num_sections']
        camber = self.camber
        camber_location = self.camber_location
        max_thickness = self.max_thickness
        camber_line = self.camber_line
        thickness = self.thickness

        # This is just so that the differentiation and unittest do not break.
        aircraft__vertical_tail__mass = 0.0 * thickness_ratio

        # TODO unused??
        span_locations = jnp.linspace(0, aircraft__vertical_tail__span, num_sections)

        # Get x_points and dx for later
        x_points, dx = precompute_airfoil_geometry(num_sections)

        # Thickness distribution
        thickness_dist = airfoil_thickness(x_points, max_thickness)

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
