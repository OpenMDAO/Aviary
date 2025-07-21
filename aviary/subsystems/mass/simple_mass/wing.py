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
from aviary.utils.functions import add_aviary_input, add_aviary_output, get_path
from aviary.utils.named_values import get_keys
from aviary.variable_info.variables import Aircraft

try:
    from quadax import quadgk
except ImportError:
    raise ImportError(
        "quadax package not found. You can install it by running 'pip install quadax'."
    )


class WingMass(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', types=int, default=10)

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

        self.options.declare('num_sections', default=10, desc='Number of sections for enumeration')

        self.options.declare('material', default='Balsa', values=list(get_keys(materials)))

        self.camber = 0
        self.camber_location = 0
        self.max_thickness = 0
        self.camber_line = 0
        self.thickness = 0

    def setup(self):
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

        airfoil_data_file = self.options['airfoil_data_file']
        if airfoil_data_file is not None:
            airfoil_data_file = get_path(airfoil_data_file)
            # try:
            airfoil_data = np.loadtxt(airfoil_data_file)
            x_coords = airfoil_data[:, 0]
            y_coords = airfoil_data[:, 1]

            (
                self.camber,
                self.camber_location,
                self.max_thickness,
                self.thickness,
                self.camber_line,
            ) = extract_airfoil_features(x_coords, y_coords)
            len(x_coords)
        else:
            NACA_digits = str(self.options['NACA_digits'])
            # Parse the NACA airfoil type (4-digit)
            self.camber = int(NACA_digits[0]) / 100.0  # Maximum camber
            self.camber_location = int(NACA_digits[1]) / 10.0  # Location of max camber
            self.max_thickness = int(NACA_digits[2:4]) / 100.0  # Max thickness
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
        material = self.options['material']
        density = materials.get_val(material, 'kg/m**3')
        airfoil_data_file = self.options['airfoil_data_file']
        num_sections = self.options['num_sections']
        camber = self.camber
        camber_location = self.camber_location
        max_thickness = self.max_thickness
        camber_line = self.camber_line
        thickness = self.thickness

        # Get material density
        density = materials.get_val(material, 'kg/m**3')

        # Wing spanwise distribution
        jnp.linspace(0, aircraft__wing__span, num_sections)

        n_points = num_sections
        jnp.linspace(0, 1, n_points)
        1 / (n_points - 1)

        if airfoil_data_file is not None:
            aircraft__wing__mass, _ = quadgk(
                density * 2 * thickness_dist * jnp.sqrt(1 + jnp.gradient(camber_line) ** 2),
                [0, 1],
                epsabs=1e-9,
                epsrel=1e-9,
            )
        else:
            aircraft__wing__mass_first_part, _ = quadgk(
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
            aircraft__wing__mass_second_part, _ = quadgk(
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

            aircraft__wing__mass = (
                aircraft__wing__mass_first_part + aircraft__wing__mass_second_part
            )

        return aircraft__wing__mass
