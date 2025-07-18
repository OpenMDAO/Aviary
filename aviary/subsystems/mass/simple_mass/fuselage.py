from pathlib import Path

import jax.numpy as jnp
import jax.scipy.interpolate as jinterp
import numpy as np
import openmdao.api as om
import openmdao.jax as omj

from aviary.subsystems.mass.simple_mass.materials_database import materials
from aviary.utils.named_values import get_keys
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class FuselageMass(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', types=int, default=10)

        self.options.declare('material', default='Aluminum Oxide', values=list(get_keys(materials)))

        self.options.declare(
            'fuselage_data_file',
            types=(Path, str),
            default=None,
            allow_none=True,
            desc='optional data file of fuselage geometry',
        )

        # TODO FunctionType is not defined?
        # self.options.declare(
        #     'custom_fuselage_function',
        #     types=FunctionType,
        #     default=None,
        #     allow_none=True,
        #     desc='optional custom function generation for fuselage geometry',
        # )

    def setup(self):
        # Inputs
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='m')

        self.add_input('base_diameter', val=0.4, units='m')  # no aviary input
        self.add_input('tip_diameter', val=0.2, units='m')  # no aviary input
        self.add_input('curvature', val=0.0, units='m')  # 0 for straight, positive for upward curve
        self.add_input('thickness', val=0.05, units='m')  # Wall thickness of the fuselage
        # Allow for asymmetry in the y and z axes -- this value acts as a slope for linear variation along these axes
        self.add_input('y_offset', val=0.0, units='m')
        self.add_input('z_offset', val=0.0, units='m')
        self.add_input(
            'is_hollow', val=True, units=None
        )  # Whether the fuselage is hollow or not (default is hollow)

        # Outputs
        add_aviary_output(self, Aircraft.Fuselage.MASS, units='kg')

    def compute_primal(
        self,
        aircraft__fuselage__length,
        base_diameter,
        tip_diameter,
        curvature,
        thickness,
        y_offset,
        z_offset,
        is_hollow,
    ):
        # Validate inputs
        if (
            aircraft__fuselage__length[0] <= 0
            or base_diameter[0] <= 0
            or tip_diameter[0] <= 0
            or thickness[0] <= 0
        ):
            raise om.AnalysisError('Length, diameter, and thickness must be positive values.')

        if is_hollow and thickness >= base_diameter / 2:
            raise om.AnalysisError('Wall thickness is too large for a hollow fuselage.')

        custom_fuselage_data_file = self.options['fuselage_data_file']
        material = self.options['material']
        num_sections = self.options['num_sections']

        density = materials.get_val(material, 'kg/m**3')

        section_locations = jnp.linspace(0, aircraft__fuselage__length, num_sections)

        aircraft__fuselage__mass = 0
        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0

        # Load fuselage data file if present
        if custom_fuselage_data_file:
            try:
                # Load the file
                custom_data = np.loadtxt(custom_fuselage_data_file)
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Fuselage data file {e}')
            else:
                fuselage_locations = custom_data[:, 0]
                fuselage_diameters = custom_data[:, 1]
                # TODO: OM interp is much more performant than scipy, use metamodel here
                interpolate_diameter = jinterp.RegularGridInterpolator(
                    fuselage_locations, fuselage_diameters, method='linear'
                )
        else:
            interpolate_diameter = None

        # Loop through each section
        for location in section_locations:
            # FunctionType is not defined, so "custom_fuselage_function" is currently broken
            # if self.options['custom_fuselage_function'] is not None:
            #     section_diameter = self.options['custom_fuselage_function'](location)
            # should be elif below once fixed
            if self.options['fuselage_data_file'] and interpolate_diameter is not None:
                section_diameter = interpolate_diameter(location)
            else:
                section_diameter = (
                    base_diameter
                    + ((tip_diameter - base_diameter) / aircraft__fuselage__length) * location
                )

            outer_radius = section_diameter / 2.0
            inner_radius = jnp.where(is_hollow, omj.smooth_max(0, outer_radius - thickness), 0)

            section_volume = (
                jnp.pi
                * (outer_radius**2 - inner_radius**2)
                * (aircraft__fuselage__length / num_sections)
            )
            section_weight = density * section_volume

            centroid_x = jnp.where(tip_diameter / base_diameter != 1, (3 / 4) * location, location)
            centroid_y = y_offset * (1 - location / aircraft__fuselage__length)
            centroid_z = (
                z_offset * (1 - location / aircraft__fuselage__length)
                + curvature * location**2 / aircraft__fuselage__length
            )

            aircraft__fuselage__mass += section_weight
            total_moment_x += centroid_x * section_weight
            total_moment_y += centroid_y * section_weight
            total_moment_z += centroid_z * section_weight

        return aircraft__fuselage__mass
