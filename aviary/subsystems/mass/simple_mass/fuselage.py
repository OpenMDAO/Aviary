import openmdao.api as om
import numpy as np
from scipy.interpolate import interp1d

import jax.numpy as jnp
import openmdao.jax as omj
import jax.scipy.interpolate as jinterp

from aviary.variable_info.variables import Aircraft
from aviary.variable_info.functions import add_aviary_output, add_aviary_input

try:
    from quadax import quadgk
except ImportError:
    raise ImportError(
        "quadax package not found. You can install it by running 'pip install quadax'."
    )

from aviary.subsystems.mass.simple_mass.materials_database import materials

from aviary.utils.named_values import get_keys

Debug = True

class FuselageMass(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', 
                             types=int, 
                             default=10)
        
        self.options.declare('material', 
                             default='Aluminum Oxide', 
                             values=list(get_keys(materials)))
        
        self.options.declare('fuselage_data_file', 
                             types=(Path, str), 
                             default=None, 
                             allow_none=True,
                             desc='optional data file of fuselage geometry')
        
        self.custom_fuselage_function = None

    def setup(self):
        self.options['use_jit'] = not(Debug)

        # Inputs
        add_aviary_input(self,
                         Aircraft.Fuselage.LENGTH, 
                         units='m')
        
        self.add_input('base_diameter', 
                       val=0.4, 
                       units='m') # no aviary input

        self.add_input('tip_diameter',
                       val=0.2, 
                       units='m') # no aviary input
        
        self.add_input('curvature', 
                       val=0.0, 
                       units='m') # 0 for straight, positive for upward curve
        
        self.add_input('thickness', 
                       val=0.05, 
                       units='m') # Wall thickness of the fuselage
        
        # Allow for asymmetry in the y and z axes -- this value acts as a slope for linear variation along these axes
        self.add_input('y_offset', 
                       val=0.0, 
                       units='m')
        
        self.add_input('z_offset', 
                       val=0.0, 
                       units='m')
        
        self.add_input('is_hollow', 
                       val=True, 
                       units=None) # Whether the fuselage is hollow or not (default is hollow)
        

        # Outputs
        add_aviary_output(self,
                          Aircraft.Fuselage.MASS, 
                          units='kg')
    
    def compute_primal(self, aircraft__fuselage__length, base_diameter, tip_diameter, curvature, thickness, y_offset, z_offset, is_hollow):
        # Input validation checks
        if aircraft__fuselage__length <= 0:
            raise AnalysisError("Length must be greater than zero.")
        
        if base_diameter <= 0 or tip_diameter <= 0:
            raise AnalysisError("Diameter must be greater than zero.")

        custom_fuselage_function = getattr(self, 'custom_fuselage_function', None) # Custom fuselage model function -- if provided

        custom_fuselage_data_file = self.options['custom_fuselage_data_file']

        material = self.options['material']
        num_sections = self.options['num_sections']
        
        self.validate_inputs(aircraft__fuselage__length, base_diameter, thickness, tip_diameter, is_hollow)

        density = materials.get_val(material, 'kg/m**3')

        section_locations = jnp.linspace(0, aircraft__fuselage__length, num_sections)
        
        aircraft__fuselage__mass = 0
        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0

        interpolate_diameter = self.load_fuselage_data(custom_fuselage_data_file)

        # Loop through each section
        for location in section_locations:
            section_diameter = self.get_section_diameter(location, aircraft__fuselage__length, base_diameter, tip_diameter, interpolate_diameter)
            outer_radius = section_diameter / 2.0 
            inner_radius = jnp.where(is_hollow, omj.smooth_max(0, outer_radius - thickness), 0)

            section_volume = jnp.pi * (outer_radius**2 - inner_radius**2) * (aircraft__fuselage__length / num_sections)
            section_weight = density * section_volume

            centroid_x, centroid_y, centroid_z = self.compute_centroid(location, aircraft__fuselage__length, y_offset, z_offset, curvature, base_diameter, tip_diameter)

            aircraft__fuselage__mass += section_weight
            total_moment_x += centroid_x * section_weight
            total_moment_y += centroid_y * section_weight
            total_moment_z += centroid_z * section_weight

        return aircraft__fuselage__mass

    def validate_inputs(self, length, base_diameter, thickness, tip_diameter, is_hollow):
        if length <= 0 or base_diameter <= 0 or tip_diameter <= 0 or thickness <= 0:
            raise AnalysisError("Length, diameter, and thickness must be positive values.")
        if is_hollow and thickness >= base_diameter / 2:
            raise AnalysisError("Wall thickness is too large for a hollow fuselage.")
    
    def load_fuselage_data(self, custom_fuselage_data_file):
        if custom_fuselage_data_file:
            try:
                # Load the file
                custom_data = np.loadtxt(custom_fuselage_data_file)
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Fuselage data file {e}')
            else:
                fuselage_locations = custom_data[:, 0]
                fuselage_diameters = custom_data[:, 1]
                return jinterp.RegularGridInterpolator(fuselage_locations, fuselage_diameters, kind='linear', fill_value='extrapolate')
        else:
            return None
    
    def get_section_diameter(self, location, length, base_diameter, tip_diameter, interpolate_diameter):
        if self.custom_fuselage_function:
            return self.custom_fuselage_function(location)
        elif self.load_fuselage_data and interpolate_diameter is not None:
            return interpolate_diameter(location)
        else:
            return base_diameter + ((tip_diameter - base_diameter) / length) * location
    
    def compute_centroid(self, location, length, y_offset, z_offset, curvature, base_diameter, tip_diameter):
        centroid_x = jnp.where(tip_diameter / base_diameter != 1, (3/4) * location, location)
        centroid_y = y_offset * (1 - location / length)
        centroid_z = z_offset * (1 - location / length) + curvature * location**2 / length
        return centroid_x, centroid_y, centroid_z




