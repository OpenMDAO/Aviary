import openmdao.api as om
import numpy as np
from scipy.interpolate import interp1d

import jax.numpy as jnp
import openmdao.jax as omj
import jax.scipy.interpolate as jinterp

try:
    from quadax import quadgk
except ImportError:
    raise ImportError(
        "quadax package not found. You can install it by running 'pip install quadax'."
    )

"""
The little bit of path code below is not important overall. This is for me to test 
within the Docker container and VS Code before I push everything fully to the Github 
repository. These lines can be deleted as things are updated further.

"""

import sys
import os


module_path = os.path.abspath("/home/omdao/Aviary/aviary/subsystems/mass")
if module_path not in sys.path:
    sys.path.append(module_path)

from simple_mass.materials_database import materials

from aviary.utils.named_values import get_keys

Debug = True

class FuselageMassAndCOG(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', 
                             types=int, 
                             default=10)
        
        self.options.declare('material', 
                             default='Aluminum Oxide', 
                             values=list(get_keys(materials)))
        
        self.options.declare('custom_fuselage_data_file', 
                             types=(str, type(None)), 
                             default=None, 
                             allow_none=True)
        
        self.custom_fuselage_function = None

    def setup(self):
        self.options['use_jit'] = not(Debug)

        # Inputs
        self.add_input('length', 
                       val=2.0, 
                       units='m')
        
        self.add_input('diameter', 
                       val=0.4, 
                       units='m')
        
        self.add_input('taper_ratio', 
                       val=1.0, 
                       units=None) # 1.0 means no taper
        
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
        self.add_output('center_of_gravity_x_fuse', 
                        val=0.0, 
                        units='m')
        
        self.add_output('center_of_gravity_y_fuse', 
                        val=0.0, 
                        units='m')
        
        self.add_output('center_of_gravity_z_fuse', 
                        val=0.0, 
                        units='m')
        
        self.add_output('total_weight_fuse', 
                        val=0.0, 
                        units='kg')
    
    def compute_primal(self, length, diameter, taper_ratio, curvature, thickness, y_offset, z_offset, is_hollow):
        # Input validation checks
        if length <= 0:
            raise ValueError("Length must be greater than zero.")
        
        if diameter <= 0:
            raise ValueError("Diameter must be greater than zero.")

        custom_fuselage_function = getattr(self, 'custom_fuselage_function', None) # Custom fuselage model function -- if provided

        custom_fuselage_data_file = self.options['custom_fuselage_data_file']

        material = self.options['material']
        num_sections = self.options['num_sections']
        
        self.validate_inputs(length, diameter, thickness, taper_ratio, is_hollow)

        density, _ = materials.get_item(material)

        section_locations = jnp.linspace(0, length, num_sections)
        
        total_weight_fuse = 0
        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0

        interpolate_diameter = self.load_fuselage_data(custom_fuselage_data_file)

        # Loop through each section
        for location in section_locations:
            section_diameter = self.get_section_diameter(location, length, diameter, taper_ratio, interpolate_diameter)
            outer_radius = section_diameter / 2.0 
            inner_radius = jnp.where(is_hollow, omj.smooth_max(0, outer_radius - thickness), 0)

            section_volume = jnp.pi * (outer_radius**2 - inner_radius**2) * (length / num_sections)
            section_weight = density * section_volume

            centroid_x, centroid_y, centroid_z = self.compute_centroid(location, length, y_offset, z_offset, curvature, taper_ratio)

            total_weight_fuse += section_weight
            total_moment_x += centroid_x * section_weight
            total_moment_y += centroid_y * section_weight
            total_moment_z += centroid_z * section_weight

        center_of_gravity_x_fuse = total_moment_x / total_weight_fuse
        center_of_gravity_y_fuse = total_moment_y / total_weight_fuse
        center_of_gravity_z_fuse = total_moment_z / total_weight_fuse

        return center_of_gravity_x_fuse, center_of_gravity_y_fuse, center_of_gravity_z_fuse, total_weight_fuse

    def validate_inputs(self, length, diameter, thickness, taper_ratio, is_hollow):
        if length <= 0 or diameter <= 0 or thickness <= 0:
            raise ValueError("Length, diameter, and thickness must be positive values.")
        if taper_ratio < 0 or taper_ratio > 1:
            raise ValueError("Taper ratio must be between 0 and 1.")
        if is_hollow and thickness >= diameter / 2:
            raise ValueError("Wall thickness is too large for a hollow fuselage.")
    
    def load_fuselage_data(self, custom_fuselage_data_file):
        if custom_fuselage_data_file:
            try:
                # Load the file
                custom_data = np.loadtxt(custom_fuselage_data_file)
                fuselage_locations = custom_data[:, 0]
                fuselage_diameters = custom_data[:, 1]
                return jinterp.RegularGridInterpolator(fuselage_locations, fuselage_diameters, kind='linear', fill_value='extrapolate')
            except Exception as e:
                raise ValueError(f"Error loading fuselage data file: {e}")
        else:
            return None
    
    def get_section_diameter(self, location, length, diameter, taper_ratio, interpolate_diameter):
        if self.custom_fuselage_function:
            return self.custom_fuselage_function(location)
        elif self.load_fuselage_data:
            return interpolate_diameter(location) if interpolate_diameter is not None else omj.smooth_max(0.01, diameter * (1 - taper_ratio * (location / length)))
        else:
            return omj.smooth_max(0.01, diameter * (1 - taper_ratio * (location / length)))
    
    def compute_centroid(self, location, length, y_offset, z_offset, curvature, taper_ratio):
        centroid_x = jnp.where(taper_ratio > 0, (3/4) * location, location)
        centroid_y = y_offset * (1 - location / length)
        centroid_z = z_offset * (1 - location / length) + curvature * location**2 / length
        return centroid_x, centroid_y, centroid_z


if __name__ == "__main__":
    prob = om.Problem()

    prob.model.add_subsystem('fuselage_cg', FuselageMassAndCOG(), promotes_inputs=['length', 'diameter', 'taper_ratio', 'curvature', 'thickness', 'y_offset', 'z_offset', 'is_hollow'])

    prob.setup()

    prob.set_val('length', 2.5)
    prob.set_val('diameter', 0.5)
    prob.set_val('taper_ratio', 0.5)
    prob.set_val('curvature', 0.0)
    prob.set_val('thickness', 0.05) # Wall thickness of 5 cm
    #prob.set_val('is_hollow', False) # Default is True, uncomment to use False -- for testing purposes

    # Example using custom function -- uncomment to run 
    #def custom_fuselage_model(location):
    #    return 0.5 * jnp.exp(-0.1 * location)

    #prob.model.fuselage_cg.custom_fuselage_function = custom_fuselage_model

    # Example for custom .dat file -- uncomment to run
    #prob.model.fuselage_cg.options['custom_fuselage_data_file'] = 'Custom_Fuselage.dat'

    prob.run_model()

    center_of_gravity_x = prob.get_val('fuselage_cg.center_of_gravity_x_fuse')
    center_of_gravity_y = prob.get_val('fuselage_cg.center_of_gravity_y_fuse')
    center_of_gravity_z = prob.get_val('fuselage_cg.center_of_gravity_z_fuse')
    total_weight = prob.get_val('fuselage_cg.total_weight_fuse')

    #data = prob.check_partials(compact_print=True, abs_err_tol=1e-04, rel_err_tol=1e-04, step=1e-8, step_calc='rel')

    print(f"Center of gravity of the fuselage: X = {center_of_gravity_x} m, Y = {center_of_gravity_y} m, Z = {center_of_gravity_z} m")
    print(f"Total mass of the fuselage: {total_weight} kg")

