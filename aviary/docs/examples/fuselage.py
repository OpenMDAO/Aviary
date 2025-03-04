import openmdao.api as om
import numpy as np
from scipy.interpolate import interp1d
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Material densities (kg/m^3)
MATERIAL_DENSITIES = {
    'wood': 600, # Not any real wood density
    'metal': 2700, # Aluminum
    'carbon_fiber': 1600,
    'foam': 300 # Example density for foam (just for something lightweight)
}

class FuselageMassAndCOG(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', 
                             types=int, 
                             default=1000)
        
        self.options.declare('material', 
                             default='foam', 
                             values=list(MATERIAL_DENSITIES.keys()))
        
        self.options.declare('custom_fuselage_data_file', 
                             types=(str, type(None)), 
                             default=None, 
                             allow_none=True)
        
        self.custom_fuselage_function = None

    def setup(self):

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
        self.add_output('center_of_gravity_x', 
                        val=0.0, 
                        units='m')
        
        self.add_output('center_of_gravity_y', 
                        val=0.0, 
                        units='m')
        
        self.add_output('center_of_gravity_z', 
                        val=0.0, 
                        units='m')
        
        self.add_output('total_weight', 
                        val=0.0, 
                        units='kg')
    
    def setup_partials(self):
        """
        Complex step is used for the derivatives for now as they are very complicated to calculate 
        analytically. Compute_partials function has the framework for analytical derivatives, but not 
        all of them match the check_partials. 
        """
        self.declare_partials('center_of_gravity_x', ['length', 'diameter', 'taper_ratio', 'curvature', 'thickness'], method='cs')
        self.declare_partials('center_of_gravity_y', ['length', 'y_offset', 'curvature'], method='cs')
        self.declare_partials('total_weight', ['length', 'diameter', 'thickness'], method='cs')
    
    def compute_partials(self, inputs, partials):
        length = inputs['length']
        diameter = inputs['diameter']
        taper_ratio = inputs['taper_ratio']
        curvature = inputs['curvature']
        thickness = inputs['thickness']
        y_offset = inputs['y_offset']
        z_offset = inputs['z_offset']
        is_hollow = inputs['is_hollow']

        #custom_fuselage_function = getattr(self, 'custom_fuselage_function', None) # Custom fuselage model function -- if provided

        custom_fuselage_data_file = self.options['custom_fuselage_data_file']

        material = self.options['material']
        num_sections = self.options['num_sections']
        
        self.validate_inputs(length, diameter, thickness, taper_ratio, is_hollow)

        density = MATERIAL_DENSITIES[material]

        section_locations = np.linspace(0, length, num_sections).flatten()
        dx = 1 / (num_sections - 1)
        
        total_weight = 0
        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0

        interpolate_diameter = self.load_fuselage_data(custom_fuselage_data_file)

        out_r = np.zeros(num_sections)
        in_r = np.zeros(num_sections)


        # Loop through each section
        for i, location in enumerate(section_locations):
            section_diameter = self.get_section_diameter(location, length, diameter, taper_ratio, interpolate_diameter)
            outer_radius = section_diameter / 2.0 
            inner_radius = max(0, outer_radius - thickness) if is_hollow else 0

            out_r[i] = outer_radius
            in_r[i] = inner_radius

            section_volume = np.pi * (outer_radius**2 - inner_radius**2) * (length / num_sections)
            section_weight = density * section_volume

            centroid_x, centroid_y, centroid_z = self.compute_centroid(location, length, y_offset, z_offset, curvature, taper_ratio)

            total_weight += section_weight
            total_moment_x += centroid_x * section_weight
            total_moment_y += centroid_y * section_weight
            total_moment_z += centroid_z * section_weight

        dzcg_dz_offset = np.sum(
            np.trapz(
                (1 - section_locations / length) * density * np.pi * (out_r**2 - in_r**2), section_locations, dx=dx
            )
        ) / total_weight
        

        partials['center_of_gravity_x', 'length'] = 3 / 4 if taper_ratio > 0 else 1
        partials['center_of_gravity_x', 'taper_ratio'] = - (3/4) * length if taper_ratio > 0 else 0
        partials['center_of_gravity_x', 'curvature'] = 0
        partials['center_of_gravity_x', 'thickness'] = 0

        partials['center_of_gravity_y', 'length'] = -y_offset / length
        partials['center_of_gravity_y', 'y_offset'] = 1

        partials['center_of_gravity_z', 'length'] = -z_offset / length
        partials['center_of_gravity_z', 'z_offset'] = dzcg_dz_offset
        partials['center_of_gravity_z', 'curvature'] = length**2 / num_sections

        partials['total_weight', 'length'] = density * np.pi * (diameter**2 - (diameter - 2 * thickness)**2) / num_sections
        partials['total_weight', 'diameter'] = 2 * density * np.pi * length * diameter / num_sections
        partials['total_weight', 'thickness'] = -2 * density * np.pi * length * (diameter - thickness) / num_sections
        
    
    def compute(self, inputs, outputs):
        length = inputs['length']
        diameter = inputs['diameter']
        taper_ratio = inputs['taper_ratio']
        curvature = inputs['curvature']
        thickness = inputs['thickness']
        y_offset = inputs['y_offset']
        z_offset = inputs['z_offset']
        is_hollow = inputs['is_hollow']

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

        density = MATERIAL_DENSITIES[material]

        section_locations = np.linspace(0, length, num_sections)
        
        total_weight = 0
        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0

        interpolate_diameter = self.load_fuselage_data(custom_fuselage_data_file)

        # Loop through each section
        for location in section_locations:
            section_diameter = self.get_section_diameter(location, length, diameter, taper_ratio, interpolate_diameter)
            outer_radius = section_diameter / 2.0 
            inner_radius = max(0, outer_radius - thickness) if is_hollow else 0

            section_volume = np.pi * (outer_radius**2 - inner_radius**2) * (length / num_sections)
            section_weight = density * section_volume

            centroid_x, centroid_y, centroid_z = self.compute_centroid(location, length, y_offset, z_offset, curvature, taper_ratio)

            total_weight += section_weight
            total_moment_x += centroid_x * section_weight
            total_moment_y += centroid_y * section_weight
            total_moment_z += centroid_z * section_weight
        
        outputs['center_of_gravity_x'] = total_moment_x / total_weight
        outputs['center_of_gravity_y'] = total_moment_y / total_weight
        outputs['center_of_gravity_z'] = total_moment_z / total_weight
        outputs['total_weight'] = total_weight

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
                return interp1d(fuselage_locations, fuselage_diameters, kind='linear', fill_value='extrapolate')
            except Exception as e:
                raise ValueError(f"Error loading fuselage data file: {e}")
        else:
            return None
    
    def get_section_diameter(self, location, length, diameter, taper_ratio, interpolate_diameter):
        if self.custom_fuselage_function:
            return self.custom_fuselage_function(location)
        elif self.load_fuselage_data:
            return interpolate_diameter(location) if interpolate_diameter is not None else max(0.01, diameter * (1 - taper_ratio * (location / length)))
        else:
            return max(0.01, diameter * (1 - taper_ratio * (location / length)))
    
    def compute_centroid(self, location, length, y_offset, z_offset, curvature, taper_ratio):
        centroid_x = (3/4) * location if taper_ratio > 0 else location # This is an approximation that could and should be better modeled in the future
        centroid_y = y_offset * (1 - location / length)
        centroid_z = z_offset * (1 - location / length) + curvature * location**2 / length
        return centroid_x, centroid_y, centroid_z
    
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
#    return 0.5 * np.exp(-0.1 * location)

#prob.model.fuselage_cg.custom_fuselage_function = custom_fuselage_model

# Example for custom .dat file -- uncomment to run
#prob.model.fuselage_cg.options['custom_fuselage_data_file'] = 'Custom_Fuselage.dat'

prob.run_model()

center_of_gravity_x = prob.get_val('fuselage_cg.center_of_gravity_x')
center_of_gravity_y = prob.get_val('fuselage_cg.center_of_gravity_y')
center_of_gravity_z = prob.get_val('fuselage_cg.center_of_gravity_z')
total_weight = prob.get_val('fuselage_cg.total_weight')

#data = prob.check_partials(compact_print=True, abs_err_tol=1e-04, rel_err_tol=1e-04, step=1e-8, step_calc='rel')

logger.info(f"Center of gravity of the fuselage: X = {center_of_gravity_x} m, Y = {center_of_gravity_y} m, Z = {center_of_gravity_z} m")
logger.info(f"Total weight of the fuselage: {total_weight} kg")

