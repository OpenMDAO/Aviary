import openmdao.api as om
import numpy as np
import scipy.integrate as spi
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import os

# Material densities, all in kg/m^3
MATERIALS = {
    'Aluminum': 2700,
    'Steel': 7850,
    'Titanium': 4500,
    'Carbon Fiber': 1600,
    'Wood': 600
}

class TailMassAndCOG(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('tail_type',
                            default='horizontal',
                            values=['horizontal', 'vertical'],
                            desc="Type of tail: 'horizontal' or 'vertical'")
        
        self.options.declare('airfoil_type', 
                             default='NACA', 
                             values=['NACA', 'file'],
                             desc="Airfoil type: 'NACA' for 4-digit or 'file' for user-provided coordinates")
        
        if self.options['airfoil_type'] == 'NACA':
            self.options.declare('NACA_digits',
                                 default='2412',
                                 desc="4 digit code for NACA airfoil, if that is given.")
        
        self.options.declare('material', 
                             default='Aluminum', 
                             values=list(MATERIALS.keys()),
                             desc="Material type")
        
        self.options.declare('airfoil_file', 
                             default=None, 
                             desc="File path for airfoil coordinates (if applicable)")

        self.options.declare('num_sections', 
                             default=1000, 
                             desc="Number of sections for enumeration")
    
    def setup(self):
        # Inputs
        self.add_input('span', 
                       val=5.0, 
                       units='m', 
                       desc="Tail span")
        
        self.add_input('root_chord', 
                       val=1.2, 
                       units='m', 
                       desc="Root chord length")
        
        self.add_input('tip_chord', 
                       val=0.8, 
                       units='m', 
                       desc="Tip chord length")
        
        self.add_input('thickness_ratio', 
                       val=0.12, 
                       desc="Max thickness to chord ratio for NACA airfoil")
        
        self.add_input('skin_thickness', 
                       val=0.002, 
                       units='m', 
                       desc="Skin panel thickness")
        
        self.add_input('twist', 
                       val=np.zeros(self.options['num_sections']), 
                       units='deg', 
                       desc="Twist distribution")
        
        # Outputs
        self.add_output('mass', 
                        val=0.0, 
                        units='kg', 
                        desc="Total mass of the tail")
        
        self.add_output('cg_x', 
                        val=0.0, 
                        units='m', 
                        desc="X location of the center of gravity")
        
        self.add_output('cg_y', 
                        val=0.0, 
                        units='m', 
                        desc="Y location of the center of gravity")
        
        self.add_output('cg_z', 
                        val=0.0, 
                        units='m', 
                        desc="Z location of the center of gravity")

    def compute(self, inputs, outputs):
        tail_type = self.options["tail_type"]
        airfoil_type = self.options["airfoil_type"]
        material = self.options['material']
        span = inputs['span']
        root_chord = inputs['root_chord']
        tip_chord = inputs['tip_chord']
        thickness_ratio = inputs['thickness_ratio']
        density = MATERIALS[material]
        airfoil_file = self.options['airfoil_file']
        skin_thickness = inputs['skin_thickness']
        num_sections = self.options['num_sections']
        twist = inputs['twist']
        NACA_digits = self.options['NACA_digits']

        # File check
        if airfoil_type == 'file':
            if airfoil_type == 'file' and (airfoil_file is None or not os.path.isfile(airfoil_file)):
                raise FileNotFoundError(f"Airfoil file '{airfoil_file}' not found or not provided.")
            try: 
                airfoil_data = np.loadtxt(airfoil_file, skiprows=1) # Assume a header
                x_coords, y_coords = airfoil_data[:, 0], airfoil_data[:, 1]
            except Exception as e:
                raise ValueError(f"Error reading airfoil file: {e}")
        
        # Compute section airfoil geometry
        if airfoil_file and os.path.exists(airfoil_file):
            airfoil_data = np.loadtxt(airfoil_file)
            x_coords = airfoil_data[:, 0]
            y_coords = airfoil_data[:, 1]

            camber, camber_location, max_thickness = self.extract_airfoil_features(x_coords, y_coords)
        else:
            # Parse the NACA airfoil type (4-digit)
            camber = int(NACA_digits[0]) / 100.0 # Maximum camber
            camber_location = int(NACA_digits[1]) / 10.0 # Location of max camber
            max_thickness = int(NACA_digits[2:4]) / 100.0 # Max thickness
        
        # Tail type check
        if tail_type not in ['horizontal', 'vertical']:
            raise ValueError("Invalid tail_type. Must be 'horizontal' or 'vertical'.")
        
        # Input validation checks
        if span <= 0:
            raise ValueError("Span must be greater than zero.")
        
        if root_chord <= 0 or tip_chord <= 0:
            raise ValueError("Root chord and tip chord must be greater than zero.")
        
        if tip_chord > root_chord:
            raise ValueError("Tip chord cannot be larger than root chord.")
        
        if thickness_ratio <= 0:
            raise ValueError("Thickness ratio must be greater than zero.")
        
        if skin_thickness <= 0:
            raise ValueError("Skin thickness must be greater than zero.")
        
        if any(abs(twist)) > np.pi / 2:
            raise ValueError("Twist angle is too extreme; must be within -90 to 90 degrees.")

        span_locations = np.linspace(0, span, num_sections)

        # Get x_points and dx for later
        x_points, dx = self.precompute_airfoil_geometry()

        # Thickness distribution
        thickness_dist = self.airfoil_thickness(x_points, max_thickness)

        total_mass = 0
        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0

        for i, y in enumerate(span_locations):
            section_chord = root_chord - (root_chord - tip_chord) * (y / span) # Assume linear variation in chord
            section_area, centroid_x, centroid_z = self.compute_airfoil_geometry(section_chord, 
                                                                                 camber, 
                                                                                 camber_location, 
                                                                                 thickness_dist, 
                                                                                 x_points, 
                                                                                 dx)

            
            section_mass = density * section_area * (span / num_sections)
            
            # Twist
            twist_angle = twist[i]
            rotated_x = centroid_x * np.cos(twist_angle) - centroid_z * np.sin(twist_angle)
            rotated_z = centroid_x * np.sin(twist_angle) + centroid_z * np.cos(twist_angle)

            total_mass += section_mass
            total_moment_x += rotated_x * section_mass
            if tail_type == 'horizontal':
                total_moment_y += y * section_mass
                total_moment_z += rotated_z * section_mass
            elif tail_type == 'vertical':
                total_moment_y += rotated_z * section_mass
                total_moment_z += y * section_mass

        # COG
        outputs['mass'] = total_mass
        outputs['cg_x'] = total_moment_x / total_mass
        outputs['cg_y'] = total_moment_y / total_mass
        outputs['cg_z'] = total_moment_z / total_mass
    
    def precompute_airfoil_geometry(self):
        num_sections = self.options['num_sections']
        n_points = num_sections
        x_points = np.linspace(0, 1, n_points)
        dx = 1 / (n_points - 1)
        return x_points, dx
    
    def compute_airfoil_geometry(self, chord, camber, camber_location, thickness_dist, x_points, dx):

        #section_area, _ = quad(lambda x: np.interp(x, x_points, thickness_dist), 0, 1, limit=100)
        section_area = np.trapz(thickness_dist, x_points, dx=dx) 
        section_area *= chord

        #centroid_x, _ = quad(lambda x: x * np.interp(x, x_points, thickness_dist), 0, 1, limit=100)
        centroid_x = np.trapz(x_points * thickness_dist, x_points, dx=dx)
        centroid_x = (centroid_x * chord) / section_area

        #centroid_z, _ = quad(lambda x: self.airfoil_camber_line(x, camber, camber_location) * np.interp(x, x_points, thickness_dist), 0, 1, limit=100)
        centroid_z = np.trapz(self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist, x_points, dx=dx)
        centroid_z = (centroid_z * chord) / section_area
        return section_area, centroid_x, centroid_z
    
    def airfoil_thickness(self, x, max_thickness):
        return 5 * max_thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    def airfoil_camber_line(self, x, camber, camber_location):
        camber_location = max(camber_location, 1e-9) # Divide by zero check
        return np.where(
            x < camber_location, 
            (camber / camber_location**2) * (2 * camber_location * x - x**2), 
        (camber / (1 - camber_location)**2) * ((1 - 2 * camber_location) + 2 * camber_location * x - x**2)
        )

    def extract_airfoil_features(self, x_coords, y_coords):
        """
        Extract camber, camber location, and max thickness from the given airfoil data.
        This method assumes x_coords are normalized (ranging from 0 to 1).
        """
        # Approximate the camber line and max thickness from the data
        # Assume the camber line is the line of symmetry between the upper and lower surfaces
        upper_surface = y_coords[:int(len(x_coords) // 2)]
        lower_surface = y_coords[int(len(x_coords) // 2):]
        x_upper = x_coords[:int(len(x_coords) // 2)]
        x_lower = x_coords[int(len(x_coords) // 2):]

        upper_spline = CubicSpline(x_upper, upper_surface, bc_type='natural')
        lower_spline = CubicSpline(x_lower, lower_surface, bc_type='natural')

        camber_line = (upper_spline(x_coords) + lower_spline(x_coords)) / 2.0

        thickness = upper_spline(x_coords) - lower_spline(x_coords)

        max_thickness_index = np.argmax(thickness)
        max_thickness_value = thickness[max_thickness_index]

        camber_slope = np.gradient(camber_line, x_coords)
        camber_location_index = np.argmax(np.abs(camber_slope))
        camber_location = x_coords[camber_location_index]

        camber = camber_line[camber_location_index]

        return camber, camber_location, max_thickness_value


prob = om.Problem()

prob.model.add_subsystem('tail', TailMassAndCOG(), promotes_inputs=['*'], promotes_outputs=['*'])

prob.setup()

# Input values
prob.set_val('span', 0.3912)
prob.set_val('tip_chord', 0.15)
prob.set_val('root_chord', 0.26)
prob.set_val('thickness_ratio', 0.12)
prob.set_val('skin_thickness', 0.002)
prob.model.tail.options['tail_type'] = 'vertical'

prob.model.tail.options['material'] = 'Carbon Fiber'

prob.run_model()

# Print
print(f"Mass: {prob.get_val('mass')} kg")
print(f"Center of gravity (X: {prob.get_val('cg_x')} m, Y: {prob.get_val('cg_y')} m, Z: {prob.get_val('cg_z')} m)")