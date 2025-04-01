import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline

# Material densities (g/cm³ converted to kg/m³) -- these are just some random examples
MATERIAL_DENSITIES = {
    'wood': 600, # This isn't even a real wood density that I'm aware of, I just made it up for the sake of debugging
    'metal': 2700, # Aluminum
    'carbon_fiber': 1600
}



class CenterOfGravity3D(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', types=int, default=50)
        self.options.declare('airfoil_type', types=str, default='2412') # use 2412 as example for default 
        self.options.declare('material', default='metal', values=['wood', 'metal', 'carbon_fiber'])
        self.options.declare('airfoil_data_file', default=None, types=str) # For user-provided airfoil data file

    def setup(self):
        # Inputs
        self.add_input('span', val=10.0, units='m')  # Full wingspan (adjustable)
        self.add_input('root_chord', val=2.0, units='m')  # Root chord length
        self.add_input('tip_chord', val=1.0, units='m')  # Tip chord length
        self.add_input('twist', val=np.zeros(self.options['num_sections']), units='deg')  # Twist angles
        self.add_input('thickness', val=0.2, units='m')  # Thickness of the wing (height)
        

        # Outputs
        self.add_output('center_of_gravity_x', val=0.0, units='m')
        self.add_output('center_of_gravity_y', val=0.0, units='m')
        self.add_output('center_of_gravity_z', val=0.0, units='m')
        self.add_output('total_weight', val=0.0, units='kg')
        self.add_output('x_coords', val=np.zeros(self.options['num_sections']), units='m')
        self.add_output('y_coords', val=np.zeros(self.options['num_sections']), units='m')
        self.add_output('z_coords', val=np.zeros(self.options['num_sections']), units='m')
        

    def compute(self, inputs, outputs):
        span = inputs['span']
        root_chord = inputs['root_chord']
        tip_chord = inputs['tip_chord']
        twist = np.radians(inputs['twist'])  # Convert twist to radians
        thickness = inputs['thickness']
        material = self.options['material'] # Material is taken from options
        num_sections = self.options['num_sections']
        airfoil_type = self.options['airfoil_type'] # NACA airfoil type 
        airfoil_data_file = self.options['airfoil_data_file']
        
        # Get material density
        density = MATERIAL_DENSITIES[material]  # in kg/m³

        if airfoil_data_file:
            airfoil_data = np.loadtxt(airfoil_data_file)
            x_coords = airfoil_data[:, 0]
            y_coords = airfoil_data[:, 1]

            camber, camber_location, max_thickness = self.extract_airfoil_features(x_coords, y_coords)
        else:
            # Parse the NACA airfoil type (4-digit)
            camber = int(airfoil_type[0]) / 100.0 # Maximum camber
            camber_location = int(airfoil_type[1]) / 10.0 # Location of max camber
            max_thickness = int(airfoil_type[2:4]) / 100.0 # Max thickness 
        
        # Wing spanwise distribution
        span_locations = np.linspace(0, span, num_sections)
        
        # Initialize total weight and moment accumulators
        total_weight = 0
        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0

        # Arrays for storing 3D coordinates of the wing
        x_coords = []
        y_coords = []
        z_coords = []
        top_coords = []
        bottom_coords = []

        # Loop over each section along the span (3D wing approximation)
        for i, location in enumerate(span_locations):
            # Calculate the chord for this section
            chord = root_chord - (root_chord - tip_chord) * (location / span) # Assuming linear variation from root to tip

            # Apply twist 
            twist_angle = twist[i]

            # Calculate the camber and thickness distribution 
            # Airfoil thickness distribution using the NACA Equation
            def airfoil_thickness(x):
                return 5 * max_thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
            
            # Airfoil camber line (NACA 4-digit)
            def airfoil_camber_line(x, camber, camber_location):
                if x < camber_location:
                    return (camber / camber_location**2) * (2 * camber_location * x - x**2)
                else:
                    return (camber / (1 - camber_location)**2) * ((1 - 2 * camber_location) + 2 * camber_location * x - x**2)
            
            # Numerical integration: sum up the areas using small segments along the chord
            n_points = 100
            x_points = np.linspace(0, 1, n_points)
            dx = 1 / (n_points - 1)

            area = 0 # Cross-sectional area of the wing section
            centroid_x = 0 
            centroid_z = 0

            for x in x_points:
                # Thickness at each point along the chord
                thickness_at_x = airfoil_thickness(x) * chord
                # Camber at each point (for z-coordinate)
                camber_at_x = airfoil_camber_line(x, camber, camber_location) * chord
                
                # Area of small rectangle at x
                area += thickness_at_x * dx

                centroid_x += (x * thickness_at_x) * dx 
                centroid_z += (camber_at_x * thickness_at_x) * dx 
            # Weight of this section (density * area * thickness)
            section_weight = density * area * thickness # Volume approximation

            # Normalize
            if area > 0:
                centroid_x /= area
                centroid_z /= area
            else:
                centroid_x = 0
                centroid_z = 0
            # Centroid of this section (assuming centroid is at half chord)
            centroid_y = location

            # Debug print line
            #print("Section " + str(i+1) + ", Location: " + str(location) + " m, Weight: " + str(section_weight) + " kg, Centroid Z: " + str(centroid_z))

            # Apply twist
            rotated_z = centroid_z * np.cos(twist_angle) - centroid_x * np.sin(twist_angle)
            rotated_x = centroid_z * np.sin(twist_angle) + centroid_x * np.cos(twist_angle)

            # Add the section's contributions 
            total_weight += section_weight
            total_moment_x += rotated_x * section_weight
            total_moment_y += centroid_y * section_weight
            total_moment_z += rotated_z * section_weight

            # Store the coordinates for plotting later
            x_coords.append(rotated_x)
            y_coords.append(centroid_y)
            z_coords.append(rotated_z)

        # Calculate the overall center of gravity
        outputs['center_of_gravity_x'] = total_moment_x / total_weight
        outputs['center_of_gravity_y'] = total_moment_y / total_weight
        outputs['center_of_gravity_z'] = total_moment_z / total_weight
        outputs['total_weight'] = total_weight

        # Store the x, y, and z coordinates for the entire wing
        outputs['x_coords'] = np.array(x_coords)
        outputs['y_coords'] = np.array(y_coords)
        outputs['z_coords'] = np.array(z_coords)

    def extract_airfoil_features(self, x_coords, y_coords):
        """
        Extract camber, camber location, and max thickness from the given airfoil data.
        This method assumes x_coords are normalized (ranging from 0 to 1).
        """
        # Approximate the camber line and max thickness from the data
        # Assume the camber line is the line of symmetry between the upper and lower surfaces
        upper_surface = y_coords[:int(len(x_coords) / 2)]
        lower_surface = y_coords[int(len(x_coords) / 2):]
        x_upper = x_coords[:int(len(x_coords) / 2)]
        x_lower = x_coords[int(len(x_coords) / 2):]

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


# Build OpenMDAO problem
prob = om.Problem()

# Add the center of gravity component
prob.model.add_subsystem('cog', CenterOfGravity3D() , promotes_inputs=['span', 'root_chord', 'tip_chord', 'twist', 'thickness'])

# Setup the problem
prob.setup()

# Define some example inputs
prob.set_val('span', 11.0)  
prob.set_val('root_chord', 1.9)  
prob.set_val('tip_chord', 0.5)  
#prob.set_val('twist', np.linspace(0, 0, 50))  
prob.set_val('thickness', 0.2)  

prob.model.cog.options['airfoil_data_file'] = 'airfoil_data_test.dat'

# Run the model
prob.run_model()

# Get the results
center_of_gravity_x = prob.get_val('cog.center_of_gravity_x')
center_of_gravity_y = prob.get_val('cog.center_of_gravity_y')
center_of_gravity_z = prob.get_val('cog.center_of_gravity_z')
total_weight = prob.get_val('cog.total_weight')

# Get the 3D coordinates for the entire wing
x_coords = prob.get_val('cog.x_coords')
y_coords = prob.get_val('cog.y_coords')
z_coords = prob.get_val('cog.z_coords')

print(f"Center of gravity: X = {center_of_gravity_x} m, Y = {center_of_gravity_y} m, Z = {center_of_gravity_z} m")
print(f"Total weight of the wing: {total_weight} kg")

