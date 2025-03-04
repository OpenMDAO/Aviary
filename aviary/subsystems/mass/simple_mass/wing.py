import openmdao.api as om
import numpy as np
import os
from scipy.interpolate import CubicSpline


# Material densities (g/cm³ converted to kg/m³) -- these are just some random examples
MATERIAL_DENSITIES = {
    'wood': 130, # balsa wood
    'metal': 2700, # Aluminum
    'carbon_fiber': 1600
}



class WingMassAndCOG(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', 
                             types=int, 
                             default=1000)
        
        self.options.declare('airfoil_type', 
                             types=str, 
                             default='2412') # use 2412 as example for default 
        
        self.options.declare('material', 
                             default='metal', 
                             values=list(MATERIAL_DENSITIES.keys()))
        
        self.options.declare('airfoil_data_file', 
                             default=None, 
                             types=str) # For user-provided airfoil data file

    def setup(self):

        # Inputs
        self.add_input('span', 
                       val=10.0, 
                       units='m')  # Full wingspan (adjustable)
        
        self.add_input('root_chord', 
                       val=2.0, 
                       units='m')  # Root chord length
        
        self.add_input('tip_chord', 
                       val=1.0, 
                       units='m')  # Tip chord length
        
        self.add_input('twist', 
                       val=np.zeros(self.options['num_sections']), 
                       units='deg')  # Twist angles
        
        self.add_input('thickness_dist', 
                       val=np.ones(self.options['num_sections']) * 0.1, 
                       shape=(self.options['num_sections'],))  # Thickness distribution of the wing (height)
        

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
        num_sections = self.options['num_sections']

        self.declare_partials(['center_of_gravity_x', 'center_of_gravity_y', 'center_of_gravity_z'], 
                              ['span', 'root_chord', 'tip_chord', 'thickness_dist', 'twist'], method='cs') 
        self.declare_partials('total_weight', ['span', 'root_chord', 'tip_chord', 'thickness_dist', 'twist'], method='cs')


    def compute(self, inputs, outputs):
        span = inputs['span']
        root_chord = inputs['root_chord']
        tip_chord = inputs['tip_chord']
        twist = np.radians(inputs['twist'])  # Convert twist to radians
        thickness_dist = inputs['thickness_dist']
        material = self.options['material'] # Material is taken from options
        #num_sections = self.options['num_sections']
        airfoil_type = self.options['airfoil_type'] # NACA airfoil type 
        airfoil_data_file = self.options['airfoil_data_file']

        # Input validation checks
        if span <= 0:
            raise ValueError("Span must be greater than zero.")
        
        if root_chord <= 0 or tip_chord <= 0:
            raise ValueError("Root chord and tip chord must be greater than zero.")
        
        if tip_chord > root_chord:
            raise ValueError("Tip chord cannot be larger than root chord.")
        
        if any(abs(twist)) > np.pi / 2:
            raise ValueError("Twist angle is too extreme; must be within -90 to 90 degrees.")
        
        # Get material density
        density = MATERIAL_DENSITIES[material]  # in kg/m³

        #x_points, dx = self.precompute_airfoil_geometry()

        if airfoil_data_file and os.path.exists(airfoil_data_file):
            airfoil_data = np.loadtxt(airfoil_data_file)
            x_coords = airfoil_data[:, 0]
            y_coords = airfoil_data[:, 1]

            camber, camber_location, max_thickness, thickness, camber_line = self.extract_airfoil_features(x_coords, y_coords)
            thickness_dist = thickness
            num_sections = len(x_coords)
        else:
            # Parse the NACA airfoil type (4-digit)
            camber = int(airfoil_type[0]) / 100.0 # Maximum camber
            camber_location = int(airfoil_type[1]) / 10.0 # Location of max camber
            max_thickness = int(airfoil_type[2:4]) / 100.0 # Max thickness 
            num_sections = self.options['num_sections']
        
        # Wing spanwise distribution
        span_locations = np.linspace(0, span, num_sections)

        #num_sections = self.options['num_sections']
        n_points = num_sections
        x_points = np.linspace(0, 1, n_points)
        dx = 1 / (n_points - 1)
        
        # Initialize total weight and moment accumulators
        total_weight = 0
        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0

        # Arrays for storing 3D coordinates of the wing
        x_coords = []
        y_coords = []
        z_coords = []

        # Loop over each section along the span (3D wing approximation)
        for i, location in enumerate(span_locations):
            # Calculate the chord for this section -- note this is an approximation
            chord = root_chord - (root_chord - tip_chord) * (location / span) # Assuming linear variation from root to tip

            # Apply twist 
            twist_angle = twist[i]

            section_area, centroid_x, centroid_z = self.compute_airfoil_geometry(chord, camber, camber_location, thickness_dist, x_points, dx)
            centroid_y = i * span / num_sections
            section_weight = density * section_area * (span / num_sections)
            
            rotated_x = centroid_x * np.cos(twist_angle) - centroid_z * np.sin(twist_angle)
            rotated_z = centroid_x * np.sin(twist_angle) + centroid_z * np.cos(twist_angle)

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

    
    def compute_partials(self, inputs, J):
        span = inputs['span']
        root_chord = inputs['root_chord']
        tip_chord = inputs['tip_chord']
        thickness_dist = inputs['thickness_dist']
        twist=np.radians(inputs['twist'])
        num_sections = self.options['num_sections']
        airfoil_type = self.options['airfoil_type'] # NACA airfoil type 
        airfoil_data_file = self.options['airfoil_data_file']
        material = self.options['material']

        # Compute section locations along the span
        span_locations = np.linspace(0, span, num_sections)
        span_locations = span_locations / span
        chord_lengths = tip_chord + (root_chord - tip_chord) * (1 - span_locations / span)

        # Compute section airfoil geometry
        if airfoil_data_file and os.path.exists(airfoil_data_file):
            airfoil_data = np.loadtxt(airfoil_data_file)
            x_coords = airfoil_data[:, 0]
            y_coords = airfoil_data[:, 1]

            camber, camber_location, max_thickness, thickness, camber_line = self.extract_airfoil_features(x_coords, y_coords)
            thickness_dist = thickness
        else:
            # Parse the NACA airfoil type (4-digit)
            camber = int(airfoil_type[0]) / 100.0 # Maximum camber
            camber_location = int(airfoil_type[1]) / 10.0 # Location of max camber
            max_thickness = int(airfoil_type[2:4]) / 100.0 # Max thickness

        # Compute section airfoil geometry
        n_points = num_sections
        x_points = np.linspace(0, 1, n_points)
        dx = 1 / (num_sections - 1)

        #A_ref, err = quad(lambda x: np.interp(x, x_points, thickness_dist), 0, 1, limit=100) # \int_0^1 t(x) dx
        A_ref = np.trapz(thickness_dist, x_points, dx=dx)

        density = MATERIAL_DENSITIES[material]

        total_moment_x = 0
        total_moment_y = 0
        total_moment_z = 0
        total_weight = 0

        rotated_x_vals = np.zeros(num_sections)
        rotated_z_vals = np.zeros(num_sections)
        #section_weights = np.zeros(num_sections)
        section_areas = np.zeros(num_sections)
        dA_dspan = 0
        dA_droot_chord = 0
        dA_dtip_chord = 0
        dweight_dspan = 0
        dmoment_x_dtwist = np.zeros(num_sections)
        dmoment_z_dtwist = np.zeros(num_sections)
        dweight_dthickness = np.zeros(num_sections)

        for i, location in enumerate(span_locations):
            # Calculate the chord for this section
            chord = root_chord - (root_chord - tip_chord) * (location / span) # Assuming linear variation from root to tip

            # Apply twist 
            twist_angle = twist[i]

            section_area, centroid_x, centroid_z = self.compute_airfoil_geometry(chord, camber, camber_location, thickness_dist, x_points, dx)
            section_weight = density * section_area * (span / num_sections)
            centroid_y = location 
            
            rotated_x_vals[i] = centroid_x * np.cos(twist_angle) - centroid_z * np.sin(twist_angle)
            rotated_z_vals[i] = centroid_x * np.sin(twist_angle) + centroid_z * np.cos(twist_angle)

            total_weight += section_weight
            total_moment_x += rotated_x_vals[i] * section_weight
            total_moment_y += centroid_y * section_weight
            total_moment_z += rotated_z_vals[i] * section_weight

            #section_weights[i] = section_weight
            section_areas[i] = section_area

            # For dweight_dspan
            dci_dspan = -(root_chord - tip_chord) * (location / span**2)
            #dA_dspan, _ = quad(lambda x: np.interp(x, x_points, thickness_dist) * dci_dspan, 0, 1, limit=100)
            dA_ds = np.trapz(thickness_dist * dci_dspan, x_points, dx=dx) 
            dA_dc_root = np.trapz(thickness_dist * (1 - i / num_sections), x_points, dx=dx)
            dA_dc_tip = np.trapz(thickness_dist * i / num_sections, x_points, dx=dx)
            dA_dspan += dA_ds
            dA_droot_chord += dA_dc_root
            dA_dtip_chord += dA_dc_tip
            dweight_dspan += density * (section_area / num_sections - dA_ds * span / num_sections)
            dmoment_x_dtwist[i] = -section_weight * (centroid_x * np.sin(twist_angle) + centroid_z * np.cos(twist_angle))
            dmoment_z_dtwist[i] = section_weight * (centroid_x * np.cos(twist_angle) - centroid_z * np.sin(twist_angle))
            dweight_dthickness[i] = density * span * chord / num_sections # dW_total / dthickness_dist -- vector value
    
        dweight_droot_chord = np.sum(density * A_ref * span / 2) # ((num_sections - 1) / (2 * num_sections)) ~ 1/2 for large N 
        dweight_dtip_chord = np.sum(density * A_ref * span / 2) # ((num_sections + 1) / (2 * num_sections)) ~ 1/2 for large N

        J['total_weight', 'span'] = dweight_dspan
        J['total_weight', 'root_chord'] = dweight_droot_chord
        J['total_weight', 'tip_chord'] = dweight_dtip_chord
        J['total_weight', 'thickness_dist'] = dweight_dthickness
        J['total_weight', 'twist'] = 0

        dxcg_droot_chord = 0
        dzcg_droot_chord = 0
        dxcg_dtip_chord = 0
        dzcg_dtip_chord = 0
        for i, location in enumerate(span_locations):
            dxcg_dcroot = np.sum(
                np.trapz(
                    (x_points * thickness_dist * np.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * np.sin(twist)) * (1 - i / num_sections), x_points, dx=dx
                    ) 
                    ) / np.sum(section_areas) - np.sum(
            (
                np.trapz(
                    x_points * thickness_dist * chord_lengths * np.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * np.sin(twist), x_points, dx=dx
                    ) * np.trapz(
                        thickness_dist * (1 - i / num_sections), x_points, dx=dx
                        )
                        ) 
                ) / np.sum(section_areas)**2
            dxcg_droot_chord += dxcg_dcroot

            dzcg_dcroot = np.sum(
                np.trapz(
                    (x_points * thickness_dist * (1 - i / num_sections) * np.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist) * (1 - i / num_sections) * np.cos(twist), x_points, dx=dx
                ) 
            ) / np.sum(section_areas) - np.sum(
                np.trapz(
                    x_points * thickness_dist * chord_lengths * np.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * np.cos(twist), x_points, dx=dx
                ) * dA_droot_chord 
            ) / np.sum(section_areas)**2
            dzcg_droot_chord += dzcg_dcroot

            dxcg_dctip = np.sum(
                np.trapz(
                    (x_points * thickness_dist *np.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * np.sin(twist)) * (i / num_sections), x_points, dx=dx
                ) / section_areas
            ) - np.sum(
                (
                    np.trapz(
                        x_points * thickness_dist * chord_lengths * np.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * np.sin(twist), x_points, dx=dx
                    )
                ) * dA_dtip_chord 
            ) / np.sum(section_areas)**2
            dxcg_dtip_chord += dxcg_dctip

            dzcg_dctip = np.sum(
                np.trapz(
                    x_points * thickness_dist * (i / num_sections) * np.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * (i / num_sections) * np.cos(twist), x_points, dx=dx
                )
            ) / np.sum(section_areas) - np.sum(
                np.trapz(
                    x_points * thickness_dist * chord_lengths * np.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * np.cos(twist), x_points, dx=dx
                ) * dA_dtip_chord
            ) / np.sum(section_areas)**2
            dzcg_dtip_chord += dzcg_dctip

        # partials of cog x
        J['center_of_gravity_x', 'span'] = 0
        J['center_of_gravity_x', 'root_chord'] = dxcg_droot_chord
        J['center_of_gravity_x', 'tip_chord'] = dxcg_dtip_chord
        J['center_of_gravity_x', 'thickness_dist'] = np.sum(
            np.trapz(
                x_points * chord_lengths * np.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * chord_lengths * np.sin(twist), x_points, dx=dx
                ) / section_areas
            ) - (
                np.sum(
            np.trapz(x_points * thickness_dist * chord_lengths * np.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * np.sin(twist), x_points, dx=dx)
        ) * np.sum(chord_lengths)
        ) / np.sum(section_areas)**2
        J['center_of_gravity_x', 'twist'] = dmoment_x_dtwist / total_weight
        
        # For center of gravity in y calculations

        sum_area_times_i = 0
        sum_darea_times_i = 0
        
        for i in range(len(x_points)):
            sum_area_times_i += i * section_areas[i]
            sum_darea_times_i += i * dA_dspan # for cg_Y calculations

        dcg_y_dspan = np.sum(sum_area_times_i / (num_sections * section_areas)) #+ np.sum((span * sum_darea_times_i) / (num_sections * section_areas)) - np.sum((span * sum_area_times_i) / (num_sections * np.sum(section_areas)**2)) * dA_dspan   

        # partials of cog y
        J['center_of_gravity_y', 'span'] = dcg_y_dspan
        J['center_of_gravity_y', 'root_chord'] = -total_moment_y / total_weight * dweight_droot_chord / total_weight
        J['center_of_gravity_y', 'tip_chord'] = -total_moment_y / total_weight * dweight_dtip_chord / total_weight
        J['center_of_gravity_y', 'thickness_dist'] = -total_moment_y / total_weight * dweight_dthickness / total_weight
        J['center_of_gravity_y', 'twist'] = 0

        # partials of cog z
        J['center_of_gravity_z', 'span'] = 0
        J['center_of_gravity_z', 'root_chord'] = dzcg_droot_chord
        J['center_of_gravity_z', 'tip_chord'] = dzcg_dtip_chord
        J['center_of_gravity_z', 'thickness_dist'] = np.sum(
            np.trapz(
                x_points * chord_lengths * np.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * chord_lengths * np.cos(twist), x_points, dx=dx
            )
        ) / np.sum(
            section_areas
        )
        J['center_of_gravity_z', 'twist'] = dmoment_z_dtwist / total_weight
    
    def precompute_airfoil_geometry(self):
        num_sections = self.options['num_sections']
        n_points = num_sections
        x_points = np.linspace(0, 1, n_points)
        dx = 1 / (n_points - 1)
        return x_points, dx
    
    def compute_airfoil_geometry(self, chord, camber, camber_location, thickness_dist, x_points, dx): 

        #section_area, _ = quad(lambda x: np.interp(x, x_points, thickness_dist), 0, 1, limit=100)
        section_area = np.trapz(thickness_dist, x_points, dx=dx) # trying np.trapz rather than quad to get rid of IntegrationWarning
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

        return camber, camber_location, max_thickness_value, thickness, camber_line


# Build OpenMDAO problem
prob = om.Problem()

# Add the center of gravity component
prob.model.add_subsystem('cog', WingMassAndCOG() , promotes_inputs=['span', 'root_chord', 'tip_chord', 'twist', 'thickness_dist'])

n_points = 1000 # = num_sections
x = np.linspace(0, 1, n_points)
max_thickness_chord_ratio = 0.12
thickness_dist = 5 * max_thickness_chord_ratio * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4) 

# Setup the problem
prob.setup()

# Define some example inputs
prob.set_val('span', 2.438)  
prob.set_val('root_chord', 0.3722)  
prob.set_val('tip_chord', 0.2792)  
prob.set_val('twist', np.linspace(0, 0, 1000))  
#prob.set_val('thickness_dist', thickness_dist)  


prob.model.cog.options['airfoil_data_file'] = 'Clark_Y.dat'
prob.model.cog.options['material'] = 'wood'
#prob.model.cog.options['airfoil_type'] = '2412'

# Run the model
prob.run_model()

# Get the results
center_of_gravity_x = prob.get_val('cog.center_of_gravity_x')
center_of_gravity_y = prob.get_val('cog.center_of_gravity_y')
center_of_gravity_z = prob.get_val('cog.center_of_gravity_z')
total_weight = prob.get_val('cog.total_weight')

#data = prob.check_partials(compact_print=True, method='cs')
#om.partial_deriv_plot(data)

print(f"Center of gravity: X = {center_of_gravity_x} m, Y = {center_of_gravity_y} m, Z = {center_of_gravity_z} m")
print(f"Total weight of the wing: {total_weight} kg")

