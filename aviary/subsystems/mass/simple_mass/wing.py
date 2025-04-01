import openmdao.api as om
import openmdao.jax as omj
import numpy as np
import jax.numpy as jnp
import jax
import os
from scipy.interpolate import CubicSpline
import jax.scipy.integrate as jint

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

Debug = False

class WingMassAndCOG(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('num_sections', 
                             types=int, 
                             default=10)
        
        self.options.declare('airfoil_type', 
                             types=str, 
                             default='2412') # use 2412 as example for default 
        
        self.options.declare('material', 
                             default='Balsa', 
                             values=list(get_keys(materials)))
        
        self.options.declare('airfoil_data_file', 
                             default=None, 
                             types=str) # For user-provided airfoil data file

    def setup(self):
        self.options['use_jit'] = not(Debug)

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
                       val=jnp.zeros(self.options['num_sections']), 
                       units='deg')  # Twist angles
        
        self.add_input('thickness_dist', 
                       val=jnp.ones(self.options['num_sections']) * 0.1, 
                       shape=(self.options['num_sections'],),
                       units='m')  # Thickness distribution of the wing (height)
        

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

    def compute_primal(self, span, root_chord, tip_chord, twist, thickness_dist):
        material = self.options['material'] # Material is taken from options
        airfoil_type = self.options['airfoil_type'] # NACA airfoil type 
        airfoil_data_file = self.options['airfoil_data_file']

        # Input validation checks
        if span <= 0:
            raise ValueError("Span must be greater than zero.")
        
        if root_chord <= 0 or tip_chord <= 0:
            raise ValueError("Root chord and tip chord must be greater than zero.")
        
        if tip_chord > root_chord:
            raise ValueError("Tip chord cannot be larger than root chord.")
        
        if any(omj.smooth_abs(twist)) > jnp.pi / 2:
            raise ValueError("Twist angle is too extreme; must be within -90 to 90 degrees.")
        
        # Get material density
        density, _ = materials.get_item(material)  # in kg/m³

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
        span_locations = jnp.linspace(0, span, num_sections)

        #num_sections = self.options['num_sections']
        n_points = num_sections
        x_points = jnp.linspace(0, 1, n_points)
        dx = 1 / (n_points - 1)
        
        weight_function = lambda x: density * self.airfoil_thickness(x, max_thickness) * (root_chord - (root_chord - tip_chord) * (x / span)) * span
        
        #total_weight = jint.trapezoid(density * self.airfoil_thickness(x_points, max_thickness) * (root_chord - (root_chord - tip_chord) * (x_points / span)) * span, x_points)

        #total_weight = self.gauss_quad(weight_function, num_sections, 0, 1)
        total_weight, _ = quadgk(weight_function, [0, 1], epsabs=1e-9, epsrel=1e-9)

        #center_of_gravity_x = jint.trapezoid(x_points * self.airfoil_thickness(x_points, max_thickness) * (root_chord - (root_chord - tip_chord) * (x_points / span) * jnp.cos(twist)) - 
        #                          self.airfoil_camber_line(x_points, camber, camber_location) * self.airfoil_thickness(x_points, max_thickness) * (root_chord - (root_chord - tip_chord) * (x_points / span)) * jnp.sin(twist), 
        #                          x_points) / jint.trapezoid(self.airfoil_thickness(x_points, max_thickness) * (root_chord - (root_chord - tip_chord) * (x_points / span)), x_points)
        
        center_of_gravity_x_num, _ = quadgk(lambda x: x * self.airfoil_thickness(x, max_thickness) * (root_chord - (root_chord - tip_chord) * (x / span) * jnp.cos(twist)) - 
                                     self.airfoil_camber_line(x, camber, camber_location) * self.airfoil_thickness(x, max_thickness) * (root_chord - (root_chord - tip_chord) * (x / span)) * jnp.sin(twist), 
                                     [0, 1], epsabs=1e-9, epsrel=1e-9) 
        center_of_gravity_x_denom, _ = quadgk(lambda x: self.airfoil_thickness(x, max_thickness) * (root_chord - (root_chord - tip_chord) * (x / span)), [0, 1], epsabs=1e-9, epsrel=1e-9)

        center_of_gravity_x = center_of_gravity_x_num / center_of_gravity_x_denom
        center_of_gravity_x = center_of_gravity_x[0]
        
        #center_of_gravity_z = jint.trapezoid(x_points * self.airfoil_thickness(x_points, max_thickness) * (root_chord - (root_chord - tip_chord) * (x_points / span) * jnp.sin(twist)) + 
        #                          self.airfoil_camber_line(x_points, camber, camber_location) * self.airfoil_thickness(x_points, max_thickness) * (root_chord - (root_chord - tip_chord) * (x_points / span)) * jnp.cos(twist), 
        #                          x_points) / jint.trapezoid(self.airfoil_thickness(x_points, max_thickness) * (root_chord - (root_chord - tip_chord) * (x_points / span)), x_points)
        
        center_of_gravity_z_num, _ = quadgk(lambda x: x * self.airfoil_thickness(x, max_thickness) * (root_chord - (root_chord - tip_chord) * (x / span) * jnp.sin(twist)) + 
                                     self.airfoil_camber_line(x, camber, camber_location) * self.airfoil_thickness(x, max_thickness) * (root_chord - (root_chord - tip_chord) * (x / span)) * jnp.cos(twist), 
                                     [0, 1], epsabs=1e-9, epsrel=1e-9)
        
        center_of_gravity_z_denom, _ = quadgk(lambda x: self.airfoil_thickness(x, max_thickness) * (root_chord - (root_chord - tip_chord) * (x / span)), [0, 1], epsabs=1e-9, epsrel=1e-9)

        center_of_gravity_z = center_of_gravity_z_num / center_of_gravity_z_denom
        center_of_gravity_z = center_of_gravity_z[0]
        
        #center_of_gravity_y = jint.trapezoid(x_points * span, x_points) 

        center_of_gravity_y, _ = quadgk(lambda x: x * span, [0, 1], epsabs=1e-9, epsrel=1e-9)

        return center_of_gravity_x, center_of_gravity_y, center_of_gravity_z, total_weight
    
    def precompute_airfoil_geometry(self):
        num_sections = self.options['num_sections']
        n_points = num_sections
        x_points = jnp.linspace(0, 1, n_points)
        dx = 1 / (n_points - 1)
        return x_points, dx
    
    def airfoil_thickness(self, x, max_thickness):
        return 5 * max_thickness * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    def airfoil_camber_line(self, x, camber, camber_location):
        camber_location = omj.smooth_max(camber_location, 1e-9) # Divide by zero check
        return jnp.where(
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

        max_thickness_index = jnp.argmax(thickness)
        max_thickness_value = thickness[max_thickness_index]

        camber_slope = jnp.gradient(camber_line, x_coords)
        camber_location_index = jnp.argmax(omj.smooth_abs(camber_slope))
        camber_location = x_coords[camber_location_index]

        camber = camber_line[camber_location_index]

        return camber, camber_location, max_thickness_value, thickness, camber_line

    # def legendre_roots_and_weights(self, n):
    #     """
    #     Compute nodes and weights for Legendre-Gauss quadrature using JAX.

    #     Parameters: 

    #     beta: 
    #         coefficient coming form Gram-Schmidt process, derived from recursive relation
    #         of Legendre polynomials.

    #     T:
    #         symmetric, tri-diagonal Jacobi matrix derived using Golub-Welsch theorem. Eigenvalues
    #         of T correspond to roots of Legendre polynomials
        
    #     Eigenvalues, Eigenvectors:
    #         The eigenvalues of T and their respective eigenvectors, calculated using 
    #         jax.scipy.special.eigh function.
        
    #     weights:
    #         Weights used for Gaussian-Legendre quadrature, calculated using the first
    #         component of the normalized eigenvector, derived from Golub & Welsch (1969). 
    #         Coefficient 2 comes from an application of Rodrigues' formula for Legendre
    #         polynomials and applying the orthonormality property for Legendre polynomials
    #         over the interval [-1, 1] with weight = 1.

    #     Returns:

    #     float
    #         Roots of the Legendre polynomials (eigenvalues of T)
    #     float
    #         weights used for G-L quadrature (formula from Golub & Welsch 1969)

    #     References: 

    #     [1] Golub, Gene H., and John H. Welsch. "Calculation of Gauss quadrature rules." 
    #         Mathematics of computation 23.106 (1969): 221-230.
        
    #     [2] André Ronveaux, Jean Mawhin, Rediscovering the contributions of Rodrigues on 
    #         the representation of special functions, Expositiones Mathematicae, Volume 23, 
    #         Issue 4, 2005, Pages 361-369, ISSN 0723-0869,
    #         https://doi.org/10.1016/j.exmath.2005.05.001. 

    #     """

    #     i = jnp.arange(1, n)
    #     beta = i / jnp.sqrt(4 * i**2 - 1)
    #     T = jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)  # Tridiagonal matrix
    #     eigenvalues, eigenvectors = jax.jit(jnp.linalg.eigh)(T)  # Compute eigenvalues (roots) and eigenvectors
    #     roots = eigenvalues
    #     weights = 2 * (eigenvectors[0, :] ** 2)  # Compute weights from first row of eigenvectors
        
    #     return roots, weights


    # def gauss_quad(self, f, n, a, b):
    #     """
    #     Computes the integral of f(x) over [a, b] using Gaussian quadrature with n points.

    #     Parameters: 

    #     f : function
    #         The function to integrate.
        
    #     n : int
    #         The number of quadrature points.
        
    #     a, b : float
    #         Integration limits.

    #     Returns:

    #     float
    #         The approximated integral of f(x) over [a, b].

    #     """
    #     x, w = self.legendre_roots_and_weights(n)

    #     x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
    #     w_mapped = 0.5 * (b - a) * w

    #     integral = jnp.sum(w_mapped * f(x_mapped))  # Compute weighted sum
    #     return integral.astype(jnp.float64)

# Build OpenMDAO problem
prob = om.Problem()

# Add the center of gravity component
prob.model.add_subsystem('cog', WingMassAndCOG() , promotes_inputs=['*'], promotes_outputs=['*'])

n_points = 10 # = num_sections
x = jnp.linspace(0, 1, n_points)
max_thickness_chord_ratio = 0.12
thickness_dist = 5 * max_thickness_chord_ratio * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4) 

# Setup the problem
prob.setup()

# Define some example inputs
prob.set_val('span', 1)  
prob.set_val('root_chord', 1)  
prob.set_val('tip_chord', 0.5)  
prob.set_val('twist', jnp.linspace(0, 0, 10))  
prob.set_val('thickness_dist', thickness_dist)  


#prob.model.cog.options['airfoil_data_file'] = 'Clark_Y.dat'
prob.model.cog.options['material'] = 'Balsa'
prob.model.cog.options['airfoil_type'] = '2412'

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


