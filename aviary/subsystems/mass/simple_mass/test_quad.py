import jax.numpy as jnp
from jax.scipy.linalg import eigh
import jax.scipy as jsp
import jax
from jax import jit, lax
from scipy.integrate import quad
import quadax
"""
Website: https://rosettacode.org/wiki/Numerical_integration/Gauss-Legendre_Quadrature#Python

"""

# def legendre_poly_and_deriv(n, x):
#     """
#     Compute the Legendre polynomial P_n(x) and its derivative P_n'(x) using 
#     recurrence relations.

#     Parameters:

#     n : int
#         Degree of the Legendre polynomial.
#     x : jnp.ndarray
#         Points at which to evalulate the polynomial.

#     Returns:

#     P_n : jnp.ndarray
#         Value of the Legendre polynomial P_n(x).
#     P_n_prime : jnp.ndarray
#         Value of the derivative P_n'(x).
    
#     """

#     if n == 0:
#         return jnp.ones_like(x), jnp.zeros_like(x)
#     elif n == 1:
#         return x, jnp.ones_like(x)
    
#     # initialize P_0(x) and P_1(x)
#     P0 = jnp.ones_like(x)
#     P1 = x

#     for k in range(2, n+1):
#         P2 = ((2 * k - 1) * x * P1 - (k - 1) * P0) / k
#         P0, P1 = P1, P2 # Shift P_k-1 to P_k, and P_k to P_k+1
    
#     # Compute derivative using recurrence relation
#     P_n = P1
#     P_n_prime = n * (x * P_n - P0) / (x**2 - 1)

#     return P_n, P_n_prime

# def legendre_roots_and_weights(n, tol=1e-15, max_iter=10):
#     """
#     Compute the Gauss-Legendre quadrature nodes (roots) and weights using the 
#     Newton method.

#     Parameters:

#     n : int
#         Number of quadrature points.
#     tol : float, optional
#         Convergence tolerance for Newton's method.
    
#     Returns:

#     roots : jnp.ndarray
#         Legendre polynomial roots (Gauss-Legendre nodes).
#     weights : jnp.ndarray
#         Gauss-Legendre quadrature weights.
    
#     """
    
#     # Initial guess for roots: use Chebyshev approximation
#     roots = jnp.cos(jnp.pi * (4 * jnp.arange(1, n + 1) - 1) / (4 * n + 2))

#     # Newton's method
#     for _ in range(max_iter):
#         P_n, P_n_prime = legendre_poly_and_deriv(n, roots)
#         roots -= P_n / P_n_prime

#     # Compute weights
#     _, P_n_prime = legendre_poly_and_deriv(n, roots)
#     weights = 2 / ((1 - roots**2) * P_n_prime**2)

#     return roots, weights




# def gauss_quad(f, n, a, b):
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
#     x, w = legendre_roots_and_weights(n)

#     x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
#     w_mapped = 0.5 * (b - a) * w
#     integral = jnp.sum(w_mapped * f(x_mapped))  # Compute weighted sum
#     return integral.astype(jnp.float64)

# def legendre_roots_and_weights(n):
#     """
#     Compute nodes and weights for Legendre-Gauss quadrature using JAX.

#     Parameters: 

#     beta: 
#             coefficient coming form Gram-Schmidt process, derived from recursive relation
#             of Legendre polynomials.

#         T:
#             symmetric, tri-diagonal Jacobi matrix derived using Golub-Welsch theorem. Eigenvalues
#             of T correspond to roots of Legendre polynomials
        
#         Eigenvalues, Eigenvectors:
#             The eigenvalues of T and their respective eigenvectors, calculated using 
#             jax.scipy.special.eigh function.
        
#         weights:
#             Weights used for Gaussian-Legendre quadrature, calculated using the first
#             component of the normalized eigenvector, derived from Golub & Welsch (1969). 
#             Coefficient 2 comes from an application of Rodrigues' formula for Legendre
#             polynomials and applying the orthonormality property for Legendre polynomials
#             over the interval [-1, 1] with weight = 1.

#         Returns:

#         float
#             Roots of the Legendre polynomials (eigenvalues of T)
#         float
#             weights used for G-L quadrature (formula from Golub & Welsch 1969)

#         References: 

#         [1] Golub, Gene H., and John H. Welsch. "Calculation of Gauss quadrature rules." 
#             Mathematics of computation 23.106 (1969): 221-230.
        
#         [2] André Ronveaux, Jean Mawhin, Rediscovering the contributions of Rodrigues on 
#             the representation of special functions, Expositiones Mathematicae, Volume 23, 
#             Issue 4, 2005, Pages 361-369, ISSN 0723-0869,
#             https://doi.org/10.1016/j.exmath.2005.05.001. 

#     """

#     i = jnp.arange(1, n)
#     beta = i / jnp.sqrt(4 * i**2 - 1)
#     T = jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)  # Tridiagonal matrix
#     eigenvalues, eigenvectors = jax.jit(jnp.linalg.eigh)(T)  # Compute eigenvalues (roots) and eigenvectors
#     roots = eigenvalues
#     weights = 2 * (eigenvectors[0, :] ** 2)  # Compute weights from first row of eigenvectors
        
#     return roots, weights


# def gauss_quad(f, n, a, b):
#     """
#         Computes the integral of f(x) over [a, b] using Gaussian quadrature with n points.

#         Parameters: 

#         f : function
#             The function to integrate.
        
#         n : int
#             The number of quadrature points.
        
#         a, b : float
#             Integration limits.

#         Returns:

#         float
#             The approximated integral of f(x) over [a, b].

#     """
#     x, w = legendre_roots_and_weights(n)

#     x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)

#     integral = jnp.sum(w * f(x_mapped)) * ((b - a) / 2)  # Compute weighted sum
#     return integral.astype(jnp.float64)


# span = 1
# root_chord = 1
# tip_chord = 0.5
# density = 130
# max_thickness = 0.12
# n = 1000

# test_f = lambda x: density * (5 * max_thickness * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)) * (root_chord - (root_chord - tip_chord) * (x / span)) * span
# integral_value = gauss_quad(test_f, n, 0, 1)
# scipy_int, _ = quad(test_f, 0, 1)
# print(f"Gauss-Legendre Quadrature Result (n={n}): {integral_value:.10f}")
# print("Expected: 4.22032, Computed with scipy.integrate.quad: ", scipy_int)

def Legendre(n, x):
    x = jnp.array(x)
    if n == 0:
        return x * 0 + 1.0
    elif n == 1:
        return x
    else:
        return ((2.0 * n - 1.0) * x * Legendre(n-1, x) - (n - 1) * Legendre(n-2, x)) / n
    
def DLegendre(n, x):
    x = jnp.array(x)
    if n == 0:
        return x * 0
    elif n == 1:
        return x * 0 + 1.0
    else: 
        return (n / (x**2 - 1.0)) * (x * Legendre(n, x) - Legendre(n-1, x))
    
def LegendreRoots(polyorder, tolerance=1e-20):
    if polyorder < 2:
        err = 1
    else:
        roots = []
        for i in range(1, jnp.int_(polyorder / 2 + 1)):
            x = jnp.cos(jnp.pi * (i - 0.25) / polyorder + 0.5)
            error = 10 * tolerance
            iters = 0
            while (error > tolerance) and (iters < 1000):
                dx = -Legendre(polyorder, x) / DLegendre(polyorder, x)
                x += dx
                iters += 1
                error = jnp.abs(dx)
            roots.append(x)
        roots = jnp.asarray(roots)
        if (polyorder % 2 == 0):
            roots = jnp.concatenate([-1.0 * roots, roots[::-1]])
        else:
            roots = jnp.concatenate([-1.0 * roots, jnp.array([0.0]), roots[::-1]])
        err = 0
    return [roots, err]

def GaussLegendreWeights(polyorder):
    W = []
    [xis, err] = LegendreRoots(polyorder)
    if err == 0:
        W = 2.0 / ((1.0 - xis**2) * (DLegendre(polyorder, xis)**2))
    else: 
        err = 1
    return [W, xis, err]

def GaussLegendreQuadrature(func, polyorder, a, b):
    [Ws, xs, err] = GaussLegendreWeights(polyorder)
    if err == 0:
        ans = (b-a) * 0.5 * jnp.sum(Ws * func((b - a) * 0.5 * xs + (b + a) * 0.5))
    else:
        err = 1
        ans = None

    return [ans, err]

span = 1
root_chord = 1
tip_chord = 0.5
density = 130
max_thickness = 0.12

test_f = lambda x: density * (5 * max_thickness * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)) * (root_chord - (root_chord - tip_chord) * (x / span)) * span

order = 3
[Ws, xs, err] = GaussLegendreWeights(order)

if err == 0:
    print("Order    : ", order)
    print("Roots    : ", xs)
    print("Weights  : ", Ws)
else:
    print("Roots/Weights evaluation failed")

[ans, err] = GaussLegendreQuadrature(test_f, order, 0, 1)
if err == 0:
    print("Integral : ", ans)
else:
    print("Integral evaluation failed")

epsabs = epsrel = 1e-9

integral, _ = quadax.quadgk(test_f, [0, 1], epsabs=epsabs, epsrel=epsrel)
print("Result using quadax: ", integral)

# Throw away code from wing.py file that I don't want to let go of 

# def compute_partials(self, inputs, J):
    #     span = inputs['span']
    #     root_chord = inputs['root_chord']
    #     tip_chord = inputs['tip_chord']
    #     thickness_dist = inputs['thickness_dist']
    #     twist=jnp.radians(inputs['twist'])
    #     num_sections = self.options['num_sections']
    #     airfoil_type = self.options['airfoil_type'] # NACA airfoil type 
    #     airfoil_data_file = self.options['airfoil_data_file']
    #     material = self.options['material']

    #     # Compute section locations along the span
    #     span_locations = jnp.linspace(0, span, num_sections)
    #     span_locations = span_locations / span
    #     chord_lengths = tip_chord + (root_chord - tip_chord) * (1 - span_locations / span)

    #     # Compute section airfoil geometry
    #     if airfoil_data_file and os.path.exists(airfoil_data_file):
    #         airfoil_data = jnp.loadtxt(airfoil_data_file)
    #         x_coords = airfoil_data[:, 0]
    #         y_coords = airfoil_data[:, 1]

    #         camber, camber_location, max_thickness, thickness, camber_line = self.extract_airfoil_features(x_coords, y_coords)
    #         thickness_dist = thickness
    #     else:
    #         # Parse the NACA airfoil type (4-digit)
    #         camber = int(airfoil_type[0]) / 100.0 # Maximum camber
    #         camber_location = int(airfoil_type[1]) / 10.0 # Location of max camber
    #         max_thickness = int(airfoil_type[2:4]) / 100.0 # Max thickness

    #     # Compute section airfoil geometry
    #     n_points = num_sections
    #     x_points = jnp.linspace(0, 1, n_points)
    #     dx = 1 / (num_sections - 1)

    #     #A_ref, err = quad(lambda x: jnp.interp(x, x_points, thickness_dist), 0, 1, limit=100) # \int_0^1 t(x) dx
    #     A_ref, _ = quad(lambda x: self.airfoil_thickness(x, max_thickness), 0, 1)

    #     density, _ = materials.get_item(material)

    #     total_moment_x = 0
    #     total_moment_y = 0
    #     total_moment_z = 0
    #     total_weight = 0

    #     rotated_x_vals = jnp.zeros(num_sections)
    #     rotated_z_vals = jnp.zeros(num_sections)
    #     #section_weights = jnp.zeros(num_sections)
    #     section_areas = jnp.zeros(num_sections)
    #     dA_dspan = 0
    #     dA_droot_chord = 0
    #     dA_dtip_chord = 0
    #     dweight_dspan = 0
    #     dmoment_x_dtwist = jnp.zeros(num_sections)
    #     dmoment_z_dtwist = jnp.zeros(num_sections)
    #     dweight_dthickness = jnp.zeros(num_sections)

    #     for i, location in enumerate(span_locations):
    #         # Calculate the chord for this section
    #         chord = root_chord - (root_chord - tip_chord) * (location / span) # Assuming linear variation from root to tip

    #         # Apply twist 
    #         twist_angle = twist[i]

    #         section_area, centroid_x, centroid_z = self.compute_airfoil_geometry(chord, camber, camber_location, thickness_dist, x_points, dx)
    #         #section_weight = density * section_area * (span / num_sections)
    #         section_weight = density * thickness_dist[i] * chord * (span / num_sections)
    #         centroid_y = location 
            
    #         rotated_x_vals[i] = centroid_x * jnp.cos(twist_angle) - centroid_z * jnp.sin(twist_angle)
    #         rotated_z_vals[i] = centroid_x * jnp.sin(twist_angle) + centroid_z * jnp.cos(twist_angle)

    #         total_weight += section_weight
    #         total_moment_x += rotated_x_vals[i] * section_weight
    #         total_moment_y += centroid_y * section_weight
    #         total_moment_z += rotated_z_vals[i] * section_weight

    #         #section_weights[i] = section_weight
    #         section_areas[i] = section_area

    #         # For dweight_dspan
    #         #dci_dspan = -(root_chord - tip_chord) * (location / span**2)
    #         #dA_dspan, _ = quad(lambda x: jnp.interp(x, x_points, thickness_dist) * dci_dspan, 0, 1, limit=100)
    #         #dA_ds = jnp.trapz(thickness_dist * dci_dspan, x_points, dx=dx) 
    #         dA_dc_root = jnp.trapz(thickness_dist * (1 - i / num_sections), x_points, dx=dx)
    #         dA_dc_tip = jnp.trapz(thickness_dist * i / num_sections, x_points, dx=dx)
    #         #dA_dspan += dA_ds
    #         dA_droot_chord += dA_dc_root
    #         dA_dtip_chord += dA_dc_tip
    #         #dweight_dspan += density * (section_area / num_sections - dA_ds * span / num_sections)
    #         dmoment_x_dtwist[i] = -section_weight * (centroid_x * jnp.sin(twist_angle) + centroid_z * jnp.cos(twist_angle))
    #         dmoment_z_dtwist[i] = section_weight * (centroid_x * jnp.cos(twist_angle) - centroid_z * jnp.sin(twist_angle))
    #         dweight_dthickness[i] = density * span * chord / num_sections # dW_total / dthickness_dist -- vector value
    
    #     dweight_droot_chord = jnp.sum(density * A_ref * span * ((num_sections - 1) / (2 * num_sections))) # ~ 1/2 for large N 
    #     dweight_dtip_chord = jnp.sum(density * A_ref * span * ((num_sections + 1) / (2 * num_sections))) # ~ 1/2 for large N

    #     dweight_dspan, _ = (density / span) * dblquad(lambda y, x: self.airfoil_thickness(x, max_thickness) * (root_chord - (root_chord - tip_chord) * (y / span)), 0, 1, 0, span)

    #     J['total_weight', 'span'] = dweight_dspan
    #     J['total_weight', 'root_chord'] = dweight_droot_chord
    #     J['total_weight', 'tip_chord'] = dweight_dtip_chord
    #     J['total_weight', 'thickness_dist'] = dweight_dthickness
    #     J['total_weight', 'twist'] = 0

    #     dxcg_droot_chord = 0
    #     dzcg_droot_chord = 0
    #     dxcg_dtip_chord = 0
    #     dzcg_dtip_chord = 0
    #     for i, location in enumerate(span_locations):
    #         dxcg_dcroot = jnp.sum(
    #             jnp.trapz(
    #                 (x_points * thickness_dist * jnp.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * jnp.sin(twist)) * (1 - i / num_sections), x_points, dx=dx
    #                 ) 
    #                 ) / jnp.sum(section_areas) - jnp.sum(
    #         (
    #             jnp.trapz(
    #                 x_points * thickness_dist * chord_lengths * jnp.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * jnp.sin(twist), x_points, dx=dx
    #                 ) * jnp.trapz(
    #                     thickness_dist * (1 - i / num_sections), x_points, dx=dx
    #                     )
    #                     ) 
    #             ) / jnp.sum(section_areas)**2
    #         dxcg_droot_chord += dxcg_dcroot

    #         dzcg_dcroot = jnp.sum(
    #             jnp.trapz(
    #                 (x_points * thickness_dist * (1 - i / num_sections) * jnp.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist) * (1 - i / num_sections) * jnp.cos(twist), x_points, dx=dx
    #             ) 
    #         ) / jnp.sum(section_areas) - jnp.sum(
    #             jnp.trapz(
    #                 x_points * thickness_dist * chord_lengths * jnp.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * jnp.cos(twist), x_points, dx=dx
    #             ) * dA_droot_chord 
    #         ) / jnp.sum(section_areas)**2
    #         dzcg_droot_chord += dzcg_dcroot

    #         dxcg_dctip = jnp.sum(
    #             jnp.trapz(
    #                 (x_points * thickness_dist *jnp.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * jnp.sin(twist)) * (i / num_sections), x_points, dx=dx
    #             ) / section_areas
    #         ) - jnp.sum(
    #             (
    #                 jnp.trapz(
    #                     x_points * thickness_dist * chord_lengths * jnp.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * jnp.sin(twist), x_points, dx=dx
    #                 )
    #             ) * dA_dtip_chord 
    #         ) / jnp.sum(section_areas)**2
    #         dxcg_dtip_chord += dxcg_dctip

    #         dzcg_dctip = jnp.sum(
    #             jnp.trapz(
    #                 x_points * thickness_dist * (i / num_sections) * jnp.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * (i / num_sections) * jnp.cos(twist), x_points, dx=dx
    #             )
    #         ) / jnp.sum(section_areas) - jnp.sum(
    #             jnp.trapz(
    #                 x_points * thickness_dist * chord_lengths * jnp.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * jnp.cos(twist), x_points, dx=dx
    #             ) * dA_dtip_chord
    #         ) / jnp.sum(section_areas)**2
    #         dzcg_dtip_chord += dzcg_dctip

    #     # partials of cog x
    #     J['center_of_gravity_x', 'span'] = 0
    #     J['center_of_gravity_x', 'root_chord'] = dxcg_droot_chord
    #     J['center_of_gravity_x', 'tip_chord'] = dxcg_dtip_chord
    #     J['center_of_gravity_x', 'thickness_dist'] = jnp.sum(
    #         jnp.trapz(
    #             x_points * chord_lengths * jnp.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * chord_lengths * jnp.sin(twist), x_points, dx=dx
    #             ) / section_areas
    #         ) - (
    #             jnp.sum(
    #         jnp.trapz(x_points * thickness_dist * chord_lengths * jnp.cos(twist) - self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist * chord_lengths * jnp.sin(twist), x_points, dx=dx)
    #     ) * jnp.sum(chord_lengths)
    #     ) / jnp.sum(section_areas)**2
    #     J['center_of_gravity_x', 'twist'] = dmoment_x_dtwist / total_weight
        
    #     # For center of gravity in y calculations

    #     sum_area_times_i = 0
    #     sum_darea_times_i = 0
        
    #     for i in range(len(x_points)):
    #         sum_area_times_i += i * section_areas[i]
    #         sum_darea_times_i += i * dA_dspan # for cg_Y calculations

    #     dcg_y_dspan = jnp.sum(sum_area_times_i / (num_sections * section_areas)) #+ jnp.sum((span * sum_darea_times_i) / (num_sections * section_areas)) - jnp.sum((span * sum_area_times_i) / (num_sections * jnp.sum(section_areas)**2)) * dA_dspan   

    #     # partials of cog y
    #     J['center_of_gravity_y', 'span'] = dcg_y_dspan
    #     J['center_of_gravity_y', 'root_chord'] = -total_moment_y / total_weight * dweight_droot_chord / total_weight
    #     J['center_of_gravity_y', 'tip_chord'] = -total_moment_y / total_weight * dweight_dtip_chord / total_weight
    #     J['center_of_gravity_y', 'thickness_dist'] = -total_moment_y / total_weight * dweight_dthickness / total_weight
    #     J['center_of_gravity_y', 'twist'] = 0

    #     # partials of cog z
    #     J['center_of_gravity_z', 'span'] = 0
    #     J['center_of_gravity_z', 'root_chord'] = dzcg_droot_chord
    #     J['center_of_gravity_z', 'tip_chord'] = dzcg_dtip_chord
    #     J['center_of_gravity_z', 'thickness_dist'] = jnp.sum(
    #         jnp.trapz(
    #             x_points * chord_lengths * jnp.sin(twist) + self.airfoil_camber_line(x_points, camber, camber_location) * chord_lengths * jnp.cos(twist), x_points, dx=dx
    #         )
    #     ) / jnp.sum(
    #         section_areas
    #     )
    #     J['center_of_gravity_z', 'twist'] = dmoment_z_dtwist / total_weight


    # def compute_airfoil_geometry(self, chord, camber, camber_location, thickness_dist, x_points, dx): 

    #     #section_area, _ = quad(lambda x: jnp.interp(x, x_points, thickness_dist), 0, 1, limit=100)
    #     section_area = jint.trapezoid(thickness_dist, x_points) # trying jnp.trapz rather than quad to get rid of IntegrationWarning
    #     section_area *= chord

    #     #centroid_x, _ = quad(lambda x: x * jnp.interp(x, x_points, thickness_dist), 0, 1, limit=100)
    #     centroid_x = jint.trapezoid(x_points * thickness_dist, x_points)
    #     centroid_x = (centroid_x * chord) / section_area

    #     #centroid_z, _ = quad(lambda x: self.airfoil_camber_line(x, camber, camber_location) * jnp.interp(x, x_points, thickness_dist), 0, 1, limit=100)
    #     centroid_z = jint.trapezoid(self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist, x_points)
    #     centroid_z = (centroid_z * chord) / section_area
    #     return section_area, centroid_x, centroid_z


    # Loop over each section along the span (3D wing approximation)
        # for i, location in enumerate(span_locations):
        #     # Calculate the chord for this section -- note this is an approximation
        #     chord = root_chord - (root_chord - tip_chord) * (location / span) # Assuming linear variation from root to tip

        #     # Apply twist 
        #     twist_angle = twist[i]

        #     section_area, centroid_x, centroid_z = self.compute_airfoil_geometry(chord, camber, camber_location, thickness_dist, x_points, dx)
        #     centroid_y = i * span / num_sections
        #     #section_weight = density * section_area * (span / num_sections)
        #     section_weight = density * thickness_dist[i] * chord * (span / num_sections)
            
        #     rotated_x = centroid_x * jnp.cos(twist_angle) - centroid_z * jnp.sin(twist_angle)
        #     rotated_z = centroid_x * jnp.sin(twist_angle) + centroid_z * jnp.cos(twist_angle)

        #     total_weight += section_weight
        #     total_moment_x += rotated_x * section_weight
        #     total_moment_y += centroid_y * section_weight
        #     total_moment_z += rotated_z * section_weight
            

        #     # Store the coordinates for plotting later
        #     x_coords.append(rotated_x)
        #     y_coords.append(centroid_y)
        #     z_coords.append(rotated_z)

# Throw away code from fuselage.py

 # def setup_partials(self):
    #     """
    #     Complex step is used for the derivatives for now as they are very complicated to calculate 
    #     analytically. Compute_partials function has the framework for analytical derivatives, but not 
    #     all of them match the check_partials. 
    #     """
    #     self.declare_partials('center_of_gravity_x', ['length', 'diameter', 'taper_ratio', 'curvature', 'thickness'], method='cs')
    #     self.declare_partials('center_of_gravity_y', ['length', 'y_offset', 'curvature'], method='cs')
    #     self.declare_partials('total_weight', ['length', 'diameter', 'thickness'], method='cs')
    
    # def compute_partials(self, inputs, partials):
    #     length = inputs['length']
    #     diameter = inputs['diameter']
    #     taper_ratio = inputs['taper_ratio']
    #     curvature = inputs['curvature']
    #     thickness = inputs['thickness']
    #     y_offset = inputs['y_offset']
    #     z_offset = inputs['z_offset']
    #     is_hollow = inputs['is_hollow']

    #     #custom_fuselage_function = getattr(self, 'custom_fuselage_function', None) # Custom fuselage model function -- if provided

    #     custom_fuselage_data_file = self.options['custom_fuselage_data_file']

    #     material = self.options['material']
    #     num_sections = self.options['num_sections']
        
    #     self.validate_inputs(length, diameter, thickness, taper_ratio, is_hollow)

    #     density, _ = materials.get_item(material)

    #     section_locations = jnp.linspace(0, length, num_sections).flatten()
    #     dx = 1 / (num_sections - 1)
        
    #     total_weight = 0
    #     total_moment_x = 0
    #     total_moment_y = 0
    #     total_moment_z = 0

    #     interpolate_diameter = self.load_fuselage_data(custom_fuselage_data_file)

    #     out_r = jnp.zeros(num_sections)
    #     in_r = jnp.zeros(num_sections)


    #     # Loop through each section
    #     for i, location in enumerate(section_locations):
    #         section_diameter = self.get_section_diameter(location, length, diameter, taper_ratio, interpolate_diameter)
    #         outer_radius = section_diameter / 2.0 
    #         inner_radius = max(0, outer_radius - thickness) if is_hollow else 0

    #         out_r[i] = outer_radius
    #         in_r[i] = inner_radius

    #         section_volume = jnp.pi * (outer_radius**2 - inner_radius**2) * (length / num_sections)
    #         section_weight = density * section_volume

    #         centroid_x, centroid_y, centroid_z = self.compute_centroid(location, length, y_offset, z_offset, curvature, taper_ratio)

    #         total_weight += section_weight
    #         total_moment_x += centroid_x * section_weight
    #         total_moment_y += centroid_y * section_weight
    #         total_moment_z += centroid_z * section_weight

    #     dzcg_dz_offset = jnp.sum(
    #         jnp.trapz(
    #             (1 - section_locations / length) * density * jnp.pi * (out_r**2 - in_r**2), section_locations, dx=dx
    #         )
    #     ) / total_weight
        

    #     partials['center_of_gravity_x', 'length'] = 3 / 4 if taper_ratio > 0 else 1
    #     partials['center_of_gravity_x', 'taper_ratio'] = - (3/4) * length if taper_ratio > 0 else 0
    #     partials['center_of_gravity_x', 'curvature'] = 0
    #     partials['center_of_gravity_x', 'thickness'] = 0

    #     partials['center_of_gravity_y', 'length'] = -y_offset / length
    #     partials['center_of_gravity_y', 'y_offset'] = 1

    #     partials['center_of_gravity_z', 'length'] = -z_offset / length
    #     partials['center_of_gravity_z', 'z_offset'] = dzcg_dz_offset
    #     partials['center_of_gravity_z', 'curvature'] = length**2 / num_sections

    #     partials['total_weight', 'length'] = density * jnp.pi * (diameter**2 - (diameter - 2 * thickness)**2) / num_sections
    #     partials['total_weight', 'diameter'] = 2 * density * jnp.pi * length * diameter / num_sections
    #     partials['total_weight', 'thickness'] = -2 * density * jnp.pi * length * (diameter - thickness) / num_sections