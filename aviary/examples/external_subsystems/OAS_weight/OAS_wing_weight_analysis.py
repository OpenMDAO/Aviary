"""
OpenMDAO component for aerostructural analysis using OpenAeroStruct.

This analysis is based on the aircraft_for_bench_FwFm.csv input data representing a
single-aisle commercial transport aircraft.  The user_mesh method is currently
hard-coded with values taken from the aircraft_for_bench_FwFm data, but should be coded
to use the Aircraft.Wing.* variables for a more general capability.

The OAStructures class performs a structural analysis of the given wing
by applying aeroelastic loads computed at the cruise condition and a
2.5g maneuver at Mach 0.64 at sea level.  The optimization determines
the optimum wing skin thickness, spar cap thickness, wing twist, wing t/c
and maneuver angle of attack that satisfies strength constraints.

The only Aviary input driving the design is fuel mass, but other variables
may be included as well.

OAStructures returns the optimized wing mass and the fuel mass burned.
Currently, only the wing mass is used to override the Aviary variable
Aircraft.Wing.MASS.

"""

import time
import warnings

import numpy as np
import openmdao.api as om

try:
    import ambiance
except ImportError:
    raise ImportError(
        "ambiance package not found. You can install it by running 'pip install ambiance'."
    )

try:
    import openaerostruct
except ImportError:
    raise ImportError(
        "openaerostruct package not found. You can install it by running 'pip install openaerostruct'."
    )

from ambiance import Atmosphere
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta


def user_mesh():
    """Generate a user defined mesh which is model specific."""
    # Planform specifications
    half_span = 17.9573
    kink_location = 4.9544

    root_chord = 5.5668
    kink_chord = 4.1302
    tip_chord = 1.5084
    inboard_LE_sweep = 25.0
    outboard_LE_sweep = 25.0

    # Mesh specifications
    nx = 2
    ny_outboard = 5
    ny_inboard = 3

    # Initialize the 3-D mesh object. Indexing: Chordwise, spanwise, then the 3-D coordinates.
    # We use ny_inboard+ny_outboard-1 because the 2 segments share the nodes where they connect.
    mesh = np.zeros((nx, ny_inboard + ny_outboard - 1, 3))

    # The form of this 3-D array can be confusing initially.
    # For each node, we are providing the x, y, and z coordinates.
    # x is streamwise, y is spanwise, and z is up.
    # For example, the node for the leading edge at the tip would be specified as mesh[0, 0, :] = np.array([x, y, z]).
    # And the node at the trailing edge at the root would be mesh[nx-1, ny-1, :] = np.array([x, y, z]).
    # We only provide the right half of the wing here because we use symmetry.
    # Print elements of the mesh to better understand the form.

    ####### THE Z-COORDINATES ######
    # Assume no dihedral, so set the z-coordinate for all the points to 0.
    mesh[:, :, 2] = 0.0

    ####### THE Y-COORDINATES ######
    # Using uniform spacing for the spanwise locations of all the nodes within each of the two trapezoidal segments:
    # Outboard
    mesh[:, :ny_outboard, 1] = np.linspace(half_span, kink_location, ny_outboard)
    # Inboard
    mesh[:, ny_outboard : ny_outboard + ny_inboard, 1] = np.linspace(kink_location, 0, ny_inboard)[
        1:
    ]

    ###### THE X-COORDINATES ######
    # Start with the leading edge and create some intermediate arrays that we will use
    x_LE = np.zeros(ny_inboard + ny_outboard - 1)

    array_for_inboard_leading_edge_x_coord = np.linspace(0, kink_location, ny_inboard) * np.tan(
        inboard_LE_sweep / 180.0 * np.pi
    )

    array_for_outboard_leading_edge_x_coord = (
        np.linspace(0, half_span - kink_location, ny_outboard)
        * np.tan(outboard_LE_sweep / 180.0 * np.pi)
        + np.ones(ny_outboard) * array_for_inboard_leading_edge_x_coord[-1]
    )

    x_LE[:ny_inboard] = array_for_inboard_leading_edge_x_coord
    x_LE[ny_inboard : ny_inboard + ny_outboard] = array_for_outboard_leading_edge_x_coord[1:]

    # Then the trailing edge
    x_TE = np.zeros(ny_inboard + ny_outboard - 1)

    array_for_inboard_trailing_edge_x_coord = np.linspace(
        array_for_inboard_leading_edge_x_coord[0] + root_chord,
        array_for_inboard_leading_edge_x_coord[-1] + kink_chord,
        ny_inboard,
    )

    array_for_outboard_trailing_edge_x_coord = np.linspace(
        array_for_outboard_leading_edge_x_coord[0] + kink_chord,
        array_for_outboard_leading_edge_x_coord[-1] + tip_chord,
        ny_outboard,
    )

    x_TE[:ny_inboard] = array_for_inboard_trailing_edge_x_coord
    x_TE[ny_inboard : ny_inboard + ny_outboard] = array_for_outboard_trailing_edge_x_coord[1:]

    for i in range(0, ny_inboard + ny_outboard - 1):
        mesh[:, i, 0] = np.linspace(np.flip(x_LE)[i], np.flip(x_TE)[i], nx)

    return mesh


class OAStructures(om.ExplicitComponent):
    """OAS structure component."""

    def initialize(self):
        self.options.declare('symmetry', default=True, desc='wing symmetry (True or False)')
        self.options.declare('chord_cos_spacing', default=0, desc='chordwise cosine spacing')
        self.options.declare('span_cos_spacing', default=0, desc='spanwise cosine spacing')
        self.options.declare(
            'num_box_cp', default=0, desc='number of chordwise CPs on the structural box'
        )
        self.options.declare('num_twist_cp', default=0, desc='number of twist CPs')
        self.options.declare(
            'S_ref_type', default='wetted', desc='type of computed wing area (wetted or projected)'
        )
        self.options.declare(
            'fem_model_type', default='wingbox', desc='type of FEM model (wingbox or tube)'
        )
        self.options.declare('with_viscous', default=True, desc='viscous drag selection')
        self.options.declare('with_wave', default=True, desc='wave drag selection')
        self.options.declare('k_lam', default=0.05, desc='fraction of chord with laminar flow')
        self.options.declare(
            'c_max_t', default=0.38, desc='chordwise location of maximum thickness'
        )
        self.options.declare('E', default=73.1e9, desc='Youngs Modulus for AL 7073 [Pa]')
        self.options.declare(
            'G', default=(73.1e9 / 2 / 1.33), desc='Shear Modulus for AL 7073 [Pa]'
        )
        self.options.declare(
            'yield', default=(420.0e6 / 1.5), desc='Allowable yield stress for AL 7073 [Pa]'
        )
        self.options.declare('mrho', default=2.78e3, desc='Material density for AL 7073 [kg/m^3]')
        self.options.declare(
            'strength_factor_for_upper_skin',
            default=1.0,
            desc='the yield stress is multiplied by this factor for the upper skin',
        )
        self.options.declare(
            'wing_weight_ratio',
            default=1.00,
            desc='Ratio of the total wing weight (including non-structural components) to the wing structural weight.',
        )
        self.options.declare(
            'exact_failure_constraint', default=False, desc='if false, use KS function'
        )
        self.options.declare(
            'struct_weight_relief',
            default=True,
            desc='if true, use structural weight as inertia relief loads',
        )
        self.options.declare(
            'distributed_fuel_weight',
            default=True,
            desc='Set True to distribute the fuel weight across the entire wing.',
        )
        self.options.declare('n_point_masses', default=0, desc='number of point masses')
        self.options.declare(
            'fuel_density',
            default=803.0,
            desc='fuel density (only needed if the fuel-in-wing volume constraint is used) [kg/m^3]',
        )

    def setup(self):
        self.add_input('box_upper_x', units='unitless', shape=(self.options['num_box_cp'],))
        self.add_input('box_lower_x', units='unitless', shape=(self.options['num_box_cp'],))
        self.add_input('box_upper_y', units='unitless', shape=(self.options['num_box_cp'],))
        self.add_input('box_lower_y', units='unitless', shape=(self.options['num_box_cp'],))
        self.add_input('twist_cp', units='deg', shape=(self.options['num_twist_cp'],))
        self.add_input('spar_thickness_cp', units='m', shape=(self.options['num_twist_cp'],))
        self.add_input('skin_thickness_cp', units='m', shape=(self.options['num_twist_cp'],))
        self.add_input('t_over_c_cp', units='unitless', shape=(self.options['num_twist_cp'],))
        self.add_input('airfoil_t_over_c', units='unitless')
        self.add_input('fuel', val=0.0, units='kg')
        self.add_input('fuel_reserve', val=0.0, units='kg')
        self.add_input('CL0', val=0.0, units='unitless')
        self.add_input('CD0', val=0.0, units='unitless')
        self.add_input('cruise_Mach', val=0.0, units='unitless')
        self.add_input('cruise_altitude', val=0.0, units='m')
        self.add_input('cruise_range', val=0.0, units='m')
        self.add_input('cruise_SFC', val=0.0, units='1/s')
        self.add_input('engine_mass', val=0.0, units='kg')
        self.add_input('engine_location', val=np.array([0.0, 0.0, 0.0]), units='m')

        self.add_output('wing_weight', units='kg')
        self.add_output('fuel_burn', units='kg')

        self.declare_partials(of=['*'], wrt=['fuel'], method='fd')

        self.previous_DV_values = {}

    def compute(self, inputs, outputs):
        start_time = time.time()

        # perform the wing structural optimization and return the wing weight

        mesh = user_mesh()

        # surface options dictionary
        # fmt: off
        surf_dict = {
            # surface name
            'name': 'meshwing',

            # wing symmetry
            'symmetry': self.options['symmetry'],

            # wing mesh
            'mesh': mesh,
            
            # wing definition
            'S_ref_type': self.options['S_ref_type'],
            'fem_model_type': self.options['fem_model_type'],

            # wing thickness data
            'data_x_upper': inputs['box_upper_x'],
            'data_x_lower': inputs['box_lower_x'],
            'data_y_upper': inputs['box_upper_y'],
            'data_y_lower': inputs['box_lower_y'],

            # wing sizing parameters
            'twist_cp': inputs['twist_cp'],
            'spar_thickness_cp': inputs['spar_thickness_cp'],
            'skin_thickness_cp': inputs['skin_thickness_cp'],
            't_over_c_cp': inputs['t_over_c_cp'],
            'original_wingbox_airfoil_t_over_c': inputs['airfoil_t_over_c'],

            # Aerodynamic deltas.
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            # They can be used to account for things that are not included, such as contributions from the fuselage, camber, etc.
            'CL0': inputs['CL0'][0],
            'CD0': inputs['CD0'][0],
            'with_viscous': self.options['with_viscous'],
            'with_wave': self.options['with_wave'],

            # Airfoil properties for viscous drag calculation
            'k_lam': self.options['k_lam'],

            # flow, used for viscous drag
            'c_max_t': self.options['c_max_t'],
            
            # Structural values
            'E': self.options['E'],
            'G': self.options['G'],
            'yield': self.options['yield'],
            'mrho': self.options['mrho'],
            'strength_factor_for_upper_skin': self.options['strength_factor_for_upper_skin'],
            'wing_weight_ratio': self.options['wing_weight_ratio'],
            'exact_failure_constraint': self.options['exact_failure_constraint'],
            
            # structural weight factors
            'struct_weight_relief': self.options['struct_weight_relief'],
            'distributed_fuel_weight': self.options['distributed_fuel_weight'],
            
            # point masses
            'n_point_masses': self.options['n_point_masses'],
            
            # fuel factors
            'fuel_density': self.options['fuel_density'],
            'Wf_reserve': inputs['fuel_reserve'][0],
        }
        # fmt: on

        # define the surfaces
        surfaces = [surf_dict]

        # Create the problem and assign the model group
        prob = om.Problem()

        fuel = inputs['fuel'][0]

        mach = np.array([inputs['cruise_Mach'][0], 0.64])
        altitude = np.array([inputs['cruise_altitude'][0], 0.0])
        cruise_range = inputs['cruise_range'][0]

        atmos = Atmosphere(altitude)

        speed_of_sound = atmos.speed_of_sound
        rho = atmos.density
        dynamic_viscosity = atmos.dynamic_viscosity

        velocity = np.array([mach[0] * speed_of_sound[0], mach[1] * speed_of_sound[1]])

        # set values on the subproblem based on what's passed in from Aviary
        # Add problem information as an independent variables component
        indep_var_comp = om.IndepVarComp()
        indep_var_comp.add_output(
            'Mach_number', val=mach, desc='Mach number for cruise and for maneuver'
        )
        indep_var_comp.add_output(
            'v', val=velocity, units='m/s', desc='velocity for cruise and for maneuver'
        )
        indep_var_comp.add_output(
            're',
            val=np.array(
                [
                    rho[0] * velocity[0] * 1.0 / (dynamic_viscosity[0]),
                    rho[1] * velocity[1] * 1.0 / (dynamic_viscosity[1]),
                ]
            ),
            desc='Reynolds number per unit length for cruise and for maneuver',
            units='1/m',
        )
        indep_var_comp.add_output(
            'rho', val=rho, units='kg/m**3', desc='atmostheric density for cruise and for maneuver'
        )
        indep_var_comp.add_output(
            'speed_of_sound',
            val=speed_of_sound,
            units='m/s',
            desc='speed of sound for cruise and for maneuver',
        )
        indep_var_comp.add_output(
            'CT', val=0.53 / 3600, units='1/s', desc='cruise thrust specific fuel consumption'
        )
        indep_var_comp.add_output('R', val=cruise_range, units='m', desc='cruise range')
        indep_var_comp.add_output(
            'W0_without_point_masses', val=40.0e3 + fuel + surf_dict['Wf_reserve'], units='kg'
        )
        indep_var_comp.add_output(
            'load_factor', val=np.array([1.0, 2.5]), desc='load factor for cruise and for maneuver'
        )
        indep_var_comp.add_output('alpha', val=0.0, units='deg')
        indep_var_comp.add_output('alpha_maneuver', val=0.0, units='deg')
        indep_var_comp.add_output('empty_cg', val=np.zeros((3)), units='m')
        indep_var_comp.add_output('fuel_mass', val=fuel, units='kg')

        # add the problem variables subsystem
        prob.model.add_subsystem('prob_vars', indep_var_comp, promotes=['*'])

        # point masses
        point_masses = inputs['engine_mass']
        point_mass_locations = inputs['engine_location']

        indep_var_comp.add_output('point_masses', val=point_masses, units='kg')
        indep_var_comp.add_output('point_mass_locations', val=point_mass_locations, units='m')

        # add an ExecComp subsystem to compute the actual W0 to be used within OAS based on the sum of the point mass and other W0 weight
        prob.model.add_subsystem(
            'W0_comp',
            om.ExecComp('W0 = W0_without_point_masses + 2 * sum(point_masses)', units='kg'),
            promotes=['*'],
        )

        # Loop over each surface in the surfaces list
        for surface in surfaces:
            # Get the surface name and create a group to contain components only for this surface
            name = surface['name']

            aerostruct_group = AerostructGeometry(surface=surface)

            # Add groups to the problem with the name of the surface.
            prob.model.add_subsystem(name, aerostruct_group)

        # Loop through and add the specified number of aerostruct points
        for i in range(2):
            point_name = 'AS_point_{}'.format(i)

            # Create the aerostruct point group and add it to the model
            AS_point = AerostructPoint(surfaces=surfaces, internally_connect_fuelburn=False)
            prob.model.add_subsystem(point_name, AS_point)

            # Connect flow properties to the analysis point
            prob.model.connect('v', point_name + '.v', src_indices=[i])
            prob.model.connect('Mach_number', point_name + '.Mach_number', src_indices=[i])
            prob.model.connect('re', point_name + '.re', src_indices=[i])
            prob.model.connect('rho', point_name + '.rho', src_indices=[i])
            prob.model.connect('CT', point_name + '.CT')
            prob.model.connect('R', point_name + '.R')
            prob.model.connect('W0', point_name + '.W0')
            prob.model.connect('speed_of_sound', point_name + '.speed_of_sound', src_indices=[i])
            prob.model.connect('empty_cg', point_name + '.empty_cg')
            prob.model.connect('load_factor', point_name + '.load_factor', src_indices=[i])
            prob.model.connect('fuel_mass', point_name + '.total_perf.L_equals_W.fuelburn')
            prob.model.connect('fuel_mass', point_name + '.total_perf.CG.fuelburn')

            for surface in surfaces:
                name = surface['name']

                if surf_dict['distributed_fuel_weight']:
                    prob.model.connect(
                        'load_factor', point_name + '.coupled.load_factor', src_indices=[i]
                    )

                com_name = point_name + '.' + name + '_perf.'
                prob.model.connect(
                    name + '.local_stiff_transformed',
                    point_name + '.coupled.' + name + '.local_stiff_transformed',
                )

                prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')

                # Connect aerodyamic mesh to coupled group mesh
                prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

                if surf_dict['struct_weight_relief']:
                    prob.model.connect(
                        name + '.element_mass', point_name + '.coupled.' + name + '.element_mass'
                    )

                # Connect performance calculation variables
                prob.model.connect(name + '.nodes', com_name + 'nodes')
                prob.model.connect(
                    name + '.cg_location', point_name + '.' + 'total_perf.' + name + '_cg_location'
                )
                prob.model.connect(
                    name + '.structural_mass',
                    point_name + '.' + 'total_perf.' + name + '_structural_mass',
                )

                # Connect wingbox properties to von Mises stress calcs
                prob.model.connect(name + '.Qz', com_name + 'Qz')
                prob.model.connect(name + '.J', com_name + 'J')
                prob.model.connect(name + '.A_enc', com_name + 'A_enc')
                prob.model.connect(name + '.htop', com_name + 'htop')
                prob.model.connect(name + '.hbottom', com_name + 'hbottom')
                prob.model.connect(name + '.hfront', com_name + 'hfront')
                prob.model.connect(name + '.hrear', com_name + 'hrear')

                prob.model.connect(name + '.spar_thickness', com_name + 'spar_thickness')
                prob.model.connect(name + '.t_over_c', com_name + 't_over_c')

                coupled_name = point_name + '.coupled.' + name
                prob.model.connect('point_masses', coupled_name + '.point_masses')
                prob.model.connect('point_mass_locations', coupled_name + '.point_mass_locations')

        # use only the first surface for constraints
        surface = surfaces[0]
        name = surface['name']

        prob.model.connect('alpha', 'AS_point_0' + '.alpha')
        prob.model.connect('alpha_maneuver', 'AS_point_1' + '.alpha')

        # Here we add the fuel volume constraint component to the model
        prob.model.add_subsystem('fuel_vol_delta', WingboxFuelVolDelta(surface=surfaces[0]))
        prob.model.connect(name + '.struct_setup.fuel_vols', 'fuel_vol_delta.fuel_vols')
        prob.model.connect('AS_point_0.fuelburn', 'fuel_vol_delta.fuelburn')

        if surf_dict['distributed_fuel_weight']:
            prob.model.connect(
                name + '.struct_setup.fuel_vols',
                'AS_point_0.coupled.' + name + '.struct_states.fuel_vols',
            )
            prob.model.connect(
                'fuel_mass', 'AS_point_0.coupled.' + name + '.struct_states.fuel_mass'
            )

            prob.model.connect(
                name + '.struct_setup.fuel_vols',
                'AS_point_1.coupled.' + name + '.struct_states.fuel_vols',
            )
            prob.model.connect(
                'fuel_mass', 'AS_point_1.coupled.' + name + '.struct_states.fuel_mass'
            )

        # add an ExecComp to compute the fuel difference
        comp = om.ExecComp('fuel_diff = (fuel_mass - fuelburn) / fuelburn', units='kg')

        # add a fuel difference subsystem
        prob.model.add_subsystem(
            'fuel_diff', comp, promotes_inputs=['fuel_mass'], promotes_outputs=['fuel_diff']
        )
        prob.model.connect('AS_point_0.fuelburn', 'fuel_diff.fuelburn')

        # add an objective function
        prob.model.add_objective('AS_point_0.fuelburn', scaler=1e-5)

        # add design variables
        prob.model.add_design_var(name + '.twist_cp', lower=-15.0, upper=15.0, scaler=0.1)
        prob.model.add_design_var(name + '.spar_thickness_cp', lower=0.003, upper=0.1, scaler=1e2)
        prob.model.add_design_var(name + '.skin_thickness_cp', lower=0.003, upper=0.1, scaler=1e2)
        prob.model.add_design_var(
            name + '.geometry.t_over_c_cp', lower=0.07, upper=0.2, scaler=10.0
        )
        prob.model.add_design_var('alpha_maneuver', lower=-15.0, upper=15)

        # add problem constraints
        prob.model.add_constraint('AS_point_0.CL', equals=0.5)
        prob.model.add_constraint('AS_point_1.L_equals_W', equals=0.0)
        prob.model.add_constraint('AS_point_1.' + name + '_perf.failure', upper=0.0)

        prob.model.add_constraint('fuel_vol_delta.fuel_vol_delta', lower=0.0)

        prob.model.add_design_var('fuel_mass', lower=0.0, upper=2e5, scaler=1e-5)
        prob.model.add_constraint('fuel_diff', equals=0.0)

        # set up the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-8
        # Set up the problem
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=om.PromotionWarning)
            prob.setup()

        # change linear solver for aerostructural coupled adjoint
        prob.model.AS_point_0.coupled.linear_solver = om.LinearBlockGS(
            iprint=0, maxiter=30, use_aitken=True
        )
        prob.model.AS_point_1.coupled.linear_solver = om.LinearBlockGS(
            iprint=0, maxiter=30, use_aitken=True
        )
        prob.model.AS_point_0.coupled.nonlinear_solver.options['iprint'] = 0
        prob.model.AS_point_1.coupled.nonlinear_solver.options['iprint'] = 0

        # Loop through self.previous_DV_values and set each prob value
        for key, value in self.previous_DV_values.items():
            prob[key] = value

        # run the problem
        prob.run_driver()

        self.previous_DV_values[name + '.twist_cp'] = prob[name + '.twist_cp']
        self.previous_DV_values[name + '.spar_thickness_cp'] = prob[name + '.spar_thickness_cp']
        self.previous_DV_values[name + '.skin_thickness_cp'] = prob[name + '.skin_thickness_cp']
        self.previous_DV_values[name + '.geometry.t_over_c_cp'] = prob[
            name + '.geometry.t_over_c_cp'
        ]
        self.previous_DV_values['alpha_maneuver'] = prob['alpha_maneuver']

        # output wing weight and fuel burn
        outputs['wing_weight'] = prob[name + '.structural_mass'][0]
        outputs['fuel_burn'] = prob['AS_point_0.fuelburn'][0]

        # calculate execution time
        delta_time = time.time() - start_time
        tm_hrs, remainder = divmod(delta_time, 3600)
        tm_min, remainder = divmod(remainder, 60)
        tm_sec = int(remainder)
        tm_msec = (remainder - tm_sec) * 1000
        print(
            'Structures OAS Compute End --- execution time {:02}:{:02}:{:02}.{:03}'.format(
                int(tm_hrs), int(tm_min), int(tm_sec), int(tm_msec)
            )
        )
