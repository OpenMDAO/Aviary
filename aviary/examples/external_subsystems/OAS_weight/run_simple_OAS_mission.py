'''
This is a simple test mission to demonstrate the inclusion of a
pre-mission user defined external subsystem. The simple mission
is based on input data read from the benchmark data file bench_4.csv,
which represents a single-aisle commercial transport aircraft.  The
OpenAeroStruct (OAS) subsystem is used to compute an optimum wing
mass which will override the Aviary computed wing mass value.

The flag 'use_OAS' is set to 'True' to include the OAS subsystem in
the mission, or set to 'False' to run the mission without the
subsystem so that wing mass values between the 2 methods may be
compared.

'''

import numpy as np
import openmdao.api as om
import aviary.api as av
from aviary.examples.external_subsystems.OAS_weight.OAS_wing_weight_builder import OASWingWeightBuilder

# flag to turn on/off OpenAeroStruct subsystem for comparison testing
use_OAS = True

wing_weight_builder = OASWingWeightBuilder()

# Load the phase_info and other common setup tasks
phase_info = {
    'climb_1': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'optimize_mach': False,
            'optimize_altitude': False,
            'polynomial_control_order': 1,
            'num_segments': 5,
            'order': 3,
            'solve_for_range': False,
            'initial_mach': (0.2, 'unitless'),
            'final_mach': (0.72, 'unitless'),
            'mach_bounds': ((0.18, 0.74), 'unitless'),
            'initial_altitude': (0.0, 'ft'),
            'final_altitude': (32000.0, 'ft'),
            'altitude_bounds': ((0.0, 34000.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'fix_initial': True,
            'constrain_final': False,
            'fix_duration': False,
            'initial_bounds': ((0.0, 0.0), 'min'),
            'duration_bounds': ((64.0, 192.0), 'min'),
        },
        'initial_guesses': {'times': ([0, 128], 'min')},
    },
    'climb_2': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'optimize_mach': False,
            'optimize_altitude': False,
            'polynomial_control_order': 1,
            'num_segments': 5,
            'order': 3,
            'solve_for_range': False,
            'initial_mach': (0.72, 'unitless'),
            'final_mach': (0.72, 'unitless'),
            'mach_bounds': ((0.7, 0.74), 'unitless'),
            'initial_altitude': (32000.0, 'ft'),
            'final_altitude': (34000.0, 'ft'),
            'altitude_bounds': ((23000.0, 38000.0), 'ft'),
            'throttle_enforcement': 'boundary_constraint',
            'fix_initial': False,
            'constrain_final': False,
            'fix_duration': False,
            'initial_bounds': ((64.0, 192.0), 'min'),
            'duration_bounds': ((56.5, 169.5), 'min'),
        },
        'initial_guesses': {'times': ([128, 113], 'min')},
    },
    'descent_1': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'optimize_mach': False,
            'optimize_altitude': False,
            'polynomial_control_order': 1,
            'num_segments': 5,
            'order': 3,
            'solve_for_range': False,
            'initial_mach': (0.72, 'unitless'),
            'final_mach': (0.36, 'unitless'),
            'mach_bounds': ((0.34, 0.74), 'unitless'),
            'initial_altitude': (34000.0, 'ft'),
            'final_altitude': (500.0, 'ft'),
            'altitude_bounds': ((0.0, 38000.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'fix_initial': False,
            'constrain_final': True,
            'fix_duration': False,
            'initial_bounds': ((120.5, 361.5), 'min'),
            'duration_bounds': ((29.0, 87.0), 'min'),
        },
        'initial_guesses': {'times': ([241, 58], 'min')},
    },
    'post_mission': {
        'include_landing': False,
        'constrain_range': True,
        'target_range': (1800., 'nmi'),
    },
}

phase_info['pre_mission'] = {'include_takeoff': False, 'optimize_mass': True}
if use_OAS:
    phase_info['pre_mission']['external_subsystems'] = [wing_weight_builder]

aircraft_definition_file = 'models/test_aircraft/aircraft_for_bench_FwFm_simple.csv'
make_plots = False
max_iter = 100
optimizer = 'SNOPT'


prob = av.AviaryProblem()

prob.load_inputs(aircraft_definition_file, phase_info)
prob.check_inputs()
prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()
prob.link_phases()

driver = prob.driver = om.pyOptSparseDriver()
driver.options["optimizer"] = optimizer
driver.declare_coloring()
driver.opt_settings["Major iterations limit"] = max_iter
driver.opt_settings["Major optimality tolerance"] = 1e-4
driver.opt_settings["Major feasibility tolerance"] = 1e-5
driver.opt_settings["iSumm"] = 6

prob.add_design_variables()
prob.add_objective()
prob.setup()

if use_OAS:
    OAS_sys = 'pre_mission.wing_weight.aerostructures.'
    prob.set_val(OAS_sys + 'box_upper_x', np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32,
                 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]), units='unitless')
    prob.set_val(OAS_sys + 'box_lower_x', np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32,
                 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]), units='unitless')
    prob.set_val(OAS_sys + 'box_upper_y', np.array([0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,
                 0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541]), units='unitless')
    prob.set_val(OAS_sys + 'box_lower_y', np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -
                 0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444]), units='unitless')
    prob.set_val(OAS_sys + 'twist_cp', np.array([-6., -6., -4., 0.]), units='deg')
    prob.set_val(OAS_sys + 'spar_thickness_cp',
                 np.array([0.004, 0.005, 0.008, 0.01]), units='m')
    prob.set_val(OAS_sys + 'skin_thickness_cp',
                 np.array([0.005, 0.01, 0.015, 0.025]), units='m')
    prob.set_val(OAS_sys + 't_over_c_cp',
                 np.array([0.08, 0.08, 0.10, 0.08]), units='unitless')
    prob.set_val(OAS_sys + 'airfoil_t_over_c', 0.12, units='unitless')
    prob.set_val(OAS_sys + 'fuel', 40044.0, units='lbm')
    prob.set_val(OAS_sys + 'fuel_reserve', 3000.0, units='lbm')
    prob.set_val(OAS_sys + 'CD0', 0.0078, units='unitless')
    prob.set_val(OAS_sys + 'cruise_Mach', 0.785, units='unitless')
    prob.set_val(OAS_sys + 'cruise_altitude', 11303.682962301647, units='m')
    prob.set_val(OAS_sys + 'cruise_range', 3500, units='nmi')
    prob.set_val(OAS_sys + 'cruise_SFC', 0.53 / 3600, units='1/s')
    prob.set_val(OAS_sys + 'engine_mass', 7400, units='lbm')
    prob.set_val(OAS_sys + 'engine_location', np.array([25, -10.0, 0.0]), units='m')

prob.set_initial_guesses()
prob.run_aviary_problem('dymos_solution.db', make_plots=False)

print('wing mass = ', prob.model.get_val(av.Aircraft.Wing.MASS, units='lbm'))
