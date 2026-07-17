'''
This example tries to run an optimization on t_duration based on the aero_model but 
doesn't work right now because of the multiple promoted outputs error - suggests issues with
wiring or loading of external subsystems. 

THE ERROR: 
    output traj.cruise.rhs_all.drag refers to multiple outputs: traj.phases.cruise.rhs_all.
    core_aerodynamics.total_aircraft_drag.drag and traj.phases.cruise.rhs_all.solver_sub.aerodynamics.
    Drag.drag.simple_drag.drag... similar error also has come up for lift, etc. 

    This implies that the external subsystem is not replacing core aviary aero, but rather that they 
    are both being looked at simultaneously. 

    My suspicions of this error lie primarily in the use of OAS_aero_analysis in aero_example and in 
    the possibility of the external subsystem being loaded incorrecty in phase_info/phase_info being 
    loaded incorrectly in general
'''

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

import aviary.api as av
from aviary.subsystems.aerodynamics.UAV_aero.aero_builder import AeroBuilder
from aviary.subsystems.aerodynamics.UAV_aero.aero_model import TotalAircraftAero
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.aviary_values import AviaryValues

from aviary.variable_info.variables import Aircraft, Dynamic

aero_builder = AeroBuilder()

phase_info = {
    'pre_mission': {
        'include_takeoff': False,
        'external_subsystems': [],
        'optimize_mass': True,
    },

    'cruise': {
        'external_subsystems': ['aero_builder'],
        'subsystem_options': {
            'aero_builder': {'method': 'external'},
        },

        'user_options': {
            'num_segments': 1,
            'order': 3,
            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.08, 'unitless'),
            'mach_final': (0.09, 'unitless'),
            'mach_bounds': ((0.07, 0.11), 'unitless'),
            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (520, 'm'),
            'altitude_final': (520, 'm'),
            'altitude_bounds': ((500, 600), 'm'),
            'throttle_enforcement': 'boundary_constraint',
            'time_initial': (0, 's'),
            'time_duration_bounds': ((1.0, 300.0), 's'),
        },

        'initial_guesses': {
            'time': ([0, 250], 's'),
            'distance': ([0, 800], 'm'),
        },
    },

    'post_mission': {
        'target_range': (5, 'm'),
        'include_landing': False,
        'external_subsystems': [],
    },
}

phase_info['cruise']['subsystem_options']['aero_builder'] = {'method': 'external'}
phase_info['cruise']['user_options']['num_segments'] = 1

max_iter = 50
optimizer = 'IPOPT' 

prob = av.AviaryProblem(verbosity=1)

prob.load_inputs('aviary/validation_cases/validation_data/test_models/aircraft_for_bench_FwFm.csv', phase_info=phase_info)
print("Builder name:", aero_builder.name)
prob.load_external_subsystems([aero_builder])

prob.aviary_inputs.set_val(Dynamic.Mission.ALTITUDE, 520, units='m') 
prob.aviary_inputs.set_val(Dynamic.Mission.VELOCITY, 36, units='m/s')

prob.aviary_inputs.set_val(Aircraft.Wing.SPAN, 1.524, units='m')  
prob.aviary_inputs.set_val(Aircraft.Wing.ROOT_CHORD, 0.508, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.10) 
prob.aviary_inputs.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.266) 
prob.aviary_inputs.set_val(Aircraft.Wing.TAPER_RATIO, 1, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Wing.SWEEP, 0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.INCIDENCE, 0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.CENTER_DISTANCE, 0.511, units='unitless')

prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SPAN, 0.711, units='m')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 0.232, units='m')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.14) 
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 1, units='unitless')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0, units='deg')

prob.aviary_inputs.set_val(Aircraft.VerticalTail.SPAN, 0.3048, units='m')
prob.aviary_inputs.set_val(Aircraft.VerticalTail.ROOT_CHORD, 0.22225, units='m')
prob.aviary_inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.14) 
prob.aviary_inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 1, units='unitless')

prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 0.172, units='m')
prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 0.114, units='m')
prob.aviary_inputs.set_val(Aircraft.Fuselage.LENGTH, 1.190244, units='m')

prob.aviary_inputs.set_val(Dynamic.Vehicle.MASS, 3.787, units='kg')

prob.check_and_preprocess_inputs()
prob.build_model()
#prob.add_driver(optimizer=optimizer, max_iter=max_iter)

#prob.add_design_variables()

#prob.model.add_design_var('aircraft:wing:span', lower=0.1, upper=2.0)
#prob.model.add_design_var('aircraft:wing:root_chord', lower=0.1, upper=1.0)
#prob.model.add_design_var('aircraft:wing:incidence', lower=-5.0, upper=10.0)
#prob.model.add_design_var('aircraft:wing:thickness_to_chord', lower=0.05, upper=0.20)
#prob.model.add_design_var('aircraft:horizontal_tail:incidence', lower=-5.0, upper=10.0)

#prob.model.add_constraint('traj.phases.cruise.rhs_all.lifting_surface_CL', lower=0.01, upper=0.2)
#prob.model.add_objective('traj.cruise.t_duration', index=-1)

#prob.driver.recording_options['record_desvars'] = False
#prob.driver.recording_options['record_responses'] = False
#prob.driver.recording_options['record_objectives'] = False
#prob.driver.recording_options['record_constraints'] = False

#prob.driver.opt_settings.update({
#    'tol': 5e-4,
#    'constr_viol_tol': 1e-6,
#    'acceptable_tol': 1e-5,
#    'acceptable_constr_viol_tol': 5e-3,
#    'line_search_method': 'filter',
#    'alpha_for_y': 'primal'
#})

prob.setup()

prob.run_model()
#prob.set_initial_guesses()

#prob.run_aviary_problem()

#with open("variables.txt", "w") as f:
#    prob.model.list_vars(out_stream=f, print_arrays=True, units=True)

#Commented out get_val's are not recognized at the moment and I don't know why
print('Lift:', prob.get_val('traj.cruise.rhs_all.lift', units='lbf')) 
print('Drag:', prob.get_val('traj.cruise.rhs_all.drag', units='lbf'))
print('CL:', prob.get_val('traj.cruise.rhs_all.lifting_surface_CL'))
print('CD:', prob.get_val('traj.cruise.rhs_all.CD'))

print('CD_fus:', prob.get_val('traj.cruise.rhs_all.CD_fus'))
print('CD_vtail:', prob.get_val('traj.cruise.rhs_all.CD_vtail'))
print('CD_gear:', prob.get_val('traj.cruise.rhs_all.CD_gear'))
print('Lifting surface CD:', prob.get_val('traj.cruise.rhs_all.lifting_surface_CD'))

print('Fuselage length:', prob.get_val('aircraft:fuselage:length'))
print('Fuselage height:', prob.get_val('aircraft:fuselage:max_height'))
print('Angle of attack:', prob.get_val('traj.cruise.rhs_all.alpha'))
print('Wing span:', prob.get_val(Aircraft.Wing.SPAN))