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
np.seterr(divide='raise', invalid='raise')
import matplotlib.pyplot as plt

import aviary.api as av
from aviary.subsystems.aerodynamics.UAV_aero.aero_builder import AeroBuilder
# from aviary.subsystems.aerodynamics.UAV_aero.aero_model import TotalAircraftAero #they are not used so commented out
# from aviary.utils.functions import set_aviary_initial_values
# from aviary.utils.aviary_values import AviaryValues


# Set True while debugging the integrated model.
# Set False for a normal cruise run.
DEBUG_MODEL = True
#import the UAV builders for mass and propulsion
from aviary.subsystems.mass.UAV_mass.mass_builder import MassBuilder
from aviary.subsystems.propulsion.UAV.UAV_Builder import UAVBuilder

from aviary.variable_info.UAV_variables import Aircraft, Dynamic

aero_builder = AeroBuilder(name='UAV_aero')

mass_builder = MassBuilder()

phase_info = {
    'pre_mission': {
    'include_takeoff': False,
    'external_subsystems': [],
    'optimize_mass': False,

    'subsystem_options': {
        'mass': {
            'method': 'external',
        },
        'aerodynamics': {
            'method': 'external',
        },
        'geometry': {
            'method': 'external',
        },  
    },
},

    'cruise': {
        'subsystem_options': {
            'aerodynamics': {'method': 'external'}, # Changed aero_builder to aerodynamics 
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


max_iter = 50
optimizer = 'IPOPT' 
prob.add_driver(
    optimizer,
    max_iter=max_iter,
)
prob.add_design_variables()

prob.model.add_design_var('aircraft:wing:span', lower=0.1, upper=2.0)
prob.model.add_design_var('aircraft:wing:root_chord', lower=0.1, upper=1.0)
prob.model.add_design_var('aircraft:wing:incidence', lower=-5.0, upper=10.0)
prob.model.add_design_var('aircraft:wing:thickness_to_chord', lower=0.05, upper=0.20)
prob.model.add_design_var('aircraft:horizontal_tail:incidence', lower=-5.0, upper=10.0)

prob.model.add_constraint('traj.phases.cruise.rhs_all.lifting_surface_CL', lower=0.01, upper=0.2)
prob.model.add_objective('traj.cruise.t_duration', index=-1)

prob.driver.recording_options['record_desvars'] = False
prob.driver.recording_options['record_responses'] = False
prob.driver.recording_options['record_objectives'] = False
prob.driver.recording_options['record_constraints'] = False

prob.driver.opt_settings.update({
   'tol': 5e-4,
   'constr_viol_tol': 1e-6,
   'acceptable_tol': 1e-5,
   'acceptable_constr_viol_tol': 5e-3,
   'line_search_method': 'filter',
   'alpha_for_y': 'primal'
})



prob.model.add_objective(
    'traj.phases.cruise.t_duration',
    ref=60.0,
)
prob = av.AviaryProblem(verbosity=1)

prob.load_inputs('aviary/validation_cases/validation_data/test_models/small_scale_uav.csv', phase_info=phase_info)
propulsion_builder = UAVBuilder( options=prob.aviary_inputs, name='rc_electric',power_balance_mode='feedforward',)

print('Aero builder:', aero_builder.name)
print('Propulsion builder:', propulsion_builder.name)
print('Mass builder:', mass_builder.name)

prob.load_external_subsystems([ mass_builder, aero_builder,  propulsion_builder,])

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

prob.setup()
prob.set_initial_guesses()
prob.final_setup()
om.n2(
prob,
outfile='uav_aero_full_n2.html',
show_browser=True,
title='UAV Mass-Aero-Propulsion Full Model',
)

# =========================================================
# DEBUGGING BEFORE RUNNING THE MODEL
# =========================================================

if DEBUG_MODEL:
    print('\nASSEMBLED UAV-RELATED SYSTEMS:\n')

    # This is a normal Python list created for debugging.
    assembled_systems = []

    for system in prob.model.system_iter(
        recurse=True,
        include_self=False,
    ):
        pathname = system.pathname
        module_name = system.__class__.__module__
        class_name = system.__class__.__name__

        system_description = (
            f'{pathname} | '
            f'{module_name}.{class_name}'
        )

        # Save a lowercase version for easier searching.
        assembled_systems.append(
            system_description.lower()
        )

        # Print only systems relevant to this integration.
        if any(
            keyword in system_description.lower()
            for keyword in (
                'uav_mass',
                'uav_aero',
                'propulsion.uav',
                'rc_electric',
                'battery',
                'esc',
                'motor',
                'propeller',
                'gasp_based',
                'flops_based',
                'turbofan',
                'engine_deck',
            )
        ):
            print(system_description)


    # -----------------------------------------------------
    # Check that the custom systems are present
    # -----------------------------------------------------

    has_uav_mass = any(
        'subsystems.mass.uav_mass' in system
        for system in assembled_systems
    )

    has_uav_aero = any(
        'subsystems.aerodynamics.uav_aero' in system
        for system in assembled_systems
    )

    has_uav_propulsion = any(
        'subsystems.propulsion.uav' in system
        for system in assembled_systems
    )

    print('\nCUSTOM SUBSYSTEM CHECKS:\n')

    print('Custom UAV mass present:', has_uav_mass)
    print('Custom UAV aero present:', has_uav_aero)
    print('Custom UAV propulsion present:', has_uav_propulsion)

    assert has_uav_mass, (
        'The custom UAV mass subsystem was not built.'
    )

    assert has_uav_aero, (
        'The custom UAV aerodynamic subsystem was not built.'
    )

    assert has_uav_propulsion, (
        'The custom UAV electric propulsion subsystem was not built.'
    )


    # -----------------------------------------------------
    # Check that unwanted large-aircraft systems are absent
    # -----------------------------------------------------

    has_large_aircraft_mass = any(
        (
            'subsystems.mass.gasp_based' in system
            or 'subsystems.mass.flops_based' in system
        )
        for system in assembled_systems
    )
    has_builtin_geometry = any(
    (
        'subsystems.geometry.gasp_based' in system
        or 'subsystems.geometry.flops_based' in system
    )
    for system in assembled_systems
)

    has_conventional_engine = any(
        (
            'turbofan' in system
            or 'engine_deck' in system
        )
        for system in assembled_systems
    )

    print('\nUNWANTED SUBSYSTEM CHECKS:\n')

    print(
        'Built-in GASP/FLOPS mass present:',
        has_large_aircraft_mass,
    )
    print(
        'Built-in GASP/FLOPS geometry present:',
        has_builtin_geometry,
    )
    print(
        'Turbofan or EngineDeck present:',
        has_conventional_engine,
    )

    assert not has_large_aircraft_mass, (
        'A built-in GASP/FLOPS large-aircraft mass '
        'subsystem is present.'
    )

    assert not has_conventional_engine, (
        'A conventional turbofan or EngineDeck '
        'propulsion model is present.'
    )

# =========================================================
# NORMAL MODEL EXECUTION
# This must remain outside the DEBUG_MODEL block.
# =========================================================

print('\nRUNNING THE INTEGRATED MODEL...\n')

prob.run_model()

print('\nMODEL RUN COMPLETED.\n')


# =========================================================
# DEBUGGING AFTER THE MODEL RUNS
# =========================================================

if DEBUG_MODEL:
    print('\nALPHA COMPONENT INPUTS:\n')

    prob.model.list_inputs(
        includes=['*alpha_comp*'],
        val=True,
        units=True,
        prom_name=True,
        print_arrays=True,
    )



#Commented out get_val's are not recognized at the moment and I don't know why
print('Lift:', prob.get_val('traj.cruise.rhs_all.lift', units='lbf')) 
print('Drag:', prob.get_val('traj.cruise.rhs_all.drag', units='lbf'))
print('CL:',prob.get_val('traj.cruise.rhs_all.lift_coefficient'),)
print('CD:', prob.get_val('traj.cruise.rhs_all.drag_coefficient'))

print('CD_fus:', prob.get_val('traj.cruise.rhs_all.CD_fus'))
print('CD_vtail:', prob.get_val('traj.cruise.rhs_all.CD_vtail'))
print('CD_gear:', prob.get_val('traj.cruise.rhs_all.CD_gear'))
print('Lifting surface CD:', prob.get_val('traj.cruise.rhs_all.lifting_surface_CD'))

print('Fuselage length:', prob.get_val('aircraft:fuselage:length'))
print('Fuselage height:', prob.get_val('aircraft:fuselage:max_height'))
print('Angle of attack:', prob.get_val('traj.cruise.rhs_all.alpha'))
print('Wing span:', prob.get_val(Aircraft.Wing.SPAN))