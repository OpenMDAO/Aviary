import openmdao.api as om
import numpy as np
from copy import deepcopy

import aviary.api as av
from aviary.variable_info.enums import AtmosphereModel
from aviary.variable_info.variables import Settings

from aviary.models.external_subsystems.UAV.aerodynamics.aero_builder import AeroBuilder
from aviary.models.external_subsystems.UAV.mass.mass_builder import MassBuilder
from aviary.models.external_subsystems.UAV.propulsion.prop_builder import PropBuilder
from aviary.models.missions.UAV_energy_phase import phase_info as full_phase_info

from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variable_meta_data import ExtendedMetaData as UAVExtendedMetaData
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft, Dynamic, Mission

DEBUG_MODEL = False  # Set to True to enable debugging output

# Build a cruise-only phase_info using UAV_energy_phase
# cruise_phase_info = {
#     'pre_mission': deepcopy(full_phase_info['pre_mission']),
#     'cruise': deepcopy(full_phase_info['cruise']),
#     'post_mission': deepcopy(full_phase_info['post_mission']),
# }

cruise_phase_info = {
    'pre_mission': {
        'include_takeoff': False,
        'external_subsystems': [],
        'optimize_mass': False,

        # CRITICAL FIX: tell Aviary to use your external atmosphere
        'subsystem_options': {
            'atmosphere': {'method': 'external'},
            'aerodynamics': {'method': 'external'},
        },
    },

    'cruise': {
        # CRITICAL FIX: same here
        'subsystem_options': {
            'atmosphere': {'method': 'external'},
            'aerodynamics': {'method': 'external'},
        },

        'user_options': {
            'num_segments': 1,
            'order': 3,

            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.08, 'unitless'),
            'mach_final': (0.08, 'unitless'),
            'mach_bounds': ((0.07, 0.11), 'unitless'),

            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (520, 'm'),
            'altitude_final': (520, 'm'),
            'altitude_bounds': ((500, 600), 'm'),

            'throttle_enforcement': 'control',
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

        # CRITICAL FIX: also here (post_mission uses the ODE too)
        'subsystem_options': {
            'atmosphere': {'method': 'external'},
            'aerodynamics': {'method': 'external'},
        },
    },
}

prob = av.AviaryProblem(verbosity=1, meta_data=UAVExtendedMetaData)
prob.options['group_by_pre_opt_post'] = True

#load inputs
prob.load_inputs('aviary/models/aircraft/UAV/small_scale_uav.csv', phase_info=cruise_phase_info)

#external subsystems
propulsion_builder = PropBuilder(name='rc_electric')
aero_builder = AeroBuilder(name='UAV_aero')
mass_builder = MassBuilder(name='UAV_mass')

#Setting the atmosphere model to Mars reference
prob.aviary_inputs.set_val(Settings.ATMOSPHERE_MODEL, AtmosphereModel.MARS_REFERENCE)

prob.load_external_subsystems(external_subsystems=[mass_builder, aero_builder, propulsion_builder])

#Setting aero conditions values
prob.aviary_inputs.set_val(Dynamic.Mission.ALTITUDE, 520.0, units='m')
prob.aviary_inputs.set_val(Dynamic.Mission.VELOCITY, 36.0, units='m/s')

#wing geomatry
prob.aviary_inputs.set_val(Aircraft.Wing.SPAN, 1.524, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.ROOT_CHORD, 0.508, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.10)
prob.aviary_inputs.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.266)
prob.aviary_inputs.set_val(Aircraft.Wing.TAPER_RATIO, 1.0, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Wing.SWEEP, 0.0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.INCIDENCE, 0.0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.CENTER_DISTANCE, 0.511, units='unitless')

#htail geometry
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SPAN, 0.711, units='m')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 0.232, units='m')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.14)
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 1.0, units='unitless')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0.0, units='deg')

#vtail geometry
prob.aviary_inputs.set_val(Aircraft.VerticalTail.SPAN, 0.3048, units='m')
prob.aviary_inputs.set_val(Aircraft.VerticalTail.ROOT_CHORD, 0.22225, units='m')
prob.aviary_inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.14)
prob.aviary_inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 1.0, units='unitless')

#fuselage geometry
prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 0.172, units='m')
prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 0.114, units='m')
prob.aviary_inputs.set_val(Aircraft.Fuselage.LENGTH, 1.190244, units='m')

#vehicle mass
prob.aviary_inputs.set_val(Dynamic.Vehicle.MASS, 3.787, units='kg')

#set up the model
prob.check_and_preprocess_inputs()
prob.build_model()
             
#set up optimization
# prob.add_driver('IPOPT', max_iter=15)
# prob.driver.opt_settings['tol'] = 1e-6

# prob.add_design_variables()
# prob.add_objective(objective_type = 'time')

prob.setup()
prob.set_initial_guesses()
prob.final_setup()

# # --- FIX INITIAL GUESSES FOR THE TRAJECTORY ---
# num_nodes = prob.model.get_io_metadata()['traj.phases.cruise.states:altitude']['size']

# prob.set_val('traj.phases.cruise.states:altitude', 520.0*np.ones(num_nodes), units='m')
# prob.set_val('traj.phases.cruise.states:velocity', 36.0*np.ones(num_nodes), units='m/s')

om.n2(
    prob,
    show_browser=True,
    title='UAV Mass-Aero-Propulsion Full Model',
)

#debugging before running the model
# if DEBUG_MODEL:
#     print('\nASSEMBLED UAV-RELATED SYSTEMS:\n')
#     assembled_systems = []

#     for system in prob.model.system_iter(recurse=True, include_self=False):
#         pathname = system.pathname
#         module_name = system.__class__.__module__
#         class_name = system.__class__.__name__
#         desc = f'{pathname} | {module_name}.{class_name}'
#         assembled_systems.append(desc.lower())

#         if any(
#             kw in desc.lower()
#             for kw in (
#                 'uav_mass',
#                 'uav_aero',
#                 'propulsion.uav',
#                 'rc_electric',
#                 'battery',
#                 'esc',
#                 'motor',
#                 'propeller',
#                 'gasp_based',
#                 'flops_based',
#                 'turbofan',
#                 'engine_deck',
#             )
#         ):
#             print(desc)

#     has_uav_mass = any('models.external_subsystems.uav.mass' in s
#                        for s in assembled_systems)
#     has_uav_aero = any('models.external_subsystems.uav.aerodynamics' in s
#                        for s in assembled_systems)
#     has_uav_prop = any('models.external_subsystems.uav.propulsion' in s
#                        for s in assembled_systems)

#     print('\nCUSTOM SUBSYSTEM CHECKS:\n')
#     print('Custom UAV mass present:', has_uav_mass)
#     print('Custom UAV aero present:', has_uav_aero)
#     print('Custom UAV propulsion present:', has_uav_prop)


#     assert has_uav_mass, (
#         'The custom UAV mass subsystem was not built.'
#     )

#     assert has_uav_aero, (
#         'The custom UAV aerodynamic subsystem was not built.'
#     )

#     assert has_uav_prop, (
#         'The custom UAV electric propulsion subsystem was not built.'
#     )
    
#     has_builtin_aero = any('solver_sub.aerodynamics' in s for s in assembled_systems)
#     print('Built-in solver_sub aerodynamics present:', has_builtin_aero)
#     # This should be False when cruise uses external aero only.
#     assert not has_builtin_aero, "Duplicate lift output from built-in aero."

# # Check that unwanted large-aircraft systems are absent

#     has_large_aircraft_mass = any(
#         (
#             'subsystems.mass.gasp_based' in system
#             or 'subsystems.mass.flops_based' in system
#         )
#         for system in assembled_systems
#     )
#     has_builtin_geometry = any(
#     (
#         'subsystems.geometry.gasp_based' in system
#         or 'subsystems.geometry.flops_based' in system
#     )
#     for system in assembled_systems
# )

#     has_conventional_engine = any(
#         (
#             'turbofan' in system
#             or 'engine_deck' in system
#         )
#         for system in assembled_systems
#     )

#     print('\nUNWANTED SUBSYSTEM CHECKS:\n')
# #these will print yes sometimes even if the subsystems are not present. and its because its present in inputs but they are set to 0 so they arent really used.
#     print(
#         'Built-in GASP/FLOPS mass present:',
#         has_large_aircraft_mass,
#     )
#     print(
#         'Built-in GASP/FLOPS geometry present:',
#         has_builtin_geometry,
#     )
#     print(
#         'Turbofan or EngineDeck present:',
#         has_conventional_engine,
#     )

#     # assert not has_large_aircraft_mass, (
#     #     'A built-in GASP/FLOPS large-aircraft mass '
#     #     'subsystem is present.'
#     # )

#     assert not has_conventional_engine, (
#         'A conventional turbofan or EngineDeck '
#         'propulsion model is present.'
#     )

# # NORMAL MODEL EXECUTION
# # This must remain outside the DEBUG_MODEL block.

prob.run_model()
# prob.run_aviary_problem(run_driver=True)

print('Structure mass:',prob.get_val(Aircraft.Design.STRUCTURE_MASS, units='kg'))
print('Cruise mass:',prob.get_val('traj.phases.cruise.indep_states.states:mass', units='kg'))
print('\nMODEL RUN COMPLETED.\n')

print("\nAVIARY ATMOSPHERE CHECK:\n")

prob.model.list_outputs(includes=['*OAS_aero.av_atmosphere*'], val=True, units=True, prom_name=True,)

# DEBUGGING AFTER THE MODEL RUNS

# if DEBUG_MODEL:
#     print('\nALPHA COMPONENT INPUTS:\n')

#     prob.model.list_inputs(
#         includes=['*alpha_comp*'],
#         val=True,
#         units=True,
#         prom_name=True,
#         print_arrays=True,
#     )
# # with open('all_UAV_Model_variables.txt', 'w') as f:
# #     prob.model.list_vars(units=True, prom_name=True,print_arrays=True, out_stream=f)

print('Lift:', prob.get_val('traj.cruise.rhs_all.lift', units='lbf')) 
print('Drag:', prob.get_val('traj.cruise.rhs_all.drag', units='lbf'))
print('CL:',prob.get_val('traj.cruise.rhs_all.lift_coefficient'),)
print('CD:', prob.get_val('traj.cruise.rhs_all.drag_coefficient'))
print('Lift balance residual:', prob.get_val( 'traj.phases.cruise.rhs_all.UAV_aero.lift_balance_residual', units='N'))
print('CD_fus:', prob.get_val('traj.cruise.rhs_all.CD_fus'))
print('CD_vtail:', prob.get_val('traj.cruise.rhs_all.CD_vtail'))
print('CD_gear:', prob.get_val('traj.cruise.rhs_all.CD_gear'))
print('Lifting surface CD:', prob.get_val('traj.cruise.rhs_all.lifting_surface_CD'))

print('Fuselage length:', prob.get_val('aircraft:fuselage:length'))
print('Fuselage height:', prob.get_val('aircraft:fuselage:max_height'))
print('Wing span:', prob.get_val(Aircraft.Wing.SPAN))

# these prints are to determine the mass of the UAV and its components to adjust the initial mission
#  gross mass since no optimization is being done 
print('Gross mass:',
      prob.get_val('mission:gross_mass', units='kg'))

print('Zero-fuel mass:',
      prob.get_val('mission:zero_fuel_mass', units='kg'))

print('Total fuel mass:',
      prob.get_val('mission:total_fuel_mass', units='kg'))

print('Engine mass:',
      prob.get_val('aircraft:engine:mass', units='kg'))

print('Total engine mass:',
      prob.get_val('aircraft:propulsion:total_engine_mass', units='kg'))