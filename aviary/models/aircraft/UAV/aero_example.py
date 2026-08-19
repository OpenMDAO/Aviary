'''

'''

import openmdao.api as om
import numpy as np
# np.seterr(divide='raise', invalid='raise')
import matplotlib.pyplot as plt

import aviary.api as av
from aviary.subsystems.aerodynamics.UAV.aero_builder import AeroBuilder
# from aviary.subsystems.aerodynamics.UAV.model import TotalAircraftAero #they are not used so commented out
# from aviary.utils.functions import set_aviary_initial_values
# from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import AtmosphereModel
from aviary.variable_info.variables import Settings

# Set True while debugging the integrated model.
# Set False for a normal cruise run.
DEBUG_MODEL = True
#import the UAV builders for mass and propulsion
from aviary.subsystems.mass.UAV.mass_builder import MassBuilder
from aviary.subsystems.propulsion.UAV.prop_builder import UAVBuilder

from aviary.variable_info.UAV_variables import Aircraft, Dynamic

aero_builder = AeroBuilder(name='UAV_aero')

mass_builder = MassBuilder()

phase_info = {
    'pre_mission': {
    'include_takeoff': False,
    'external_subsystems': [],
    'optimize_mass': False,

    'subsystem_options': {
        
        'aerodynamics': {
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
    },
}


# max_iter = 50
# optimizer = 'IPOPT' 

prob = av.AviaryProblem(verbosity=1)

prob.load_inputs('aviary/models/aircraft/UAV/small_scale_uav.csv', phase_info=phase_info)
propulsion_builder = UAVBuilder( options=prob.aviary_inputs, name='rc_electric',)

print('Aero builder:', aero_builder.name)
print('Propulsion builder:', propulsion_builder.name)
print('Mass builder:', mass_builder.name)

prob.load_external_subsystems([ mass_builder, aero_builder,  propulsion_builder,])

prob.aviary_inputs.set_val(
    Settings.ATMOSPHERE_MODEL,
    AtmosphereModel.MARS_REFERENCE,
)

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
# prob.add_driver('IPOPT',use_coloring=False, max_iter=max_iter)
prob.setup()
prob.set_initial_guesses()
prob.final_setup()

################
# Check what supplies mass to AlphaComp
mass_inputs = prob.model.list_inputs(
    includes=['*alpha_comp*mass*'],
    val=True,
    units=True,
    prom_name=True,
    out_stream=None,
)

for name, meta in mass_inputs:
    source = prob.model._conn_global_abs_in2out.get(name, 'UNCONNECTED')

    print('\nAlphaComp mass check:')
    print('  Input: ', name)
    print('  Source:', source)
    print('  Value: ', meta['val'], meta['units'])

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
#they will print yes sometimes even if the subsystems are not present. and its because its present in inputs but they are set to 0 so they arent really used.
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

    # assert not has_large_aircraft_mass, (
    #     'A built-in GASP/FLOPS large-aircraft mass '
    #     'subsystem is present.'
    # )

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
###################
print(
    'Structure mass:',
    prob.get_val(
        Aircraft.Design.STRUCTURE_MASS,
        units='kg',
    ),
)

print(
    'Cruise mass:',
    prob.get_val(
        'traj.phases.cruise.indep_states.states:mass',
        units='kg',
    ),
)
print('\nMODEL RUN COMPLETED.\n')

print("\nAVIARY ATMOSPHERE CHECK:\n")

prob.model.list_outputs(
    includes=['*OAS_aero.aviary_atmosphere*'],
    val=True,
    units=True,
    prom_name=True,
)
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
with open('all_UAV_Model_variables.txt', 'w') as f:
    prob.model.list_vars(units=True, prom_name=True,print_arrays=True, out_stream=f)
 


#Commented out get_val's are not recognized at the moment and I don't know why
print('Lift:', prob.get_val('traj.cruise.rhs_all.lift', units='lbf')) 
print('Drag:', prob.get_val('traj.cruise.rhs_all.drag', units='lbf'))
print('CL:',prob.get_val('traj.cruise.rhs_all.lift_coefficient'),)
print('CD:', prob.get_val('traj.cruise.rhs_all.drag_coefficient'))
print('Lift balance residual:', prob.get_val( 'traj.phases.cruise.rhs_all.UAV_aero.lift_balance_residual', units='N') #printing the residual to ensure that the lift is equal to the weight of the UAV
)
print('CD_fus:', prob.get_val('traj.cruise.rhs_all.CD_fus'))
print('CD_vtail:', prob.get_val('traj.cruise.rhs_all.CD_vtail'))
print('CD_gear:', prob.get_val('traj.cruise.rhs_all.CD_gear'))
print('Lifting surface CD:', prob.get_val('traj.cruise.rhs_all.lifting_surface_CD'))

print('Fuselage length:', prob.get_val('aircraft:fuselage:length'))
print('Fuselage height:', prob.get_val('aircraft:fuselage:max_height'))
print('Angle of attack:', prob.get_val('traj.cruise.rhs_all.UAV_aero.alpha'))
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