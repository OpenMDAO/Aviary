"""
This example file runs a cruise in the Mars Hellas Hot atmosphere and attempts to optimize it for time.
The optimization crashes because of an issue with the Aircraft.Wing.SPAN design variable.
However, if Aircraft.Wing.SPAN is REMOVED from get_design_vars in the UAV mass builder, the optimization
will actually run, and eventually fail.
"""

import aviary.api as av
from aviary.models.external_subsystems.UAV.aerodynamics.aero_builder import AeroBuilder
from aviary.models.external_subsystems.UAV.mass.mass_builder import MassBuilder
from aviary.models.external_subsystems.UAV.propulsion.prop_builder import PropBuilder
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variable_meta_data import (
    ExtendedMetaData as UAVExtendedMetaData,
)
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import (
    Aircraft,
    Dynamic,
    Settings,
)
from aviary.variable_info.enums import AtmosphereModel

cruise_phase_info = {
    'pre_mission': {
        'include_takeoff': False,
        'external_subsystems': [],
        'optimize_mass': False,
        'subsystem_options': {
            'atmosphere': {'method': 'external'},
            'aerodynamics': {'method': 'external'},
        },
    },
    'cruise': {
        'external_subsystems': ['alpha_comp'],
        'subsystem_options': {
            'atmosphere': {'method': 'external'},
            'aerodynamics': {'method': 'external'},
        },
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'mach_optimize': True,
            'mach_polynomial_order': 1,
            'mach_initial': (0.15, 'unitless'),
            'mach_final': (0.15, 'unitless'),
            'mach_bounds': ((0.01, 0.3), 'unitless'),
            'altitude_optimize': True,
            'altitude_polynomial_order': 1,
            'altitude_initial': (-7500, 'm'),
            'altitude_final': (-7500, 'm'),
            'altitude_bounds': ((-7750, -7250), 'm'),
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
        'subsystem_options': {
            'atmosphere': {'method': 'external'},
            'aerodynamics': {'method': 'external'},
        },
    },
}

# Create problem and load inputs
prob = av.AviaryProblem(verbosity=1, meta_data=UAVExtendedMetaData)
prob.load_inputs('aviary/models/aircraft/UAV/small_scale_uav.csv', phase_info=cruise_phase_info)

# load subsystems and atmosphere model
propulsion_builder = PropBuilder(name='rc_electric')
aero_builder = AeroBuilder(name='UAV_aero')
mass_builder = MassBuilder(name='UAV_mass')
prob.load_external_subsystems(external_subsystems=[mass_builder, aero_builder, propulsion_builder])
prob.aviary_inputs.set_val(Settings.ATMOSPHERE_MODEL, AtmosphereModel.MARS_HELLAS_HOT)

# set up the model and optimization
prob.check_and_preprocess_inputs()
prob.build_model()
prob.add_driver('IPOPT', max_iter=50)
prob.add_design_variables()
prob.add_objective(objective_type='time')
prob.setup()

# wing geometry (Aircraft.Wing.SPAN as a design variable breaks the optimization)
prob.aviary_inputs.set_val(Aircraft.Wing.SPAN, 2.0, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.10)
prob.aviary_inputs.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.266)
prob.aviary_inputs.set_val(Aircraft.Wing.TAPER_RATIO, 1.0, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Wing.SWEEP, 0.0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.INCIDENCE, 0.0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.CENTER_DISTANCE, 0.511, units='unitless')

# htail geometry
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.14)
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 1.0, units='unitless')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0.0, units='deg')

# vtail geometry
prob.aviary_inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.14)
prob.aviary_inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 1.0, units='unitless')

# fuselage geometry
prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 0.172, units='m')
prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 0.114, units='m')
prob.aviary_inputs.set_val(Aircraft.Fuselage.LENGTH, 1.190244, units='m')

# vehicle mass
prob.aviary_inputs.set_val(Dynamic.Vehicle.MASS, 3.833, units='kg')

prob.set_initial_guesses()
prob.final_setup()

# Setting values for slack variables/constraints
prob.set_val('traj.cruise.controls:rpm_slack', 60.0, units='rev/s')
prob.set_val('traj.phases.cruise.controls:alpha', 9, units='deg')

prob.run_aviary_problem(run_driver=True)

print('Structure mass:', prob.get_val(Aircraft.Design.STRUCTURE_MASS, units='kg'))
print('Cruise mass:', prob.get_val('traj.phases.cruise.indep_states.states:mass', units='kg'))
print('\nMODEL RUN COMPLETED.\n')

# printed rounded first values in arrays of nodes,
print('Velocity: ', round(prob.get_val('traj.cruise.rhs_all.velocity')[0], 3), 'ft/s')
print('Lift:', round(prob.get_val('traj.cruise.rhs_all.lift', units='lbf')[0], 3), 'N')
print('Drag:', round(prob.get_val('traj.cruise.rhs_all.drag', units='lbf')[0], 3), 'N')
print('CL:', round(prob.get_val('traj.cruise.rhs_all.lift_coefficient')[0], 3))
print('CD:', round(prob.get_val('traj.cruise.rhs_all.drag_coefficient')[0], 3))
print(
    'Lift balance residual:',
    round(prob.get_val('traj.phases.cruise.rhs_all.lift_balance_residual', units='N')[0], 3),
)
print('Fuselage CD:', round(prob.get_val('traj.cruise.rhs_all.CD_fus')[0], 3))
print('Vertical Tail CD:', round(prob.get_val('traj.cruise.rhs_all.CD_vtail')[0], 3))
print('Lifting surface CD:', round(prob.get_val('traj.cruise.rhs_all.lifting_surface_CD')[0], 3))
print('Gross mass:', round(prob.get_val('mission:gross_mass', units='kg')[0], 3))
print('Zero-fuel mass:', round(prob.get_val('mission:zero_fuel_mass', units='kg')[0], 3))
print(
    'Fuel mass (should be zero):', round(prob.get_val('mission:total_fuel_mass', units='kg')[0], 3)
)
