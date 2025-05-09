import dymos as dm
import numpy as np
import openmdao.api as om
from dymos.transcriptions.transcription_base import TranscriptionBase
from dymos.utils.misc import _unspecified

from aviary.mission.flops_based.phases.simplified_landing import LandingGroup
from aviary.variable_info.variables import Aircraft, Mission
from subsystems.mass.flops_based import engine

if hasattr(TranscriptionBase, 'setup_polynomial_controls'):
    use_new_dymos_syntax = False
else:
    use_new_dymos_syntax = True

### This is the full list of functions I need to analyze from level 2
# prob.add_post_mission_systems()
# prob.link_phases()
# prob.add_driver('SNOPT', max_iter=50, verbosity=1)

# ##########################
# # Design Variables       #
# ##########################
# prob.add_design_variables()

# # Nudge it a bit off the correct answer to verify that the optimize takes us there.
# prob.aviary_inputs.set_val(Mission.Design.GROSS_MASS, 135000.0, units='lbm')

# ##########################
# # Add Objective Function #
# ##########################
# prob.add_objective()

# ############################################
# # Initial Settings for States and Controls #
# ############################################
# prob.setup()
# prob.set_initial_guesses()
# prob.run_aviary_problem('dymos_solution.db')


##### The following code pertains to add_post_mission_systems code
### Note: a lot of the code is dependent on the problem model having a
### PostMissionGroup defined internally

# `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
# override for just this method
# verbosity = self.verbosity  ### Not sure why this is needed at all?

prob = om.prob()
post_mission = om.group()

prob.add_subsystem(
    'post_mission',
    post_mission,
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)

### Following code pertains to add_post_mission_systems
# prob.builder.add_post_mission_systems(prob, include_landing)
### include_takeoff is true in N3CC case
# if prob.pre_mission_info['include_takeoff']:
# self._add_post_mission_takeoff_systems(prob)
# else:
#     first_flight_phase_name = list(prob.phase_info.keys())[0]
#     first_flight_phase = prob.traj._phases[first_flight_phase_name]
#     first_flight_phase.set_state_options(Dynamic.Vehicle.MASS, fix_initial=False)

### From the code snippet above, the _add_post_mission_takeoff_systems
### needs to be implemented here
first_flight_phase_name = list(prob.phase_info.keys())[0]
# connect_takeoff_to_climb = not prob.phase_info[first_flight_phase_name]['user_options'].get(
#     'add_initial_mass_constraint', True
# )

### N3CC connects takeoff to climb
# if connect_takeoff_to_climb:
prob.model.connect(
    Mission.Takeoff.FINAL_MASS, f'traj.{first_flight_phase_name}.initial_states:mass'
)
prob.model.connect(
    Mission.Takeoff.GROUND_DISTANCE,
    f'traj.{first_flight_phase_name}.initial_states:distance',
)

control_type_string = 'control_values'

### use polynomial control is not defined, so defaults to true here
# if prob.phase_info[first_flight_phase_name]['user_options'].get(
#     'use_polynomial_control', True
# ):
if not use_new_dymos_syntax:
    control_type_string = 'polynomial_control_values'

### N3CC is optimizing mach
# if prob.phase_info[first_flight_phase_name]['user_options'].get('optimize_mach', False):
# Create an ExecComp to compute the difference in mach
mach_diff_comp = om.ExecComp('mach_resid_for_connecting_takeoff = final_mach - initial_mach')
prob.model.add_subsystem('mach_diff_comp', mach_diff_comp)

# Connect the inputs to the mach difference component
prob.model.connect(Mission.Takeoff.FINAL_MACH, 'mach_diff_comp.final_mach')
prob.model.connect(
    f'traj.{first_flight_phase_name}.{control_type_string}:mach',
    'mach_diff_comp.initial_mach',
    src_indices=[0],
)

# Add constraint for mach difference
prob.model.add_constraint('mach_diff_comp.mach_resid_for_connecting_takeoff', equals=0.0)

### Optimize altitude is true
# if prob.phase_info[first_flight_phase_name]['user_options'].get(
#     'optimize_altitude', False
# ):
# Similar steps for altitude difference
alt_diff_comp = om.ExecComp(
    'altitude_resid_for_connecting_takeoff = final_altitude - initial_altitude',
    units='ft',
)
prob.model.add_subsystem('alt_diff_comp', alt_diff_comp)

prob.model.connect(Mission.Takeoff.FINAL_ALTITUDE, 'alt_diff_comp.final_altitude')
prob.model.connect(
    f'traj.{first_flight_phase_name}.{control_type_string}:altitude',
    'alt_diff_comp.initial_altitude',
    src_indices=[0],
)

prob.model.add_constraint('alt_diff_comp.altitude_resid_for_connecting_takeoff', equals=0.0)

### _add_landing_systems needs to be included here
### Both the include_landing and 'post_mission_info is true here
# if include_landing and prob.post_mission_info['include_landing']:
# self._add_landing_systems(prob)
# landing_options = Landing(
#     ref_wing_area=prob.aviary_inputs.get_val(Aircraft.Wing.AREA, units='ft**2'),
#     Cl_max_ldg=prob.aviary_inputs.get_val(Mission.Landing.LIFT_COEFFICIENT_MAX),  # no units
# )

### This is the equivalent of landing = landing_options.build_phase(False)

landing = LandingGroup()
landing.set_input_defaults(Aircraft.Wing.AREA, 1220.0, units='ft**2')
landing.set_input_defaults(Mission.Landing.LIFT_COEFFICIENT_MAX, 2.0, units='unitless')

prob.model.add_subsystem(
    'landing',
    landing,
    promotes_inputs=['aircraft:*', 'mission:*'],
    promotes_outputs=['mission:*'],
)

last_flight_phase_name = list(prob.phase_info.keys())[-1]

control_type_string = 'control_values'
if prob.phase_info[last_flight_phase_name]['user_options'].get('use_polynomial_control', True):
    if not use_new_dymos_syntax:
        control_type_string = 'polynomial_control_values'

last_regular_phase = prob.regular_phases[-1]
prob.model.connect(
    f'traj.{last_regular_phase}.states:mass',
    Mission.Landing.TOUCHDOWN_MASS,
    src_indices=[-1],
)
prob.model.connect(
    f'traj.{last_regular_phase}.{control_type_string}:altitude',
    Mission.Landing.INITIAL_ALTITUDE,
    src_indices=[0],
)
### End of _add_landing_systems

# connect summary mass to the initial guess of mass in the first phase

### This should result in false...
# if not prob.pre_mission_info['include_takeoff']:
#     first_flight_phase_name = list(prob.phase_info.keys())[0]

#     eq = prob.model.add_subsystem(
#         f'link_{first_flight_phase_name}_mass',
#         om.EQConstraintComp(),
#         promotes_inputs=[('rhs:mass', Mission.Summary.GROSS_MASS)],
#     )

#     eq.add_eq_output(
#         'mass', eq_units='lbm', normalize=False, ref=100000.0, add_constraint=True
#     )

#     prob.model.connect(
#         f'traj.{first_flight_phase_name}.states:mass',
#         f'link_{first_flight_phase_name}_mass.lhs:mass',
#         src_indices=[0],
#         flat_src_indices=True,
#     )

prob.model.add_subsystem(
    'range_constraint',
    om.ExecComp(
        'range_resid = target_range - actual_range',
        target_range={'val': prob.target_range, 'units': 'NM'},
        actual_range={'val': prob.target_range, 'units': 'NM'},
        range_resid={'val': 30, 'units': 'NM'},
    ),
    promotes_inputs=[
        ('actual_range', Mission.Summary.RANGE),
        'target_range',
    ],
    promotes_outputs=[('range_resid', Mission.Constraints.RANGE_RESIDUAL)],
)

prob.post_mission.add_constraint(Mission.Constraints.MASS_RESIDUAL, equals=0.0, ref=1.0e5)

### NO SUBSYSTEMS IN THE PHASE INFO
# Add all post-mission external subsystems.
# for external_subsystem in self.post_mission_info['external_subsystems']:
#     subsystem_postmission = external_subsystem.build_post_mission(self.aviary_inputs)

#     if subsystem_postmission is not None:
#         self.post_mission.add_subsystem(external_subsystem.name, subsystem_postmission)
###

# Check if regular_phases[] is accessible
try:
    prob.regular_phases[0]
except BaseException:
    raise ValueError(
        'regular_phases[] dictionary is not accessible. For HEIGHT_ENERGY and '
        'SOLVED_2DOF missions, check_and_preprocess_inputs() must be called '
        'before add_post_mission_systems().'
    )

# Fuel burn in regular phases
ecomp = om.ExecComp(
    'fuel_burned = initial_mass - mass_final',
    initial_mass={'units': 'lbm'},
    mass_final={'units': 'lbm'},
    fuel_burned={'units': 'lbm'},
)

post_mission.add_subsystem(
    'fuel_burned',
    ecomp,
    promotes=[('fuel_burned', Mission.Summary.FUEL_BURNED)],
)

### N3CC uses collocation I think. And pre_mission includes_takeoff
# if prob.analysis_scheme is AnalysisScheme.SHOOTING:
#     # shooting method currently doesn't have timeseries
#     self.post_mission.promotes(
#         'fuel_burned',
#         [
#             ('initial_mass', Mission.Summary.GROSS_MASS),
#             ('mass_final', Mission.Landing.TOUCHDOWN_MASS),
#         ],
#     )
# else:
#     if self.pre_mission_info['include_takeoff']:
prob.post_mission.promotes(
    'fuel_burned',
    [('initial_mass', Mission.Summary.GROSS_MASS)],
)
# else:
#     # timeseries has to be used because Breguet cruise phases don't have
#     # states
#     self.model.connect(
#         f'traj.{self.regular_phases[0]}.timeseries.mass',
#         'fuel_burned.initial_mass',
#         src_indices=[0],
#     )

prob.model.connect(
    f'traj.{prob.regular_phases[-1]}.timeseries.mass',
    'fuel_burned.mass_final',
    src_indices=[-1],
)

### N3CC example does not have reserve phases
# Fuel burn in reserve phases
# if self.reserve_phases:
#     ecomp = om.ExecComp(
#         'reserve_fuel_burned = initial_mass - mass_final',
#         initial_mass={'units': 'lbm'},
#         mass_final={'units': 'lbm'},
#         reserve_fuel_burned={'units': 'lbm'},
#     )

#     self.post_mission.add_subsystem(
#         'reserve_fuel_burned',
#         ecomp,
#         promotes=[('reserve_fuel_burned', Mission.Summary.RESERVE_FUEL_BURNED)],
#     )

#     if self.analysis_scheme is AnalysisScheme.SHOOTING:
#         # shooting method currently doesn't have timeseries
#         self.post_mission.promotes(
#             'reserve_fuel_burned',
#             [('initial_mass', Mission.Landing.TOUCHDOWN_MASS)],
#         )
#         self.model.connect(
#             f'traj.{self.reserve_phases[-1]}.states:mass',
#             'reserve_fuel_burned.mass_final',
#             src_indices=[-1],
#         )
#     else:
#         # timeseries has to be used because Breguet cruise phases don't have
#         # states
#         self.model.connect(
#             f'traj.{self.reserve_phases[0]}.timeseries.mass',
#             'reserve_fuel_burned.initial_mass',
#             src_indices=[0],
#         )
#         self.model.connect(
#             f'traj.{self.reserve_phases[-1]}.timeseries.mass',
#             'reserve_fuel_burned.mass_final',
#             src_indices=[-1],
#         )

### The following all comes from the _add_fuel_reserve_component()
### self._add_fuel_reserve_component()
### post_mission Set to true by default
# if post_mission:
# reserve_calc_location = prob.post_mission
# else:
#     reserve_calc_location = self.model

### Not in N3CC data, so should be 0
# RESERVE_FUEL_FRACTION = prob.aviary_inputs.get_val(
#     Aircraft.Design.RESERVE_FUEL_FRACTION, units='unitless'
# )
# if RESERVE_FUEL_FRACTION != 0:
#     reserve_fuel_frac = om.ExecComp(
#         'reserve_fuel_frac_mass = reserve_fuel_fraction * (takeoff_mass - final_mass)',
#         reserve_fuel_frac_mass={'units': 'lbm'},
#         reserve_fuel_fraction={
#             'units': 'unitless',
#             'val': RESERVE_FUEL_FRACTION,
#         },
#         final_mass={'units': 'lbm'},
#         takeoff_mass={'units': 'lbm'},
#     )

#     post_mission.add_subsystem(
#         'reserve_fuel_frac',
#         reserve_fuel_frac,
#         promotes_inputs=[
#             ('takeoff_mass', Mission.Summary.GROSS_MASS),
#             ('final_mass', Mission.Landing.TOUCHDOWN_MASS),
#             ('reserve_fuel_fraction', Aircraft.Design.RESERVE_FUEL_FRACTION),
#         ],
#         promotes_outputs=['reserve_fuel_frac_mass'],
#     )

### also equal to 0
# RESERVE_FUEL_ADDITIONAL = prob.aviary_inputs.get_val(
#     Aircraft.Design.RESERVE_FUEL_ADDITIONAL, units='lbm'
# )
reserve_fuel = om.ExecComp(
    'reserve_fuel = reserve_fuel_frac_mass + reserve_fuel_additional + reserve_fuel_burned',
    reserve_fuel={'units': 'lbm', 'shape': 1},
    reserve_fuel_frac_mass={'units': 'lbm', 'val': 0},
    reserve_fuel_additional={'units': 'lbm', 'val': RESERVE_FUEL_ADDITIONAL},
    reserve_fuel_burned={'units': 'lbm', 'val': 0},
)

post_mission.add_subsystem(
    'reserve_fuel',
    reserve_fuel,
    promotes_inputs=[
        'reserve_fuel_frac_mass',
        ('reserve_fuel_additional', Aircraft.Design.RESERVE_FUEL_ADDITIONAL),
        ('reserve_fuel_burned', Mission.Summary.RESERVE_FUEL_BURNED),
    ],
    promotes_outputs=[('reserve_fuel', reserves_name)],
)

# TODO: need to add some sort of check that this value is less than the fuel capacity
# TODO: the overall_fuel variable is the burned fuel plus the reserve, but should
# also include the unused fuel, and the hierarchy variable name should be
# more clear
ecomp = om.ExecComp(
    'overall_fuel = (1 + fuel_margin/100)*fuel_burned + reserve_fuel',
    overall_fuel={'units': 'lbm', 'shape': 1},
    fuel_margin={'units': 'unitless', 'val': 0},
    fuel_burned={'units': 'lbm'},  # from regular_phases only
    reserve_fuel={'units': 'lbm', 'shape': 1},
)
post_mission.add_subsystem(
    'fuel_calc',
    ecomp,
    promotes_inputs=[
        ('fuel_margin', Aircraft.Fuel.FUEL_MARGIN),
        ('fuel_burned', Mission.Summary.FUEL_BURNED),
        ('reserve_fuel', Mission.Design.RESERVE_FUEL),
    ],
    promotes_outputs=[('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS)],
)

# If a target distance (or time) has been specified for this phase
# distance (or time) is measured from the start of this phase to the end
# of this phase
# for phase_name in self.phase_info:
### No target distance specified ###
# if 'target_distance' in self.phase_info[phase_name]['user_options']:
#     target_distance = wrapped_convert_units(
#         self.phase_info[phase_name]['user_options']['target_distance'],
#         'nmi',
#     )
#     self.post_mission.add_subsystem(
#         f'{phase_name}_distance_constraint',
#         om.ExecComp(
#             'distance_resid = target_distance - (final_distance - initial_distance)',
#             distance_resid={'units': 'nmi'},
#             target_distance={'val': target_distance, 'units': 'nmi'},
#             final_distance={'units': 'nmi'},
#             initial_distance={'units': 'nmi'},
#         ),
#     )
#     self.model.connect(
#         f'traj.{phase_name}.timeseries.distance',
#         f'{phase_name}_distance_constraint.final_distance',
#         src_indices=[-1],
#     )
#     self.model.connect(
#         f'traj.{phase_name}.timeseries.distance',
#         f'{phase_name}_distance_constraint.initial_distance',
#         src_indices=[0],
#     )
#     self.model.add_constraint(
#         f'{phase_name}_distance_constraint.distance_resid',
#         equals=0.0,
#         ref=1e2,
#     )

# this is only used for analytic phases with a target duration
### No target duration ###
# if 'target_duration' in self.phase_info[phase_name]['user_options'] and self.phase_info[
#     phase_name]['user_options'].get('analytic', False):
#     target_duration = wrapped_convert_units(
#         self.phase_info[phase_name]['user_options']['target_duration'],
#         'min',
#     )
#     self.post_mission.add_subsystem(
#         f'{phase_name}_duration_constraint',
#         om.ExecComp(
#             'duration_resid = target_duration - (final_time - initial_time)',
#             duration_resid={'units': 'min'},
#             target_duration={'val': target_duration, 'units': 'min'},
#             final_time={'units': 'min'},
#             initial_time={'units': 'min'},
#         ),
#     )
#     self.model.connect(
#         f'traj.{phase_name}.timeseries.time',
#         f'{phase_name}_duration_constraint.final_time',
#         src_indices=[-1],
#     )
#     self.model.connect(
#         f'traj.{phase_name}.timeseries.time',
#         f'{phase_name}_duration_constraint.initial_time',
#         src_indices=[0],
#     )
#     self.model.add_constraint(
#         f'{phase_name}_duration_constraint.duration_resid',
#         equals=0.0,
#         ref=1e2,
#     )

ecomp = om.ExecComp(
    'mass_resid = operating_empty_mass + overall_fuel + payload_mass - initial_mass',
    operating_empty_mass={'units': 'lbm'},
    overall_fuel={'units': 'lbm'},
    payload_mass={'units': 'lbm'},
    initial_mass={'units': 'lbm'},
    mass_resid={'units': 'lbm'},
)

post_mission.add_subsystem(
    'mass_constraint',
    ecomp,
    promotes_inputs=[
        ('operating_empty_mass', Aircraft.Design.OPERATING_MASS),
        ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
        ('payload_mass', Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS),
        ('initial_mass', Mission.Summary.GROSS_MASS),
    ],
    promotes_outputs=[('mass_resid', Mission.Constraints.MASS_RESIDUAL)],
)

##### The following code pertains to link_phases code

### Dont think we need to specify verbosity here
# `self.verbosity` is "true" verbosity for entire run. `verbosity` is verbosity
# override for just this method
# if verbosity is not None:
# compatibility with being passed int for verbosity
#    verbosity = Verbosity(verbosity)
# else:
#    verbosity = self.verbosity  # defaults to BRIEF

### Adding the code for the _add_bus_variables_and_connect function
# self._add_bus_variables_and_connect()
### Adding the code for the _get_all_subsystems function
# all_subsystems = self._get_all_subsystems()
all_subsystems = []
### external_subsystems defaults to None. Also don't believe there are any
### external subsystems in the phase info for N3CC, so commenting out
# if external_subsystems is None:
# all_subsystems.extend(self.pre_mission_info['external_subsystems'])
# else:
#    all_subsystems.extend(external_subsystems)

all_subsystems.append(prob.core_subsystems['aerodynamics'])
all_subsystems.append(prob.core_subsystems['propulsion'])

### Spelling this out
# base_phases = list(self.phase_info.keys())
base_phases = ['pre_mission', 'climb', 'cruise', 'descent', 'post_mission']

### Dont have any external subsystems though right? Or is this just iterating through whatever is there?
### Think this might be the CoreAerodynamicsBuilder and CorePropulsionBuilder
### Aerodynamics does nothing since this isn't a gasp model
### PropulsionBuilder has some output from get_bus_variables()

### Only external subsystem with bus variables is the engine I believe
# for external_subsystem in all_subsystems:
### Think this line needs to just be replaced by an engine call? Engine will need to be defined above
# bus_variables = external_subsystem.get_bus_variables()
bus_variables = engine.get_bus_variables()
### bus_variables isn't empty I think
# if bus_variables is not None:
for bus_variable, variable_data in bus_variables.items():
    mission_variable_name = variable_data['mission_name']

    # check if mission_variable_name is a list
    if not isinstance(mission_variable_name, list):
        mission_variable_name = [mission_variable_name]

    # loop over the mission_variable_name list and add each variable to
    # the trajectory
    for mission_var_name in mission_variable_name:
        if mission_var_name not in prob.meta_data:
            # base_units = self.model.get_io_metadata(includes=f'pre_mission.{external_subsystem.name}.{bus_variable}')[f'pre_mission.{external_subsystem.name}.{bus_variable}']['units']
            base_units = variable_data['units']

            shape = variable_data.get('shape', _unspecified)

            targets = mission_var_name
            if '.' in mission_var_name:
                # Support for non-hierarchy variables as parameters.
                mission_var_name = mission_var_name.split('.')[-1]

            if 'phases' in variable_data:
                # Support for connecting bus variables into a subset of
                # phases.
                for phase_name in variable_data['phases']:
                    phase = getattr(prob.traj.phases, phase_name)

                    phase.add_parameter(
                        mission_var_name,
                        opt=False,
                        static_target=True,
                        units=base_units,
                        shape=shape,
                        targets=targets,
                    )

                    prob.model.connect(
                        f'pre_mission.{bus_variable}',
                        f'traj.{phase_name}.parameters:{mission_var_name}',
                    )

            else:
                prob.traj.add_parameter(
                    mission_var_name,
                    opt=False,
                    static_target=True,
                    units=base_units,
                    shape=shape,
                    targets={phase_name: [mission_var_name] for phase_name in base_phases},
                )

                prob.model.connect(
                    f'pre_mission.{bus_variable}',
                    'traj.parameters:' + mission_var_name,
                )

        if 'post_mission_name' in variable_data:
            prob.model.connect(
                f'pre_mission.{external_subsystem.name}.{bus_variable}',
                f'post_mission.{external_subsystem.name}.{variable_data["post_mission_name"]}',
            )

phases = list(self.phase_info.keys())

### N3CC has more phases
# if len(phases) <= 1:
#     return

# In summary, the following code loops over all phases in self.phase_info, gets
# the linked variables from each external subsystem in each phase, and stores
# the lists of linked variables in lists_to_link. It then gets a list of
# unique variable names from lists_to_link and loops over them, creating
# a list of phase names for each variable and linking the phases
# using self.traj.link_phases().

### Dont think there are any external subsystems in the N3CC phase info
### This section can possibly just be commented out
# lists_to_link = []
# for idx, phase_name in enumerate(self.phase_info):
#     lists_to_link.append([])
#     for external_subsystem in self.phase_info[phase_name]['external_subsystems']:
#         lists_to_link[idx].extend(external_subsystem.get_linked_variables())

# get unique variable names from lists_to_link
# unique_vars = list(set([var for sublist in lists_to_link for var in sublist]))

# Phase linking.
# If we are under mpi, and traj.phases is running in parallel, then let the
# optimizer handle the linkage constraints.  Note that we can technically
# parallelize connected phases, but it requires a solver that we would like
# to avoid.
true_unless_mpi = True
if self.comm.size > 1 and self.traj.options['parallel_phases']:
    true_unless_mpi = False

# loop over unique variable names
for var in unique_vars:
    phases_to_link = []
    for idx, phase_name in enumerate(self.phase_info):
        if var in lists_to_link[idx]:
            phases_to_link.append(phase_name)

    if len(phases_to_link) > 1:  # TODO: hack
        self.traj.link_phases(phases=phases_to_link, vars=[var], connected=True)

        ### Adding the height_energy_problem_configurator code here
        # self.builder.link_phases(self, phases, connect_directly=true_unless_mpi)

        # connect regular_phases with each other if you are optimizing alt or mach
        prob._link_phases_helper_with_options(
            prob.regular_phases,
            'optimize_altitude',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        prob._link_phases_helper_with_options(
            prob.regular_phases, 'optimize_mach', Dynamic.Atmosphere.MACH
        )

        # connect reserve phases with each other if you are optimizing alt or mach
        prob._link_phases_helper_with_options(
            prob.reserve_phases,
            'optimize_altitude',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        prob._link_phases_helper_with_options(
            prob.reserve_phases, 'optimize_mach', Dynamic.Atmosphere.MACH
        )

        # connect mass and distance between all phases regardless of reserve /
        # non-reserve status
        prob.traj.link_phases(
            phases, ['time'], ref=None if connect_directly else 1e3, connected=connect_directly
        )
        prob.traj.link_phases(
            phases,
            [Dynamic.Vehicle.MASS],
            ref=None if connect_directly else 1e6,
            connected=connect_directly,
        )
        prob.traj.link_phases(
            phases,
            [Dynamic.Mission.DISTANCE],
            ref=None if connect_directly else 1e3,
            connected=connect_directly,
        )

        prob.model.connect(
            f'traj.{prob.regular_phases[-1]}.timeseries.distance',
            Mission.Summary.RANGE,
            src_indices=[-1],
            flat_src_indices=True,
        )
