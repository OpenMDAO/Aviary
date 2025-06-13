import aviary.api as av
import openmdao.api as om
from aviary.interface.default_phase_info.height_energy import phase_info
from aviary.variable_info.variables import Aircraft, Mission, Dynamic, Settings
from aviary.mission.flops_based.phases.energy_phase import EnergyPhase
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.utils.aviary_values import AviaryValues
import dymos as dm
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import setup_trajectory_params


class L3SubsystemsGroup(om.Group):
    """Group that contains all pre-mission groups of core Aviary subsystems (geometry, mass, propulsion, aerodynamics)."""

    def initialize(self):
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.manual_overrides = []


prob = av.AviaryProblem()
# prob = om.Problem() maybe we need to set up L3 as a

# prob.load_inputs(csv_path, phase_info)
csv_path = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'

aviary_inputs, _ = av.create_vehicle('models/test_aircraft/aircraft_for_bench_FwFm.csv')

# let's change this to be a 'proper example' N3CC

engine = av.build_engine_deck(aviary_inputs)

prob.phase_info = {}
for phase_name in phase_info:
    if phase_name not in ['pre_mission', 'post_mission']:
        prob.phase_info[phase_name] = phase_info[phase_name]
aviary_inputs.set_val(Mission.Summary.RANGE, 1906.0, units='NM')
prob.require_range_residual = True
prob.target_range = 1906.0

# prob.pre_mission_info = phase_info['pre_mission']
# prob.post_mission_info = phase_info['post_mission']

# prob.check_and_preprocess_inputs()
av.preprocess_options(aviary_inputs, engine_models=[engine])

# prob.add_pre_mission_systems()
aerodynamics = av.CoreAerodynamicsBuilder(code_origin=av.LegacyCode('FLOPS'))
geometry = av.CoreGeometryBuilder(code_origin=av.LegacyCode('FLOPS'))
mass = av.CoreMassBuilder(code_origin=av.LegacyCode('FLOPS'))
propulsion = av.CorePropulsionBuilder(engine_models=engine)

prob.core_subsystems = {
    'propulsion': propulsion,
    'geometry': geometry,
    'mass': mass,
    'aerodynamics': aerodynamics,
}
prob.meta_data = BaseMetaData.copy()

# prob.add_pre_mission_systems()
# overwrites calculated values in pre-mission with override values from .csv
prob.model.add_subsystem(
    'pre_mission',
    av.PreMissionGroup(),
    promotes_inputs=['aircraft:*', 'mission:*'],
    promotes_outputs=['aircraft:*', 'mission:*'],
)

# This is a combination of prob.add_pre_mission_systems and prob.setup()
# In the aviary code add_pre_mission_systems only instantiates the objects and methods, the build method is called in prob.setup()
prob.model.pre_mission.add_subsystem(
    'core_propulsion',
    propulsion.build_pre_mission(aviary_inputs),
)

# add another group subsystem to match the L2 example
prob.model.pre_mission.add_subsystem(
    'core_subsystems',
    L3SubsystemsGroup(aviary_options=aviary_inputs),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)
prob.model.pre_mission.core_subsystems.add_subsystem(
    'core_geometry',
    geometry.build_pre_mission(aviary_inputs),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)
prob.model.pre_mission.core_subsystems.add_subsystem(
    'core_aerodynamics',
    aerodynamics.build_pre_mission(aviary_inputs),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)
prob.model.pre_mission.core_subsystems.add_subsystem(
    'core_mass',
    mass.build_pre_mission(aviary_inputs),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)

##########
# prob.add_phases()
phases = ['climb', 'cruise', 'descent']
prob.traj = prob.model.add_subsystem('traj', dm.Trajectory())
for phase_idx, phase_name in enumerate(phases):
    base_phase_options = prob.phase_info[phase_name]
    phase_options = {}
    for key, val in base_phase_options.items():
        phase_options[key] = val
    phase_options['user_options'] = {}
    for key, val in base_phase_options['user_options'].items():
        phase_options['user_options'][key] = val
    default_mission_subsystems = [
        prob.core_subsystems['aerodynamics'],
        prob.core_subsystems['propulsion'],
    ]
    phase_builder = EnergyPhase
    phase_object = phase_builder.from_phase_info(
        phase_name, phase_options, default_mission_subsystems, meta_data=prob.meta_data
    )
    phase = phase_object.build_phase(aviary_options=aviary_inputs)

    user_options = AviaryValues(phase_options.get('user_options', ()))

    phase = prob.traj.add_phase(phase_name, phase)

externs = {'climb': {}, 'cruise': {}, 'descent': {}}
for default_subsys in default_mission_subsystems:
    params = default_subsys.get_parameters(aviary_inputs=aviary_inputs, phase_info={})
    for key, val in params.items():
        for phname in externs:
            externs[phname][key] = val

prob.traj = setup_trajectory_params(
    prob.model, prob.traj, aviary_inputs, external_parameters=externs
)

# need aviary inputs assigned to the problem object for other functions below
# this maybe needs a better location in this script.
prob.aviary_inputs = aviary_inputs

##########
#  prob.add_post_mission_systems()
prob.model.add_subsystem(
    'post_mission',
    om.Group(),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)

prob.traj._phases['climb'].set_state_options(Dynamic.Vehicle.MASS, fix_initial=False)

eq = prob.model.add_subsystem(
    f'link_climb_mass',
    om.EQConstraintComp(),
    promotes_inputs=[('rhs:mass', Mission.Summary.GROSS_MASS)],
)

eq.add_eq_output('mass', eq_units='lbm', normalize=False, ref=100000.0, add_constraint=True)

prob.model.connect(
    f'traj.climb.states:mass',
    f'link_climb_mass.lhs:mass',
    src_indices=[0],
    flat_src_indices=True,
)

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

prob.model.add_constraint(Mission.Constraints.MASS_RESIDUAL, equals=0.0, ref=1.0e5)
# for reference this is the end of builder.add_post_mission_systems()

ecomp = om.ExecComp(
    'fuel_burned = initial_mass - mass_final',
    initial_mass={'units': 'lbm'},
    mass_final={'units': 'lbm'},
    fuel_burned={'units': 'lbm'},
)

prob.model.post_mission.add_subsystem(
    'fuel_burned',
    ecomp,
    promotes=[('fuel_burned', Mission.Summary.FUEL_BURNED)],
)

prob.model.connect(
    f'traj.climb.timeseries.mass',
    'fuel_burned.initial_mass',
    src_indices=[0],
)

prob.model.connect(
    f'traj.descent.timeseries.mass',
    'fuel_burned.mass_final',
    src_indices=[-1],
)

# maybe should split this out and define explicitly
prob._add_fuel_reserve_component()

ecomp = om.ExecComp(
    'overall_fuel = (1 + fuel_margin/100)*fuel_burned + reserve_fuel',
    overall_fuel={'units': 'lbm', 'shape': 1},
    fuel_margin={'units': 'unitless', 'val': 0},
    fuel_burned={'units': 'lbm'},  # from regular_phases only
    reserve_fuel={'units': 'lbm', 'shape': 1},
)
prob.model.post_mission.add_subsystem(
    'fuel_calc',
    ecomp,
    promotes_inputs=[
        ('fuel_margin', Aircraft.Fuel.FUEL_MARGIN),
        ('fuel_burned', Mission.Summary.FUEL_BURNED),
        ('reserve_fuel', Mission.Design.RESERVE_FUEL),
    ],
    promotes_outputs=[('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS)],
)

# If target distances have been set per phase then there is a block of code to add here.
# In this case individual phases don't have target distances.

ecomp = om.ExecComp(
    'mass_resid = operating_empty_mass + overall_fuel + payload_mass - initial_mass',
    operating_empty_mass={'units': 'lbm'},
    overall_fuel={'units': 'lbm'},
    payload_mass={'units': 'lbm'},
    initial_mass={'units': 'lbm'},
    mass_resid={'units': 'lbm'},
)

# this seems clunky - we could just move this directly into the promotes inputs block?
payload_mass_src = Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS

prob.model.post_mission.add_subsystem(
    'mass_constraint',
    ecomp,
    promotes_inputs=[
        ('operating_empty_mass', Aircraft.Design.OPERATING_MASS),
        ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
        ('payload_mass', payload_mass_src),
        ('initial_mass', Mission.Summary.GROSS_MASS),
    ],
    promotes_outputs=[('mass_resid', Mission.Constraints.MASS_RESIDUAL)],
)

##########
# prob.link_phases()
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
base_phases = ['climb', 'cruise', 'descent']

### Dont have any external subsystems though right? Or is this just iterating through whatever is there?
### Think this might be the CoreAerodynamicsBuilder and CorePropulsionBuilder
### Aerodynamics does nothing since this isn't a gasp model
### PropulsionBuilder has some output from get_bus_variables() - not for FwFm

### Only external subsystem with bus variables is the engine I believe - I don't see any bus variables when running L2!

for external_subsystem in all_subsystems:  # I think this is badly named - this is not just 'external subsystems' so I ignored this entire block of code at first!
    ### Think this line needs to just be replaced by an engine call? Engine will need to be defined above
    bus_variables = external_subsystem.get_bus_variables()
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

phases = list(prob.phase_info.keys())
prob.traj.link_phases(phases, ['time'], ref=None, connected=True)
prob.traj.link_phases(phases, [Dynamic.Vehicle.MASS], ref=None, connected=True)
prob.traj.link_phases(phases, [Dynamic.Mission.DISTANCE], ref=None, connected=True)

prob.model.connect(
    f'traj.descent.timeseries.distance',
    Mission.Summary.RANGE,
    src_indices=[-1],
    flat_src_indices=True,
)

##########
# prob.add_driver('IPOPT', max_iter=50)
prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'IPOPT'
prob.driver.declare_coloring(show_summary=False)
prob.driver.opt_settings['print_user_options'] = 'no'
prob.driver.opt_settings['print_frequency_iter'] = 10
prob.driver.opt_settings['print_level'] = 3
prob.driver.opt_settings['tol'] = 1.0e-6
prob.driver.opt_settings['mu_init'] = 1e-5
prob.driver.opt_settings['max_iter'] = 50
prob.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
prob.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
prob.driver.opt_settings['mu_strategy'] = 'monotone'
prob.driver.options['print_results'] = 'minimal'

##########
# prob.add_design_variables()
prob.model.add_design_var(
    Mission.Design.GROSS_MASS,
    lower=10.0,
    upper=None,
    units='lbm',
    ref=175e3,
)
prob.model.add_design_var(
    Mission.Summary.GROSS_MASS,
    lower=10.0,
    upper=None,
    units='lbm',
    ref=175e3,
)

prob.model.add_subsystem(
    'gtow_constraint',
    om.EQConstraintComp(
        'GTOW',
        eq_units='lbm',
        normalize=True,
        add_constraint=True,
    ),
    promotes_inputs=[
        ('lhs:GTOW', Mission.Design.GROSS_MASS),
        ('rhs:GTOW', Mission.Summary.GROSS_MASS),
    ],
)
prob.model.add_constraint(Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10)

##########
# prob.add_objective()
prob.model.add_subsystem(
    'fuel_obj',
    om.ExecComp(
        'reg_objective = overall_fuel/10000 + ascent_duration/30.',
        reg_objective={'val': 0.0, 'units': 'unitless'},
        ascent_duration={'units': 's', 'shape': 1},
        overall_fuel={'units': 'lbm'},
    ),
    promotes_inputs=[
        ('ascent_duration', Mission.Takeoff.ASCENT_DURATION),
        ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
    ],
    promotes_outputs=[('reg_objective', Mission.Objectives.FUEL)],
)
prob.model.add_objective(Mission.Objectives.FUEL, ref=1)

##########
# What does this actually do?
# is it worth splitting this out in more detail?
prob.setup()

##########
# prob.set_initial_guesses()
control_keys = ['mach', 'altitude']
state_keys = ['mass', Dynamic.Mission.DISTANCE]
guesses = {}
guesses['mach_climb'] = ([0.2, 0.72], 'unitless')
guesses['altitude_climb'] = ([0, 32000.0], 'ft')
guesses['time_climb'] = ([0, 3840.0], 's')
guesses['mach_cruise'] = ([0.72, 0.72], 'unitless')
guesses['altitude_cruise'] = ([32000.0, 34000.0], 'ft')
guesses['time_cruise'] = ([3840.0, 3390.0], 's')
guesses['mach_descent'] = ([0.72, 0.36], 'unitless')
guesses['altitude_descent'] = ([34000.0, 500.0], 'ft')
guesses['time_descent'] = ([7230.0, 1740.0], 's')

prob.set_val('traj.climb.t_initial', guesses['time_climb'][0][0], units='s')
prob.set_val('traj.climb.t_duration', guesses['time_climb'][0][1], units='s')
prob.set_val(
    'traj.climb.controls:mach',
    prob.model.traj.phases.climb.interp('mach', xs=[-1, 1], ys=guesses['mach_climb'][0]),
    units='unitless',
)
prob.set_val(
    'traj.climb.controls:altitude',
    prob.model.traj.phases.climb.interp('altitude', xs=[-1, 1], ys=guesses['altitude_climb'][0]),
    units='ft',
)

prob.set_val('traj.cruise.t_initial', guesses['time_cruise'][0][0], units='s')
prob.set_val('traj.cruise.t_duration', guesses['time_cruise'][0][1], units='s')
prob.set_val(
    'traj.cruise.controls:mach',
    prob.model.traj.phases.cruise.interp('mach', xs=[-1, 1], ys=guesses['mach_cruise'][0]),
    units='unitless',
)
prob.set_val(
    'traj.cruise.controls:altitude',
    prob.model.traj.phases.cruise.interp('altitude', xs=[-1, 1], ys=guesses['altitude_cruise'][0]),
    units='ft',
)

prob.set_val('traj.descent.t_initial', guesses['time_descent'][0][0], units='s')
prob.set_val('traj.descent.t_duration', guesses['time_descent'][0][1], units='s')
prob.set_val(
    'traj.descent.controls:mach',
    prob.model.traj.phases.climb.interp('mach', xs=[-1, 1], ys=guesses['mach_descent'][0]),
    units='unitless',
)
prob.set_val(
    'traj.descent.controls:altitude',
    prob.model.traj.phases.climb.interp('altitude', xs=[-1, 1], ys=guesses['altitude_descent'][0]),
    units='ft',
)
prob.set_val('traj.climb.states:mass', 125000, units='lbm')
prob.set_val('traj.cruise.states:mass', 125000, units='lbm')
prob.set_val('traj.descent.states:mass', 125000, units='lbm')

prob.verbosity = Verbosity.VERBOSE

"""
try:
    prob.run_model()
except:
    pass
"""

# prob.model.list_vars(units=True, print_arrays=True)

prob.run_aviary_problem()

# prob.model.list_vars(out_stream='FwFm_L2L3_listvars_postrun.txt')

prob.list_driver_vars()
