import warnings

import dymos as dm
import openmdao.api as om

import aviary.api as av
from aviary.core.pre_mission_group import PreMissionGroup
from aviary.mission.flops_based.phases.energy_phase import EnergyPhase
from aviary.models.missions.height_energy_default import phase_info
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import setup_model_options, setup_trajectory_params
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class L3SubsystemsGroup(om.Group):
    """Group that contains all pre-mission groups of core Aviary subsystems (geometry, mass, propulsion, aerodynamics)."""

    def initialize(self):
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.code_origin_overrides = []


prob = av.AviaryProblem()

#####
# prob.load_inputs(csv_path, phase_info)
csv_path = 'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv'

aviary_inputs, _ = av.create_vehicle(csv_path)

engine = av.build_engine_deck(aviary_inputs)

prob.model.phase_info = {}
for phase_name in phase_info:
    if phase_name not in ['pre_mission', 'post_mission']:
        prob.model.phase_info[phase_name] = phase_info[phase_name]
aviary_inputs.set_val(Mission.Summary.RANGE, 1906.0, units='NM')
prob.require_range_residual = True
prob.target_range = 1906.0

#####
# prob.check_and_preprocess_inputs()
av.preprocess_options(aviary_inputs, engine_models=[engine])

#####
# prob.add_pre_mission_systems()
aerodynamics = av.CoreAerodynamicsBuilder(code_origin=av.LegacyCode('FLOPS'))
geometry = av.CoreGeometryBuilder(code_origin=av.LegacyCode('FLOPS'))
mass = av.CoreMassBuilder(code_origin=av.LegacyCode('FLOPS'))
propulsion = av.CorePropulsionBuilder(engine_models=engine)

prob.model.core_subsystems = {
    'propulsion': propulsion,
    'geometry': geometry,
    'mass': mass,
    'aerodynamics': aerodynamics,
}
prob.meta_data = BaseMetaData.copy()

#####
# prob.add_pre_mission_systems()
# overwrites calculated values in pre-mission with override values from .csv
prob.model.add_subsystem(
    'pre_mission',
    PreMissionGroup(),
    promotes_inputs=['aircraft:*', 'mission:*'],
    promotes_outputs=['aircraft:*', 'mission:*'],
)

#####
# This is a combination of prob.add_pre_mission_systems and prob.setup()
# In the aviary code add_pre_mission_systems only instantiates the objects and methods, the build method is called in prob.setup()
prob.model.pre_mission.add_subsystem(
    'core_propulsion',
    propulsion.build_pre_mission(aviary_inputs),
)

# adding another group subsystem to match the L2 example
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

#####
# prob.add_phases()
phases = ['climb', 'cruise', 'descent']
prob.traj = prob.model.add_subsystem('traj', dm.Trajectory())
default_mission_subsystems = [
    prob.model.core_subsystems['aerodynamics'],
    prob.model.core_subsystems['propulsion'],
]
for phase_idx, phase_name in enumerate(phases):
    base_phase_options = prob.model.phase_info[phase_name]
    phase_options = {}
    for key, val in base_phase_options.items():
        phase_options[key] = val
    phase_options['user_options'] = {}
    for key, val in base_phase_options['user_options'].items():
        phase_options['user_options'][key] = val
    phase_builder = EnergyPhase
    phase_object = phase_builder.from_phase_info(
        phase_name, phase_options, default_mission_subsystems, meta_data=prob.meta_data
    )
    phase = phase_object.build_phase(aviary_options=aviary_inputs)
    prob.traj.add_phase(phase_name, phase)

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

#####
#  prob.add_post_mission_systems()
prob.model.add_subsystem(
    'post_mission',
    om.Group(),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)

prob.traj._phases['climb'].set_state_options(
    Dynamic.Vehicle.MASS, fix_initial=False, input_initial=False
)

prob.traj._phases['climb'].set_state_options(
    Dynamic.Mission.DISTANCE, fix_initial=True, input_initial=False
)

prob.traj._phases['climb'].set_time_options(
    fix_initial=False,
    initial_bounds=(0, 0),
    initial_ref=600,
    duration_bounds=(3840, 11520),
    duration_ref=7680.0,
)

prob.traj._phases['cruise'].set_time_options(
    duration_bounds=(3390, 10170),
    duration_ref=6780.0,
)

prob.traj._phases['descent'].set_time_options(
    duration_bounds=(1740, 5220),
    duration_ref=3480.0,
)

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

RESERVE_FUEL_ADDITIONAL = prob.aviary_inputs.get_val(
    Aircraft.Design.RESERVE_FUEL_ADDITIONAL, units='lbm'
)

reserve_fuel = om.ExecComp(
    'reserve_fuel = reserve_fuel_frac_mass + reserve_fuel_additional + reserve_fuel_burned',
    reserve_fuel={'units': 'lbm', 'shape': 1},
    reserve_fuel_frac_mass={'units': 'lbm', 'val': 0},
    reserve_fuel_additional={'units': 'lbm', 'val': RESERVE_FUEL_ADDITIONAL},
    reserve_fuel_burned={'units': 'lbm', 'val': 0},
)
prob.model.post_mission.add_subsystem(
    'reserve_fuel',
    reserve_fuel,
    promotes_inputs=[
        'reserve_fuel_frac_mass',
        ('reserve_fuel_additional', Aircraft.Design.RESERVE_FUEL_ADDITIONAL),
        ('reserve_fuel_burned', Mission.Summary.RESERVE_FUEL_BURNED),
    ],
    promotes_outputs=[('reserve_fuel', Mission.Design.RESERVE_FUEL)],
)

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

ecomp = om.ExecComp(
    'excess_fuel_capacity = total_fuel_capacity - unusable_fuel - overall_fuel',
    total_fuel_capacity={'units': 'lbm'},
    unusable_fuel={'units': 'lbm'},
    overall_fuel={'units': 'lbm'},
    excess_fuel_capacity={'units': 'lbm'},
)

prob.model.post_mission.add_subsystem(
    'excess_fuel_constraint',
    ecomp,
    promotes_inputs=[
        ('total_fuel_capacity', Aircraft.Fuel.TOTAL_CAPACITY),
        ('unusable_fuel', Aircraft.Fuel.UNUSABLE_FUEL_MASS),
        ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
    ],
    promotes_outputs=[('excess_fuel_capacity', Mission.Constraints.EXCESS_FUEL_CAPACITY)],
)

prob.model.add_constraint(Mission.Constraints.EXCESS_FUEL_CAPACITY, lower=0, units='lbm')

#####
# prob.link_phases()

all_subsystems = []
all_subsystems.append(prob.model.core_subsystems['propulsion'])

phases = list(prob.model.phase_info.keys())
prob.traj.link_phases(phases, ['time'], ref=None, connected=True)
prob.traj.link_phases(phases, [Dynamic.Vehicle.MASS], ref=None, connected=True)
prob.traj.link_phases(phases, [Dynamic.Mission.DISTANCE], ref=None, connected=True)

prob.model.connect(
    f'traj.descent.timeseries.distance',
    Mission.Summary.RANGE,
    src_indices=[-1],
    flat_src_indices=True,
)
#### End of link_phases

#####
# prob.add_driver('SLSQP', max_iter=50)
# SLSQP Optimizer Settings
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.declare_coloring(show_summary=False)
prob.driver.options['disp'] = True
prob.driver.options['tol'] = 1e-9
prob.driver.options['maxiter'] = 50

# IPOPT Optimizer Settings
# prob.driver.opt_settings['print_user_options'] = 'no'
# prob.driver.opt_settings['print_frequency_iter'] = 10
# prob.driver.opt_settings['print_level'] = 3
# prob.driver.opt_settings['tol'] = 1.0e-6
# prob.driver.opt_settings['mu_init'] = 1e-5
# prob.driver.opt_settings['max_iter'] = 50
# prob.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
# prob.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
# prob.driver.opt_settings['mu_strategy'] = 'monotone'
# prob.driver.options['print_results'] = 'minimal'
# prob.driver.opt_settings['iSumm'] = 6
# prob.driver.opt_settings['iPrint'] = 0

# SNOPT Optimizer Settings #
# prob.driver.opt_settings['Major iterations limit'] = 50
# prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
# prob.driver.opt_settings['Major feasibility tolerance'] = 1e-7
# prob.driver.opt_settings['iSumm'] = 6
# prob.driver.opt_settings['iPrint'] = 0

#####
# prob.add_design_variables()
prob.model.add_design_var(
    Mission.Design.GROSS_MASS,
    lower=100000.0,
    upper=None,
    units='lbm',
    ref=175e3,
)
prob.model.add_design_var(
    Mission.Summary.GROSS_MASS,
    lower=100000.0,
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

#####
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

prob.model.add_subsystem(
    'range_obj',
    om.ExecComp(
        'reg_objective = -actual_range/1000 + ascent_duration/30.',
        reg_objective={'val': 0.0, 'units': 'unitless'},
        ascent_duration={'units': 's', 'shape': 1},
        actual_range={'val': prob.target_range, 'units': 'NM'},
    ),
    promotes_inputs=[
        ('actual_range', Mission.Summary.RANGE),
        ('ascent_duration', Mission.Takeoff.ASCENT_DURATION),
    ],
    promotes_outputs=[('reg_objective', Mission.Objectives.RANGE)],
)

#####
# prob.setup()
setup_model_options(prob, prob.aviary_inputs, prob.meta_data)

with warnings.catch_warnings():
    prob.model.aviary_inputs = prob.aviary_inputs
    prob.model.meta_data = prob.meta_data

with warnings.catch_warnings():
    warnings.simplefilter('ignore', om.OpenMDAOWarning)
    warnings.simplefilter('ignore', om.PromotionWarning)

    om.Problem.setup(prob, check=False)

# set initial guesses manually
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

prob.set_val(Mission.Design.GROSS_MASS, 175400, units='lbm')
prob.set_val(Mission.Summary.GROSS_MASS, 175400, units='lbm')

prob.verbosity = Verbosity.BRIEF

prob.run_aviary_problem()

# prob.model.list_vars(units=True, print_arrays=True)
# prob.list_driver_vars(print_arrays=True)
