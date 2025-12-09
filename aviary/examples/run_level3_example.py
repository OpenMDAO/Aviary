import warnings

import dymos as dm
import openmdao.api as om
import numpy as np

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


# Toggle this boolean option to run with shooting vs collocation transcription:
shooting = True

prob = av.AviaryProblem()

#####
# prob.load_inputs(csv_path, phase_info)
csv_path = 'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv'

aviary_inputs, _ = av.create_vehicle(csv_path)

engine = av.build_engine_deck(aviary_inputs)

prob.model.mission_info = {}
for phase_name in phase_info:
    if phase_name not in ['pre_mission', 'post_mission']:
        prob.model.mission_info[phase_name] = phase_info[phase_name]
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
phase_list = ['climb', 'cruise', 'descent']
prob.traj = prob.model.add_subsystem('traj', dm.Trajectory())
default_mission_subsystems = [
    prob.model.core_subsystems['aerodynamics'],
    prob.model.core_subsystems['propulsion'],
]
phases = {}
for phase_idx, phase_name in enumerate(phase_list):
    base_phase_options = prob.model.mission_info[phase_name]
    phase_options = {}
    for key, val in base_phase_options.items():
        phase_options[key] = val
    phase_options['user_options'] = {}
    for key, val in base_phase_options['user_options'].items():
        phase_options['user_options'][key] = val

    # now need to create the dymos phase object and add it to the trajectory
    # phase_builder = EnergyPhase
    from aviary.mission.flops_based.ode.energy_ODE import EnergyODE

    default_ode_class = EnergyODE
    # Unpack the phase info
    # phase_object = phase_builder.from_phase_info(
    #     phase_name, phase_options, default_mission_subsystems, meta_data=prob.meta_data
    # )
    # loop over user_options dict entries
    # if the value is not a tuple, wrap it in a tuple with the second entry of 'unitless'
    for key, value in phase_options['user_options'].items():
        if not isinstance(value, tuple):
            phase_options['user_options'][key] = (value, 'unitless')
    subsystem_options = phase_options.get('subsystem_options', {})
    user_options = phase_options.get('user_options', ())
    initial_guesses = AviaryValues(
        phase_options.get('initial_guesses', ())
    )  # None of these not necessary for this example
    external_subsystems = phase_options.get(
        'external_subsystems', []
    )  # None of these not necessary for this example

    # instantiate the PhaseBuilderBaseClass:
    # phase_builder = cls(
    #     name,
    #     subsystem_options=subsystem_options,
    #     user_options=user_options,
    #     initial_guesses=initial_guesses,
    #     meta_data=meta_data,
    #     core_subsystems=core_subsystems,
    #     external_subsystems=external_subsystems,
    #     transcription=transcription,
    # )
    # this basically just adds these objects to the class - we have them all available in the L3 script so have no need for the extra class!
    # phase_name
    # subsystem_options
    # user_options
    # initial_guesses
    # prob.meta_data
    # default_mission_subsystems
    # external_subsystems
    transcription = None
    ode_class = None
    is_analytic_phase = False
    num_nodes = 5

    # Now build the phase using the instantiated phase object:
    # phase = phase_object.build_phase(aviary_options=aviary_inputs)
    # phase: dm.Phase = super().build_phase(aviary_options)

    # if ode_class is None:
    ode_class = default_ode_class
    # if transcription is None and not is_analytic_phase:
    # transcription = self.make_default_transcription()
    from aviary.mission.flight_phase_builder import FlightPhaseOptions

    user_options = FlightPhaseOptions(user_options)
    num_segments = user_options['num_segments']
    order = user_options['order']

    if shooting:
        transcription = dm.PicardShooting(num_segments=3, solve_segments='forward')
    else:
        transcription = dm.Radau(num_segments=num_segments, order=order, compressed=True)

    kwargs = {
        'external_subsystems': external_subsystems,
        'meta_data': prob.meta_data,
        'subsystem_options': subsystem_options,
        'throttle_enforcement': user_options['throttle_enforcement'],
        'throttle_allocation': user_options['throttle_allocation'],
        'core_subsystems': default_mission_subsystems,
        'external_subsystems': external_subsystems,
    }
    kwargs = {'aviary_options': aviary_inputs, **kwargs}
    phase = dm.Phase(ode_class=ode_class, transcription=transcription, ode_init_kwargs=kwargs)
    tx_mission_bus = dm.GaussLobatto(
        num_segments=transcription.options['num_segments'], order=3, compressed=True
    )
    phase.add_timeseries(name='mission_bus_variables', transcription=tx_mission_bus, subset='all')
    num_engine_type = len(aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES))
    throttle_enforcement = user_options['throttle_enforcement']
    no_descent = user_options['no_descent']
    no_climb = user_options['no_climb']
    constraints = user_options['constraints']
    ground_roll = user_options['ground_roll']

    def add_l3_state(phase, options, name, target, rate_source):
        initial, _ = options[f'{name}_initial']
        final, _ = options[f'{name}_final']
        bounds, units = options[f'{name}_bounds']
        ref, _ = options[f'{name}_ref']
        ref0, _ = options[f'{name}_ref0']
        defect_ref, _ = options[f'{name}_defect_ref']
        solve_segments = options[f'{name}_solve_segments']
        phase.add_state(
            target,
            fix_initial=initial is not None,
            input_initial=False,
            lower=bounds[0],
            upper=bounds[1],
            units=units,
            rate_source=rate_source,
            ref=ref,
            ref0=ref0,
            defect_ref=defect_ref,
            solve_segments='forward' if solve_segments else None,
        )

        if final is not None:
            constraint_ref, _ = options[f'{name}_constraint_ref']
            if constraint_ref is None:
                # If unspecified, final is a good value for it.
                constraint_ref = final
            phase.add_boundary_constraint(
                target,
                loc='final',
                equals=final,
                units=units,
                ref=final,
            )

    add_l3_state(
        phase,
        user_options,
        'mass',
        Dynamic.Vehicle.MASS,
        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
    )
    add_l3_state(
        phase, user_options, 'distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE
    )

    def add_l3_control(
        phase, options, name, target, rate_targets=None, rate2_targets=None, add_constraints=True
    ):
        """
        Add a control to this phase using the options in the phase-info - this is similar to the class method
        """
        initial, _ = options[f'{name}_initial']
        final, _ = options[f'{name}_final']
        bounds, units = options[f'{name}_bounds']
        ref, _ = options[f'{name}_ref']
        ref0, _ = options[f'{name}_ref0']
        polynomial_order = options[f'{name}_polynomial_order']
        opt = options[f'{name}_optimize']

        if ref == 1.0:
            # This has not been moved from default, so find a good value.
            candidates = [x for x in (bounds[0], bounds[1], initial, final) if x is not None]
            if len(candidates) > 0:
                ref = np.max(np.abs(np.array(candidates)))

        extra_options = {}
        if polynomial_order is not None:
            extra_options['control_type'] = 'polynomial'
            extra_options['order'] = polynomial_order

        if opt is True:
            extra_options['lower'] = bounds[0]
            extra_options['upper'] = bounds[1]
            extra_options['ref'] = ref
            extra_options['ref0'] = ref0

        if units not in ['unitless', None]:
            extra_options['units'] = units

        if rate_targets is not None:
            extra_options['rate_targets'] = rate_targets

        if rate2_targets is not None:
            extra_options['rate2_targets'] = rate2_targets

        phase.add_control(target, targets=target, opt=opt, **extra_options)

        # Add timeseries for any control.
        phase.add_timeseries_output(target)

        if not add_constraints:
            return

        # Add an initial constraint.
        if opt and initial is not None:
            phase.add_boundary_constraint(
                target, loc='initial', equals=initial, units=units, ref=ref
            )

        # Add a final constraint.
        if opt and final is not None:
            phase.add_boundary_constraint(target, loc='final', equals=final, units=units, ref=ref)

    add_l3_control(
        phase,
        user_options,
        'mach',
        target=Dynamic.Atmosphere.MACH,
        rate_targets=[Dynamic.Atmosphere.MACH_RATE],
        rate2_targets=None,
        add_constraints=Dynamic.Atmosphere.MACH not in constraints,
    )

    add_l3_control(
        phase,
        user_options,
        'altitude',
        target=Dynamic.Mission.ALTITUDE,
        rate_targets=[Dynamic.Mission.ALTITUDE_RATE],
        rate2_targets=None,
        add_constraints=Dynamic.Mission.ALTITUDE not in constraints,
    )
    if throttle_enforcement == 'control':
        add_l3_control(
            phase,
            user_options,
            'throttle',
            Dynamic.Vehicle.Propulsion.THROTTLE,
            rate_targets=None,
            rate2_targets=None,
            add_constraints=True,
        )

    phase.add_timeseries_output(
        Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
        output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
        units='lbf',
    )

    phase.add_timeseries_output(Dynamic.Vehicle.DRAG, output_name=Dynamic.Vehicle.DRAG, units='lbf')

    phase.add_timeseries_output(
        Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
        output_name=Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
        units='m/s',
    )
    phase.add_timeseries_output(
        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        output_name=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        units='lbm/h',
    )

    phase.add_timeseries_output(
        Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
        output_name=Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
        units='kW',
    )

    phase.add_timeseries_output(
        Dynamic.Mission.ALTITUDE_RATE,
        output_name=Dynamic.Mission.ALTITUDE_RATE,
        units='ft/s',
    )

    if throttle_enforcement != 'control':
        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            output_name=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
        )

    phase.add_timeseries_output(
        Dynamic.Mission.VELOCITY,
        output_name=Dynamic.Mission.VELOCITY,
        units='m/s',
    )

    phase.add_path_constraint(
        Dynamic.Vehicle.Propulsion.THROTTLE,
        lower=0.0,
        upper=1.0,
        units='unitless',
    )
    prob.traj.add_phase(phase_name, phase)
    phases[phase_name] = phase

climb = phases['climb']
cruise = phases['cruise']
descent = phases['descent']

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

climb.set_state_options(Dynamic.Vehicle.MASS, fix_initial=False, input_initial=False)

climb.set_state_options(Dynamic.Mission.DISTANCE, fix_initial=True, input_initial=False)

climb.set_time_options(
    fix_initial=False,
    initial_bounds=(0, 0),
    initial_ref=600,
    duration_bounds=(3840, 11520),
    duration_ref=7680.0,
)

cruise.set_time_options(
    duration_bounds=(3390, 10170),
    duration_ref=6780.0,
)

descent.set_time_options(
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
        ('operating_empty_mass', Mission.Summary.OPERATING_MASS),
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

prob.traj.link_phases(phase_list, ['time'], ref=None, connected=True)
prob.traj.link_phases(phase_list, [Dynamic.Vehicle.MASS], ref=None, connected=True)
prob.traj.link_phases(phase_list, [Dynamic.Mission.DISTANCE], ref=None, connected=True)
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

# prob.add_driver('IPOPT', max_iter=50)
# IPOPT Optimizer Settings
# prob.driver = om.pyOptSparseDriver()
# prob.driver.options['optimizer'] = 'IPOPT'
# prob.driver.declare_coloring(show_summary=False)
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

# prob.add_driver('SNOPT', max_iter=50)
# SNOPT Optimizer Settings #
# prob.driver = om.pyOptSparseDriver()
# prob.driver.options['optimizer'] = 'SNOPT'
# prob.driver.declare_coloring(show_summary=False)
# prob.driver.opt_settings['iSumm'] = 6
# prob.driver.opt_settings['iPrint'] = 0
# prob.driver.opt_settings['Major iterations limit'] = 50
# prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
# prob.driver.opt_settings['Major feasibility tolerance'] = 1e-7
# prob.driver.options['print_results'] = 'minimal'

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

climb.set_time_val(
    initial=guesses['time_climb'][0][0], duration=guesses['time_climb'][0][1], units='s'
)
climb.set_control_val('mach', vals=guesses['mach_climb'][0], time_vals=[-1, 1], units='unitless')
climb.set_control_val('altitude', vals=guesses['altitude_climb'][0], time_vals=[-1, 1], units='ft')
climb.set_state_val('mass', 125000, units='lbm')

cruise.set_time_val(
    initial=guesses['time_cruise'][0][0], duration=guesses['time_cruise'][0][1], units='s'
)
cruise.set_control_val('mach', vals=guesses['mach_cruise'][0], time_vals=[-1, 1], units='unitless')
cruise.set_control_val(
    'altitude', vals=guesses['altitude_cruise'][0], time_vals=[-1, 1], units='ft'
)
cruise.set_state_val('mass', 125000, units='lbm')

descent.set_time_val(
    initial=guesses['time_descent'][0][0], duration=guesses['time_descent'][0][1], units='s'
)
descent.set_control_val(
    'mach', vals=guesses['mach_descent'][0], time_vals=[-1, 1], units='unitless'
)
descent.set_control_val(
    'altitude', vals=guesses['altitude_descent'][0], time_vals=[-1, 1], units='ft'
)
descent.set_state_val('mass', 125000, units='lbm')

prob.set_val(Mission.Design.GROSS_MASS, 175400, units='lbm')
prob.set_val(Mission.Summary.GROSS_MASS, 175400, units='lbm')

prob.verbosity = Verbosity.BRIEF

prob.run_aviary_problem()

# Uncomment these lines to get printouts of every variable in the openmdao model
# prob.model.list_vars(units=True, print_arrays=True)
# prob.list_driver_vars(print_arrays=True)
