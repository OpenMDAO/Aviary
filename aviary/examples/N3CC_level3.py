"""
Sizing the N3CC using the level 3 API.

Includes:
  Takeoff, Climb, Cruise, Descent, Landing
  Computed Aero
  N3CC data
"""

import scipy.constants as _units

import dymos as dm
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.testing_utils import require_pyoptsparse

from aviary.mission.flops_based.phases.energy_phase import EnergyPhase
from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_aviary_input_defaults
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_crewpayload, preprocess_propulsion
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.functions import setup_trajectory_params, setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData


try:
    import pyoptsparse
except ImportError:
    pyoptsparse = None


from dymos.transcriptions.transcription_base import TranscriptionBase

FLOPS = LegacyCode.FLOPS

# benchmark for simple sizing problem on the N3CC


def run_trajectory(sim=True):
    # Level 2 Equivalent: Defining an AviaryProblem() from the interface methods
    prob = om.Problem(model=om.Group())

    # Interface methods for level 2 processes drivers through the prob.add_driver
    # function (specifying driver type, iterations, and verbosity)
    if pyoptsparse:
        driver = prob.driver = om.pyOptSparseDriver()
        driver.options['optimizer'] = 'SNOPT'
        # driver.declare_coloring()  # currently disabled pending resolve of issue 2507
        if driver.options['optimizer'] == 'SNOPT':
            driver.opt_settings['Major iterations limit'] = 45
            driver.opt_settings['Major optimality tolerance'] = 1e-4
            driver.opt_settings['Major feasibility tolerance'] = 1e-6
            driver.opt_settings['iSumm'] = 6
        elif driver.options['optimizer'] == 'IPOPT':
            driver.opt_settings['max_iter'] = 100
            driver.opt_settings['tol'] = 1e-3
            driver.opt_settings['print_level'] = 4

    else:
        driver = prob.driver = om.ScipyOptimizeDriver()
        opt_settings = prob.driver.opt_settings

        driver.options['optimizer'] = 'SLSQP'
        opt_settings['maxiter'] = 100
        opt_settings['ftol'] = 5.0e-3
        opt_settings['eps'] = 1e-2

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    # Level 2 Equivalent: loads inputs through the prob.load_inputs function,
    # with arguments for the input csv and phase info python file.

    aviary_inputs = get_flops_inputs('N3CC')

    aviary_inputs.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX, 2.4, units='unitless')
    aviary_inputs.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX, 2.0, units='unitless')
    aviary_inputs.set_val(
        Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, val=0.0175, units='unitless'
    )

    # Fuel is not included in the level 2 functionality, and these values must remain

    takeoff_fuel_burned = 577  # lbm TODO: where should this get connected from?
    takeoff_thrust_per_eng = 24555.5  # lbf TODO: where should this get connected from?
    takeoff_L_over_D = 17.35  # TODO: should this come from aero?

    aviary_inputs.set_val(Mission.Takeoff.FUEL_SIMPLE, takeoff_fuel_burned, units='lbm')
    aviary_inputs.set_val(Mission.Takeoff.LIFT_OVER_DRAG, takeoff_L_over_D, units='unitless')
    aviary_inputs.set_val(
        Mission.Design.THRUST_TAKEOFF_PER_ENG, takeoff_thrust_per_eng, units='lbf'
    )

    # Level 2 Equivalent: Constants are provided in a phase_info object, specifying
    # parameters for the climb/cruise/descent phase. Phase_info is read through the
    # load_inputs level 2 method
    alt_airport = 0  # ft

    alt_i_climb = 0 * _units.foot  # m
    alt_f_climb = 35000.0 * _units.foot  # m
    mass_i_climb = 131000 * _units.lb  # kg
    mass_f_climb = 126000 * _units.lb  # kg
    # initial mach set to lower value so it can intersect with takeoff end mach
    # mach_i_climb = 0.3
    mach_i_climb = 0.2
    mach_f_climb = 0.79
    distance_i_climb = 0 * _units.nautical_mile  # m
    distance_f_climb = 160.3 * _units.nautical_mile  # m
    t_i_climb = 2 * _units.minute  # sec
    t_f_climb = 26.20 * _units.minute  # sec
    t_duration_climb = t_f_climb - t_i_climb

    alt_i_cruise = 35000 * _units.foot  # m
    alt_f_cruise = 35000 * _units.foot  # m
    alt_min_cruise = 35000 * _units.foot  # m
    alt_max_cruise = 35000 * _units.foot  # m
    mass_i_cruise = 126000 * _units.lb  # kg
    mass_f_cruise = 102000 * _units.lb  # kg
    cruise_mach = 0.79
    distance_i_cruise = 160.3 * _units.nautical_mile  # m
    distance_f_cruise = 3243.9 * _units.nautical_mile  # m
    t_i_cruise = 26.20 * _units.minute  # sec
    t_f_cruise = 432.38 * _units.minute  # sec
    t_duration_cruise = t_f_cruise - t_i_cruise

    alt_i_descent = 35000 * _units.foot
    # final altitude set to 35 to ensure landing is feasible point
    # alt_f_descent = 0*_units.foot
    alt_f_descent = 35 * _units.foot
    mach_i_descent = 0.79
    mach_f_descent = 0.3
    mass_i_descent = 102000 * _units.pound
    mass_f_descent = 101000 * _units.pound
    distance_i_descent = 3243.9 * _units.nautical_mile
    distance_f_descent = 3378.7 * _units.nautical_mile
    t_i_descent = 432.38 * _units.minute
    t_f_descent = 461.62 * _units.minute
    t_duration_descent = t_f_descent - t_i_descent

    ##########################
    # Design Variables       #
    ##########################

    # Before design variables are added in level 2 formats, the program runs through
    # the following commands
    # prob.check_and_preprocess_inputs()
    # prob.add_pre_mission_systems()
    # prob.add_phases()
    # prob.add_post_mission_systems()
    # prob.link_phases()

    # Nudge it a bit off the correct answer to verify that the optimize takes us there.
    aviary_inputs.set_val(Mission.Design.GROSS_MASS, 135000.0, units='lbm')

    # Level 2 Equivalent: Design variables are added through prob.add_design_variables()
    # This function adds variables depending on the type of mission method
    # (N3CC is HEIGHT_ENERGY)
    prob.model.add_design_var(
        Mission.Design.GROSS_MASS, units='lbm', lower=100000.0, upper=200000.0, ref=135000
    )
    prob.model.add_design_var(
        Mission.Summary.GROSS_MASS, units='lbm', lower=100000.0, upper=200000.0, ref=135000
    )

    # Level 2 Equivalent: Takeoff is included by specifying "include_takeoff": True in the
    # "pre_mission" portion of the phase info. That line is processed in the
    # add_pre_mission_systems function
    takeoff_options = Takeoff(
        airport_altitude=alt_airport,  # ft
        # no units
        num_engines=aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES),
    )

    ##################
    # Define Phases  #
    ##################

    # Level 2 Equivalent: Phase information is specified in phase_info
    num_segments_climb = 6
    num_segments_cruise = 1
    num_segments_descent = 5

    climb_seg_ends, _ = dm.utils.lgl.lgl(num_segments_climb + 1)
    descent_seg_ends, _ = dm.utils.lgl.lgl(num_segments_descent + 1)

    # Level 2 Equivalent: Phase builder processes the phase info and creates the
    # transcription. All a part of add_phases
    transcription_climb = dm.Radau(
        num_segments=num_segments_climb, order=3, compressed=True, segment_ends=climb_seg_ends
    )
    transcription_cruise = dm.Radau(num_segments=num_segments_cruise, order=3, compressed=True)
    transcription_descent = dm.Radau(
        num_segments=num_segments_descent, order=3, compressed=True, segment_ends=descent_seg_ends
    )

    # Level 2 Equivalent: Load inputs creates the engine
    # default subsystems
    engines = [build_engine_deck(aviary_inputs)]
    preprocess_propulsion(aviary_inputs, engines)
    default_mission_subsystems = get_default_mission_subsystems('FLOPS', engines)

    # Level 2 Equivalent: Phase options are specified during the add_phases function.
    # In this example, the height energy problem configurator would pass the proper phase
    # builder (EnergyPhase) to the add_phases function
    climb_options = EnergyPhase(
        'test_climb',
        user_options=AviaryValues(
            {
                'altitude_initial': (alt_i_climb, 'm'),
                'altitude_final': (alt_f_climb, 'm'),
                'mach_initial': (mach_i_climb, 'unitless'),
                'mach_final': (mach_f_climb, 'unitless'),
                'fix_initial': (False, 'unitless'),
                'input_initial': (True, 'unitless'),
                'use_polynomial_control': (False, 'unitless'),
            }
        ),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_climb,
    )

    cruise_options = EnergyPhase(
        'test_cruise',
        user_options=AviaryValues(
            {
                'altitude_initial': (alt_min_cruise, 'm'),
                'altitude_final': (alt_max_cruise, 'm'),
                'mach_initial': (cruise_mach, 'unitless'),
                'final_mach': (cruise_mach, 'unitless'),
                'required_available_climb_rate': (300, 'ft/min'),
                'fix_initial': (False, 'unitless'),
            }
        ),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_cruise,
    )

    descent_options = EnergyPhase(
        'test_descent',
        user_options=AviaryValues(
            {
                'altitude_final': (alt_f_descent, 'm'),
                'altitude_initial': (alt_i_descent, 'm'),
                'mach_initial': (mach_i_descent, 'unitless'),
                'mach_final': (mach_f_descent, 'unitless'),
                'fix_initial': (False, 'unitless'),
                'use_polynomial_control': (False, 'unitless'),
            }
        ),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_descent,
    )

    # Level 2 Equivalent: Landing follows the same structure as Takeoff,
    # where specification in the phase info is processed in the add
    # post mission systems function call
    landing_options = Landing(
        ref_wing_area=aviary_inputs.get_val(Aircraft.Wing.AREA, units='ft**2'),
        Cl_max_ldg=aviary_inputs.get_val(Mission.Landing.LIFT_COEFFICIENT_MAX),  # no units
    )

    # Level 2 Equivalent: The builders and preprocessing is handled in the
    # check_and_preprocess_inputs level 2 function
    preprocess_crewpayload(aviary_inputs)

    prop = CorePropulsionBuilder('core_propulsion', BaseMetaData, engines)
    mass = CoreMassBuilder('core_mass', BaseMetaData, FLOPS)
    aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, FLOPS)
    geom = CoreGeometryBuilder('core_geometry', BaseMetaData, code_origin=FLOPS)

    core_subsystems = [prop, geom, mass, aero]

    # Level 2 Equivalent: add_pre_mission_systems function
    # Upstream static analysis for aero
    prob.model.add_subsystem(
        'pre_mission',
        CorePreMission(aviary_options=aviary_inputs, subsystems=core_subsystems),
        promotes_inputs=['aircraft:*', 'mission:*'],
        promotes_outputs=['aircraft:*', 'mission:*'],
    )

    # directly connect phases (strong_couple = True), or use linkage constraints (weak
    # coupling / strong_couple=False)
    strong_couple = False

    # Level 2 Equivalent: Called in add_takeoff_systems in height energy problem configuration,
    # which is called in level 2 add pre mission systems
    takeoff = takeoff_options.build_phase(False)

    # Level 2 Equivalent: Completed through the add_phases routine in level 2,
    # where each phase specified in phase info gets processed
    climb = climb_options.build_phase(aviary_options=aviary_inputs)

    cruise = cruise_options.build_phase(aviary_options=aviary_inputs)

    descent = descent_options.build_phase(aviary_options=aviary_inputs)

    landing = landing_options.build_phase(False)

    # Level 2 Equivalent: Called in add_takeoff_systems in height energy problem
    # configuration, which is called in add pre mission systems
    prob.model.add_subsystem(
        'takeoff',
        takeoff,
        promotes_inputs=['aircraft:*', 'mission:*'],
        promotes_outputs=['mission:*'],
    )

    # Level 2 Equivalent: Completed in add_phases
    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    # Level 2 Equivalent: Completed through set_phase_options, which is called in add_phases

    # if fix_initial is false, can we always set input_initial to be true for
    # necessary states, and then ignore if we use a linkage?
    climb.set_time_options(
        fix_initial=True,
        fix_duration=False,
        units='s',
        duration_bounds=(t_duration_climb * 0.5, t_duration_climb * 2),
        duration_ref=t_duration_climb,
    )
    cruise.set_time_options(
        fix_initial=False,
        fix_duration=False,
        units='s',
        duration_bounds=(t_duration_cruise * 0.5, t_duration_cruise * 2),
        duration_ref=t_duration_cruise,
        initial_bounds=(t_duration_climb * 0.5, t_duration_climb * 2),
    )
    descent.set_time_options(
        fix_initial=False,
        fix_duration=False,
        units='s',
        duration_bounds=(t_duration_descent * 0.5, t_duration_descent * 2),
        duration_ref=t_duration_descent,
        initial_bounds=(
            (t_duration_cruise + t_duration_climb) * 0.5,
            (t_duration_cruise + t_duration_climb) * 2,
        ),
    )

    # Level 2 Equivalent: Completed through add_phase
    traj.add_phase('climb', climb)

    traj.add_phase('cruise', cruise)

    traj.add_phase('descent', descent)

    # Level 2 Equivalent: Processed similarly to takeoff, but in add_post_mission_systems
    prob.model.add_subsystem(
        'landing',
        landing,
        promotes_inputs=['aircraft:*', 'mission:*'],
        promotes_outputs=['mission:*'],
    )

    # Level 2 Equivalent: link_phases function
    traj.link_phases(
        ['climb', 'cruise', 'descent'],
        ['time', Dynamic.Vehicle.MASS, Dynamic.Mission.DISTANCE],
        connected=strong_couple,
    )

    # Level 2 Equivalent: Completed in final steps of add_phases function
    # Need to declare dymos parameters for every input that is promoted out of the missions.
    externs = {'climb': {}, 'cruise': {}, 'descent': {}}
    for default_subsys in default_mission_subsystems:
        params = default_subsys.get_parameters(aviary_inputs=aviary_inputs, phase_info={})
        for key, val in params.items():
            for phname in externs:
                externs[phname][key] = val

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs, external_parameters=externs)

    # Level 2 Equivalent: Completed in add_post_mission_systems
    ##################################
    # Connect in Takeoff and Landing #
    ##################################
    prob.model.connect(Mission.Takeoff.FINAL_MASS, 'traj.climb.initial_states:mass')
    prob.model.connect(Mission.Takeoff.GROUND_DISTANCE, 'traj.climb.initial_states:distance')

    prob.model.connect('traj.descent.states:mass', Mission.Landing.TOUCHDOWN_MASS, src_indices=[-1])
    # TODO: approach velocity should likely be connected
    prob.model.connect(
        'traj.descent.control_values:altitude', Mission.Landing.INITIAL_ALTITUDE, src_indices=[-1]
    )

    ##########################
    # Constraints            #
    ##########################
    # Level 2 Equivalent: Completed at the end of the add_post_mission_systems
    ecomp = om.ExecComp(
        'fuel_burned = initial_mass - descent_mass_final',
        initial_mass={'units': 'lbm', 'shape': 1},
        descent_mass_final={'units': 'lbm', 'shape': 1},
        fuel_burned={'units': 'lbm', 'shape': 1},
    )

    prob.model.add_subsystem(
        'fuel_burn',
        ecomp,
        promotes_inputs=[('initial_mass', Mission.Design.GROSS_MASS)],
        promotes_outputs=['fuel_burned'],
    )

    prob.model.connect('traj.descent.states:mass', 'fuel_burn.descent_mass_final', src_indices=[-1])

    # TODO: need to add some sort of check that this value is less than the fuel capacity
    # TODO: need to update this with actual FLOPS value, this gives unrealistic
    # appearance of accuracy
    # TODO: the overall_fuel variable is the burned fuel plus the reserve, but should
    # also include the unused fuel, and the hierarchy variable name should be more clear
    ecomp = om.ExecComp(
        'overall_fuel = fuel_burned + fuel_reserve',
        fuel_burned={'units': 'lbm', 'shape': 1},
        fuel_reserve={'units': 'lbm', 'val': 2173.0},
        overall_fuel={'units': 'lbm'},
    )
    prob.model.add_subsystem(
        'fuel_calc', ecomp, promotes_inputs=['fuel_burned'], promotes_outputs=['overall_fuel']
    )

    ecomp = om.ExecComp(
        'mass_resid = operating_empty_mass + overall_fuel + payload_mass - initial_mass',
        operating_empty_mass={'units': 'lbm'},
        overall_fuel={'units': 'lbm'},
        payload_mass={'units': 'lbm'},
        initial_mass={'units': 'lbm'},
        mass_resid={'units': 'lbm'},
    )

    prob.model.add_subsystem(
        'mass_constraint',
        ecomp,
        promotes_inputs=[
            ('operating_empty_mass', Aircraft.Design.OPERATING_MASS),
            'overall_fuel',
            ('payload_mass', Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS),
            ('initial_mass', Mission.Design.GROSS_MASS),
        ],
        promotes_outputs=['mass_resid'],
    )

    prob.model.add_constraint('mass_resid', equals=0.0, ref=1.0)

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

    ##########################
    # Add Objective Function #
    ##########################

    # Level 2 Equivalent: Done in add_objective

    # This is an example of a overall mission objective
    # create a compound objective that minimizes climb time and maximizes final mass
    # we are maxing final mass b/c we don't have an independent value for fuel_mass yet
    # we are going to normalize these (making each of the sub-objectives approx = 1 )
    # TODO: change the scaling on climb_duration
    prob.model.add_subsystem(
        'regularization',
        om.ExecComp(
            'reg_objective = fuel_mass/1500',
            reg_objective=0.0,
            fuel_mass={'units': 'lbm', 'shape': 1},
        ),
        promotes_inputs=[('fuel_mass', Mission.Design.FUEL_MASS)],
        promotes_outputs=['reg_objective'],
    )

    prob.model.add_objective('reg_objective', ref=1)

    # Level 2 Equivalent: load_inputs
    varnames = [
        Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
        Aircraft.Wing.SWEEP,
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.THICKNESS_TO_CHORD,
        Mission.Design.GROSS_MASS,
        Mission.Summary.GROSS_MASS,
    ]
    set_aviary_input_defaults(prob.model, varnames, aviary_inputs)

    # Level 2 Equivalent: setup function

    setup_model_options(prob, aviary_inputs)

    prob.setup(force_alloc_complex=True)

    set_aviary_initial_values(prob, aviary_inputs)

    ############################################
    # Initial Settings for States and Controls #
    ############################################

    # Level 2 Equivalent: set_initial_guesses

    prob.set_val('traj.climb.t_initial', t_i_climb, units='s')
    prob.set_val('traj.climb.t_duration', t_duration_climb, units='s')

    prob.set_val(
        'traj.climb.controls:altitude',
        climb.interp(Dynamic.Mission.ALTITUDE, ys=[alt_i_climb, alt_f_climb]),
        units='m',
    )
    prob.set_val(
        'traj.climb.controls:mach',
        climb.interp(Dynamic.Atmosphere.MACH, ys=[mach_i_climb, mach_f_climb]),
        units='unitless',
    )
    prob.set_val(
        'traj.climb.states:mass',
        climb.interp(Dynamic.Vehicle.MASS, ys=[mass_i_climb, mass_f_climb]),
        units='kg',
    )
    prob.set_val(
        'traj.climb.states:distance',
        climb.interp(Dynamic.Mission.DISTANCE, ys=[distance_i_climb, distance_f_climb]),
        units='m',
    )

    prob.set_val('traj.cruise.t_initial', t_i_cruise, units='s')
    prob.set_val('traj.cruise.t_duration', t_duration_cruise, units='s')

    prob.set_val(
        f'traj.cruise.controls:altitude',
        cruise.interp(Dynamic.Mission.ALTITUDE, ys=[alt_i_cruise, alt_f_cruise]),
        units='m',
    )
    prob.set_val(
        f'traj.cruise.controls:mach',
        cruise.interp(Dynamic.Atmosphere.MACH, ys=[cruise_mach, cruise_mach]),
        units='unitless',
    )
    prob.set_val(
        'traj.cruise.states:mass',
        cruise.interp(Dynamic.Vehicle.MASS, ys=[mass_i_cruise, mass_f_cruise]),
        units='kg',
    )
    prob.set_val(
        'traj.cruise.states:distance',
        cruise.interp(Dynamic.Mission.DISTANCE, ys=[distance_i_cruise, distance_f_cruise]),
        units='m',
    )

    prob.set_val('traj.descent.t_initial', t_i_descent, units='s')
    prob.set_val('traj.descent.t_duration', t_duration_descent, units='s')

    prob.set_val(
        'traj.descent.controls:altitude',
        descent.interp(Dynamic.Mission.ALTITUDE, ys=[alt_i_descent, alt_f_descent]),
        units='m',
    )
    prob.set_val(
        'traj.descent.controls:mach',
        descent.interp(Dynamic.Atmosphere.MACH, ys=[mach_i_descent, mach_f_descent]),
        units='unitless',
    )
    prob.set_val(
        'traj.descent.states:mass',
        descent.interp(Dynamic.Vehicle.MASS, ys=[mass_i_descent, mass_f_descent]),
        units='kg',
    )
    prob.set_val(
        'traj.descent.states:distance',
        descent.interp(Dynamic.Mission.DISTANCE, ys=[distance_i_descent, distance_f_descent]),
        units='m',
    )

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    # Level 2 Equivalent: run_aviary_problem
    dm.run_problem(
        prob,
        simulate=sim,
        make_plots=False,
        simulate_kwargs={'times_per_seg': 100, 'atol': 1e-9, 'rtol': 1e-9},
        solution_record_file='N3CC_sizing.db',
    )
    # prob.run_model()
    # z=prob.check_totals(method='cs', step=2e-40, compact_print=False)
    # exit()
    prob.record('final')
    prob.cleanup()

    return prob


@use_tempdirs
class ProblemPhaseTestCase:
    """
    Test sizing using N3CC data.
    """

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_sizing_N3CC(self):
        prob = run_trajectory(sim=False)


if __name__ == '__main__':
    z = ProblemPhaseTestCase()
    z.bench_test_sizing_N3CC()
