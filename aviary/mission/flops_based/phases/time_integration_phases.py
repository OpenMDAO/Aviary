import numpy as np
import openmdao.api as om

from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.mission.flops_based.ode.landing_ode import LandingODE
from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj
from aviary.mission.gasp_based.ode.time_integration_base_classes import SimuPyProblem
from aviary.variable_info.enums import AnalysisScheme, Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.variables_in import VariablesIn
from aviary.subsystems.premission import CorePreMission
from aviary.utils.functions import set_aviary_initial_values
import warnings


class SGMHeightEnergy(SimuPyProblem):
    def __init__(
        self,
        phase_name='cruise',
        ode_args={},
        simupy_args={},
    ):
        super().__init__(MissionODE(
            analysis_scheme=AnalysisScheme.SHOOTING,
            **ode_args),
            problem_name=phase_name,
            outputs=[],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
        ],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args)

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Mission.MASS, 150000, units='lbm')


class SGMDetailedTakeoff(SimuPyProblem):
    def __init__(
        self,
        phase_name='detailed_takeoff',
        ode_args={},
        simupy_args={},
    ):
        super().__init__(TakeoffODE(
            analysis_scheme=AnalysisScheme.SHOOTING,
            **ode_args),
            problem_name=phase_name,
            outputs=[],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
        ],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args)

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Mission.ALTITUDE, 50, units='ft')


class SGMDetailedLanding(SimuPyProblem):
    def __init__(
        self,
        phase_name='detailed_landing',
        ode_args={},
        simupy_args={},
    ):
        super().__init__(LandingODE(
            analysis_scheme=AnalysisScheme.SHOOTING,
            **ode_args),
            problem_name=phase_name,
            outputs=[],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
        ],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args)

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Mission.ALTITUDE, 0, units='ft')


def test_phase(phases, ode_args_tab):
    prob = om.Problem()
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = 'IPOPT'
    prob.driver.opt_settings['tol'] = 1.0E-6
    prob.driver.opt_settings['mu_init'] = 1e-5
    prob.driver.opt_settings['max_iter'] = 50
    prob.driver.opt_settings['print_level'] = 5

    aviary_options = ode_args_tab['aviary_options']
    subsystems = ode_args_tab['core_subsystems']

    traj = FlexibleTraj(
        Phases=phases,
        promote_all_auto_ivc=True,
        traj_final_state_output=[Dynamic.Mission.MASS,
                                 Dynamic.Mission.DISTANCE,
                                 Dynamic.Mission.ALTITUDE],
        traj_initial_state_input=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
    )
    prob.model = om.Group()
    prob.model.add_subsystem(
        'pre_mission',
        CorePreMission(aviary_options=aviary_options,
                       subsystems=subsystems),
        promotes_inputs=['aircraft:*', 'mission:*'],
        promotes_outputs=['aircraft:*', 'mission:*']
    )
    prob.model.add_subsystem('traj', traj,
                             promotes=['aircraft:*', 'mission:*']
                             )

    prob.model.add_subsystem(
        "fuel_obj",
        om.ExecComp(
            "reg_objective = overall_fuel/10000",
            reg_objective={"val": 0.0, "units": "unitless"},
            overall_fuel={"units": "lbm"},
        ),
        promotes_inputs=[
            ("overall_fuel", Mission.Summary.TOTAL_FUEL_MASS),
        ],
        promotes_outputs=[("reg_objective", Mission.Objectives.FUEL)],
    )

    prob.model.add_objective(Mission.Objectives.FUEL, ref=1e4)

    prob.model.add_subsystem(
        'input_sink',
        VariablesIn(aviary_options=aviary_inputs,
                    meta_data=BaseMetaData),
        promotes_inputs=['*'],
        promotes_outputs=['*'])

    with warnings.catch_warnings():

        # Set initial default values for all LEAPS aircraft variables.
        set_aviary_initial_values(
            prob.model, aviary_inputs, meta_data=BaseMetaData)

        warnings.simplefilter("ignore", om.PromotionWarning)

        prob.setup()

    initial_values_takeoff = {
        "traj.altitude_initial": {'val': 0, 'units': "ft"},
        "traj.mass_initial": {'val': 171000, 'units': "lbm"},
        "traj.distance_initial": {'val': 0, 'units': "NM"},
        "traj.velocity": {'val': 0, 'units': "m/s"},
    }
    initial_values_cruise = {
        "traj.altitude_initial": {'val': 35000, 'units': "ft"},
        "traj.mass_initial": {'val': 171000, 'units': "lbm"},
        "traj.distance_initial": {'val': 0, 'units': "NM"},
        "traj.mach": {'val': .8, 'units': "unitless"},
    }
    initial_values_landing = {
        "traj.altitude_initial": {'val': 35000, 'units': "ft"},
        "traj.mass_initial": {'val': 171000, 'units': "lbm"},
        "traj.distance_initial": {'val': 0, 'units': "NM"},
        "traj.velocity": {'val': 300, 'units': "m/s"},
    }

    for key, val in initial_values_cruise.items():
        prob.set_val(key, **val)

    prob.run_model()

    with open('input_list.txt', 'w') as outfile:
        prob.model.list_inputs(out_stream=outfile,)
    with open('output_list.txt', 'w') as outfile:
        prob.model.list_outputs(out_stream=outfile)

    final_distance = prob.get_val('traj.distance_final', units='NM')[0]
    final_mass = prob.get_val('traj.mass_final', units='lbm')[0]
    print(final_distance, final_mass)


if __name__ == '__main__':
    from aviary.interface.default_phase_info.height_energy import phase_info, aero, prop, geom
    from aviary.subsystems.propulsion.engine_deck import EngineDeck
    from aviary.variable_info.variables import Aircraft, Mission, Dynamic
    from aviary.utils.process_input_decks import create_vehicle
    from aviary.utils.preprocessors import preprocess_propulsion
    from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData

    core_subsystems = [prop, geom, aero]

    aviary_inputs, initial_guesses = create_vehicle(
        'models/test_aircraft/aircraft_for_bench_FwFm.csv')
    # aviary_inputs.set_val('debug_mode', False)
    aviary_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=28690, units="lbf")
    aviary_inputs.set_val(Dynamic.Mission.THROTTLE, val=0, units="unitless")
    aviary_inputs.set_val(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT,
                          val=0.0175, units="unitless")
    aviary_inputs.set_val(Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT,
                          val=0.35, units="unitless")
    ode_args_tab = dict(aviary_options=aviary_inputs, core_subsystems=core_subsystems)
    engine = EngineDeck(options=aviary_inputs)
    preprocess_propulsion(aviary_inputs, [engine])

    ode_args_tab['num_nodes'] = 1
    ode_args_tab['subsystem_options'] = {'core_aerodynamics': {'method': 'computed'}}
    # ode_args_tab['friction_key'] = Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT
    # ode_args_tab['friction_key'] = Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT

    phase_kwargs = dict(
        ode_args=ode_args_tab,
        simupy_args=dict(
            verbosity=Verbosity.QUIET,
        ),
    )

    SGMCruise = SGMHeightEnergy(**phase_kwargs)
    SGMCruise.triggers[0].value = 160000

    phase_vals = {
    }
    phases = {'HE': {
        'ode': SGMCruise,
        'vals_to_set': phase_vals
    }}

    test_phase(phases, ode_args_tab)
