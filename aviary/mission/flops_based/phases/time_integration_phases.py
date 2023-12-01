import numpy as np
import dymos as dm
import openmdao.api as om

from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.mission.flops_based.phases.cruise_phase import Cruise
from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj
from aviary.mission.gasp_based.ode.time_integration_base_classes import SimuPyProblem
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import AlphaModes, AnalysisScheme, SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class SGMHeightEnergy(SimuPyProblem):
    def __init__(self,
                 ode_args={},
                 simupy_args={},):
        super().__init__(MissionODE(
            analysis_scheme=AnalysisScheme.SHOOTING,
            **ode_args),
            # alternate_state_names={Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: Dynamic.Mission.MASS,},
            state_names=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.RANGE,
                Dynamic.Mission.ALTITUDE,
        ],
            alternate_state_rate_names={
                Dynamic.Mission.MASS_RATE: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args)

        self.event_channel_names = [
            # Dynamic.Mission.DISTANCE,
            Dynamic.Mission.MASS,
        ]
        self.num_events = len(self.event_channel_names)

    def event_equation_function(self, t, x):
        self.output_equation_function(t, x)
        distance = self.get_val(Dynamic.Mission.DISTANCE,
                                units=self.distance_trigger_units).squeeze()
        distance_trigger = self.get_val(
            "distance_trigger", units=self.distance_trigger_units).squeeze()

        current_mass = self.get_val(Dynamic.Mission.MASS, units="lbm").squeeze()
        mass_trigger = 150000

        return np.array([
            current_mass - mass_trigger
        ])


def test_phase(phases):
    prob = om.Problem()
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = 'IPOPT'
    prob.driver.opt_settings['tol'] = 1.0E-6
    prob.driver.opt_settings['mu_init'] = 1e-5
    prob.driver.opt_settings['max_iter'] = 50
    prob.driver.opt_settings['print_level'] = 5

    traj = FlexibleTraj(
        Phases=phases,
        traj_final_state_output=[Dynamic.Mission.MASS,
                                 Dynamic.Mission.RANGE,
                                 Dynamic.Mission.ALTITUDE],
        traj_initial_state_input=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.RANGE,
            Dynamic.Mission.ALTITUDE,
        ],
    )
    prob.model = om.Group()
    prob.model.add_subsystem('traj', traj)

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

    prob.setup()
    prob.set_val("traj.altitude_initial", val=35000, units="ft")
    prob.set_val("traj.mass_initial", val=171000, units="lbm")
    prob.set_val("traj.range_initial", val=0, units="NM")

    prob.run_model()


if __name__ == '__main__':
    from aviary.interface.default_flops_phases import phase_info
    from aviary.subsystems.propulsion.engine_deck import EngineDeck
    from aviary.variable_info.variables import Aircraft, Mission, Dynamic
    from aviary.utils.UI import create_vehicle
    from aviary.utils.preprocessors import preprocess_propulsion
    from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData

    from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
    from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
    core_subsystems = [CorePropulsionBuilder('core_propulsion'), CoreAerodynamicsBuilder(
        'core_aerodynamics', code_origin='FLOPS')]

    aviary_inputs, initial_guesses = create_vehicle(
        'validation_cases/benchmark_tests/bench_4.csv')
    aviary_inputs.set_val('debug_mode', False)
    aviary_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=28690, units="lbf")
    aviary_inputs.set_val(Dynamic.Mission.THROTTLE, val=0, units="unitless")
    ode_args_tab = dict(aviary_options=aviary_inputs, core_subsystems=core_subsystems)
    engine = EngineDeck(options=aviary_inputs)
    preprocess_propulsion(aviary_inputs, [engine])

    ode_args_tab['num_nodes'] = 1
    ode_args_tab['subsystem_options'] = {'core_aerodynamics': {'method': 'computed'}}
    phase_kwargs = dict(
        ode_args=ode_args_tab,
        simupy_args=dict(
            DEBUG=True,
            # blocked_state_names=['engine.nox', 'nox', 'specific_energy'],
        ),
    )
    phase_vals = {
    }
    phases = {'HE': {
        'ode': SGMHeightEnergy(**phase_kwargs),
        'vals_to_set': phase_vals
    }}

    test_phase(phases)
