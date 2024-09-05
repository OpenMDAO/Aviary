from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.mission.flops_based.ode.landing_ode import LandingODE
from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.mission.gasp_based.ode.time_integration_base_classes import SimuPyProblem
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Dynamic


class SGMHeightEnergy(SimuPyProblem):
    def __init__(
        self,
        ode_args,
        phase_name='mission',
        simupy_args={},
        mass_trigger=(150000, 'lbm')
    ):
        super().__init__(
            MissionODE(analysis_scheme=AnalysisScheme.SHOOTING, **ode_args),
            problem_name=phase_name,
            outputs=[],
            states=[
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Atmosphere.ALTITUDE,
            ],
            alternate_state_rate_names={
                Dynamic.Vehicle.MASS: Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL
            },
            aviary_options=ode_args['aviary_options'],
            **simupy_args
        )

        self.phase_name = phase_name
        self.mass_trigger = mass_trigger
        self.add_trigger(Dynamic.Vehicle.MASS, 'mass_trigger')


class SGMDetailedTakeoff(SimuPyProblem):
    def __init__(
        self,
        ode_args,
        phase_name='detailed_takeoff',
        simupy_args={},
    ):
        super().__init__(
            TakeoffODE(analysis_scheme=AnalysisScheme.SHOOTING, **ode_args),
            problem_name=phase_name,
            outputs=[],
            states=[
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Atmosphere.ALTITUDEUDE,
            ],
            alternate_state_rate_names={
                Dynamic.Vehicle.MASS: Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVETE_NEGATIVE_TOTAL
            },
            aviary_options=ode_args['aviary_options'],
            **simupy_args
        )

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Atmosphere.ALTITUDEUDE, 50, units='ft')


class SGMDetailedLanding(SimuPyProblem):
    def __init__(
        self,
        ode_args,
        phase_name='detailed_landing',
        simupy_args={},
    ):
        super().__init__(
            LandingODE(analysis_scheme=AnalysisScheme.SHOOTING, **ode_args),
            problem_name=phase_name,
            outputs=[],
            states=[
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Atmosphere.ALTITUDEUDE,
            ],
            alternate_state_rate_names={
                Dynamic.Vehicle.MASS: Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVETE_NEGATIVE_TOTAL
            },
            aviary_options=ode_args['aviary_options'],
            **simupy_args
        )

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Atmosphere.ALTITUDEUDE, 0, units='ft')
