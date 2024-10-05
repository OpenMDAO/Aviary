from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.mission.flops_based.ode.landing_ode import LandingODE
from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.mission.gasp_based.ode.time_integration_base_classes import SimuPyProblem
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Dynamic


class SGMHeightEnergy(SimuPyProblem):
    """
    This creates a subproblem that will be used by most height energy phases during a trajectory that will
    be solved using SGM.
    A mass trigger is added as an example, but any other trigger can be added as necessary.
    """

    def __init__(
        self,
        ode_args,
        phase_name='mission',
        simupy_args={},
        mass_trigger=(150000, 'lbm')
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
            aviary_options=ode_args['aviary_options'],
            **simupy_args)

        self.phase_name = phase_name
        self.mass_trigger = mass_trigger
        self.add_trigger(Dynamic.Mission.MASS, 'mass_trigger')


class SGMDetailedTakeoff(SimuPyProblem):
    """
    This creates a subproblem that will be used by height energy phases during detailed takeoff that will
    be solved using SGM.
    An altitude trigger is added as an example, but any other trigger can be added as necessary in order to
    string together the phases needed for a noise certification takeoff.
    """

    def __init__(
        self,
        ode_args,
        phase_name='detailed_takeoff',
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
            aviary_options=ode_args['aviary_options'],
            **simupy_args)

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Mission.ALTITUDE, 50, units='ft')


class SGMDetailedLanding(SimuPyProblem):
    """
    This creates a subproblem that will be used by height energy phases during detailed landing that will
    be solved using SGM.
    An altitude trigger is added as an example, but any other trigger can be added as necessary in order to
    string together the phases needed for a noise certification landing.
    """

    def __init__(
        self,
        ode_args,
        phase_name='detailed_landing',
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
            aviary_options=ode_args['aviary_options'],
            **simupy_args)

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Mission.ALTITUDE, 0, units='ft')
