import numpy as np
import openmdao.api as om

from aviary.mission.gasp_based.ode.accel_ode import AccelODE
from aviary.mission.gasp_based.ode.ascent_ode import AscentODE
from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.mission.gasp_based.ode.descent_ode import DescentODE
from aviary.mission.gasp_based.ode.flight_path_ode import FlightPathODE
from aviary.mission.gasp_based.phases.landing_group import LandingSegment
from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.mission.gasp_based.ode.rotation_ode import RotationODE
from aviary.mission.gasp_based.ode.time_integration_base_classes import SimuPyProblem
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import AlphaModes, AnalysisScheme, SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

DEBUG = 0


class SGMGroundroll(SimuPyProblem):
    '''
    This creates a subproblem for the groundroll phase of the trajectory that will
    be solved using SGM.
    Groundroll ends when TAS reaches rotation speed.
    '''

    def __init__(
        self,
        phase_name='groundroll',
        VR_value=143.1,
        VR_units="kn",
        ode_args={},
        simupy_args={},
    ):

        super().__init__(
            GroundrollODE(analysis_scheme=AnalysisScheme.SHOOTING, **ode_args),
            outputs=["normal_force"],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
                Dynamic.Mission.VELOCITY,
            ],
            # state_units=['lbm','nmi','ft','ft/s'],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args,
        )

        self.phase_name = phase_name
        self.VR_value = VR_value
        # self.VR_units = VR_units
        self.add_trigger(Dynamic.Mission.VELOCITY, "VR_value", units='ft/s')

    def event_equation_function(self, t, x):
        self.time = t
        self.state = x
        return self.get_val(Dynamic.Mission.VELOCITY, units='ft/s') - self.VR_value


class SGMRotation(SimuPyProblem):
    '''
    This creates a subproblem for the rotation phase of the trajectory that will
    be solved using SGM.
    Rotation ends when the normal force on the runway reaches 0.
    '''

    def __init__(
        self,
        phase_name='rotation',
        ode_args={},
        simupy_args={},
    ):
        super().__init__(
            RotationODE(analysis_scheme=AnalysisScheme.SHOOTING, **ode_args),
            outputs=["normal_force", "alpha"],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
            ],
            # state_units=['lbm','nmi','ft'],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args,
        )

        self.phase_name = phase_name
        self.add_trigger("normal_force", 0, units='lbf')


# TODO : turn these into parameters? inputs? they need to match between
# ODE and SimuPy wrappers
load_factor_max = 1.10
velocity_rate_safety = -np.inf  # 100.
fuselage_pitch_max = 15.0
gear_retraction_alt = 50.0
flap_retraction_alt = 400.0
ascent_termination_alt = 500.0


class SGMAscent(SimuPyProblem):
    '''
    This creates a subproblem for the ascent phase of the trajectory that will
    be solved using SGM.
    Ascent ends at ascent_termination_alt and retracts the gear and flaps at their
    respective retraction altitudes.
    '''

    def __init__(
        self,
        phase_name='ascent',
        alpha_mode=AlphaModes.DEFAULT,
        ode_args={},
        simupy_args={},
    ):
        controls = None
        super().__init__(
            AscentODE(analysis_scheme=AnalysisScheme.SHOOTING,
                      alpha_mode=alpha_mode, **ode_args),
            outputs=[
                "load_factor",
                "fuselage_pitch",
                "normal_force",
                "alpha",
            ],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
            ],
            # state_units=['lbm','nmi','ft'],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            controls=controls,
            **simupy_args,
        )

        self.phase_name = phase_name
        self.event_channel_names = [
            Dynamic.Mission.ALTITUDE,
            Dynamic.Mission.ALTITUDE,
            Dynamic.Mission.ALTITUDE,
        ]
        self.num_events = len(self.event_channel_names)

        self.event_names = [
            "termination",
            "flaps",
            "gear",
        ]

    def event_equation_function(self, t, x):
        alpha = self.get_alpha(t, x)
        self.ode0.set_val("alpha", alpha)
        self.ode0.output_equation_function(t, x)
        alt = self.ode0.get_val(Dynamic.Mission.ALTITUDE).squeeze()
        return np.array(
            [
                alt - ascent_termination_alt,
                alt - flap_retraction_alt,
                alt - gear_retraction_alt,
            ]
        ).squeeze()

    def update_equation_function(self, t, x, event_channels=None):
        if 0 in event_channels:
            self.output_nan = True
            return x
        elif 1 in event_channels:
            if DEBUG:
                print("flaps!", t)
            self.set_val("t_init_flaps", t)
        elif 2 in event_channels:
            if DEBUG:
                print("gear!", t)
            self.set_val("t_init_gear", t)
        else:
            return np.nan * np.ones(self.dim_state)
        return x


class SGMAscentCombined(SGMAscent):
    '''
    This combines the different methods of limiting angle of attack to ensure that
    none of the constraints are violated.
    '''

    def __init__(
        self,
        phase_name='ascent_combined',
        fuselage_pitch_max=0,
        ode_args={},
        simupy_args={},
    ):
        self.ode_args = ode_args
        super().__init__(alpha_mode=AlphaModes.DEFAULT, ode_args=ode_args, simupy_args=simupy_args)

        self.phase_name = phase_name
        self.fuselage_pitch_max = fuselage_pitch_max

        ode0 = SGMAscent(alpha_mode=AlphaModes.DEFAULT,
                         ode_args=ode_args, simupy_args=simupy_args)
        rotation = SGMAscent(alpha_mode=AlphaModes.ROTATION,
                             ode_args=ode_args, simupy_args=simupy_args)
        load_factor = SGMAscent(alpha_mode=AlphaModes.LOAD_FACTOR,
                                ode_args=ode_args, simupy_args=simupy_args)
        fuselage_pitch = SGMAscent(
            alpha_mode=AlphaModes.FUSELAGE_PITCH, ode_args=ode_args, simupy_args=simupy_args)
        decel = SGMAscent(alpha_mode=AlphaModes.DECELERATION,
                          ode_args=ode_args, simupy_args=simupy_args)

        self.odes = (ode0, rotation, load_factor, fuselage_pitch, decel,)
        (
            self.ode0,
            self.rotation,
            self.load_factor,
            self.fuselage_pitch,
            self.decel
        ) = (
            ode0, rotation, load_factor, fuselage_pitch, decel,
        )

        self.set_val("t_init_flaps", 500.0, units='s')
        self.set_val("t_init_gear", 500.0, units='s')

        self.ode_name = {
            ode0: "ode0",
            rotation: "rotation",
            load_factor: "load_factor",
            fuselage_pitch: "fuselage pitch",
            decel: "decel"
        }

    def prepare_to_integrate(self, t0, x0):
        self.output_nan = False
        self.alpha_cache = {}
        self.prob_cache = {}
        self.last_prob = self.rotation
        for ode in self.odes[:]:
            ode.prepare_to_integrate(t0, x0)
        return self.output_equation_function(t0, x0)
        # or can this just be done with prepare to integrate??

    def set_val(self, *args, **kwargs):
        for ode in self.odes:
            ode.set_val(*args, **kwargs)

    def compute_alpha(self, ode, t, x):
        return ode.output_equation_function(t, x)[list(ode.outputs.keys()).index("alpha")]

    def get_alpha(self, t, x):
        a_key = (t,) + tuple(x)
        # TODO: I think there's a pythonic way to do this
        if a_key in self.alpha_cache:
            alpha = self.alpha_cache[a_key]
        else:
            # in deriv.f, alpha from previous time-step is known and used as seed
            # assume scheduled increment on alpha
            # then check fuselage pitch, clip to max
            # then check load factor, line search -alpha until satisfied
            # then check end SPEED -- keep tas_rate = 0 if tas > vend
            # (not implemented in gaspy)
            # then check decel, line search -alpha until satisfied
            (ode0, rotation, load_factor, fuselage_pitch, decel,) = self.odes
            ode = self.last_prob
            SATISFIED_CONSTRAINTS = False
            for count in range(4):
                alpha = self.compute_alpha(ode, t, x)
                load_factor_val = ode.get_val("load_factor")
                fuselage_pitch_val = ode.get_val("fuselage_pitch", units="deg")
                velocity_rate_val = ode.get_val("velocity_rate")

                if (
                    (load_factor_val > load_factor_max) and not
                    np.isclose(load_factor_val, load_factor_max)
                ):
                    print('*'*20, 'switching to load_factor', '*'*20)
                    ode = load_factor
                    continue
                elif (
                    (fuselage_pitch_val > fuselage_pitch_max) and not
                    np.isclose(fuselage_pitch_val, fuselage_pitch_max)
                ):
                    print('*'*20, 'switching to fuselage_pitch', '*'*20)
                    ode = fuselage_pitch
                    continue
                elif (
                    (velocity_rate_val < velocity_rate_safety) and not
                    np.isclose(velocity_rate_val, velocity_rate_safety)
                ):
                    print('*'*20, 'switching to decel', '*'*20)
                    ode = decel
                    continue
                else:
                    if (
                        np.isnan(load_factor_val) or
                        np.isnan(fuselage_pitch_val) or
                        np.isnan(velocity_rate_val)
                    ):
                        continue
                    SATISFIED_CONSTRAINTS = True
                    break
            if SATISFIED_CONSTRAINTS:
                self.alpha_cache[a_key] = alpha
                self.prob_cache[a_key] = ode
                self.last_prob = ode
            else:
                print("time :", t)
                print("ode :", self.ode_name[ode])
                for key in ["load_factor", "fuselage_pitch", "velocity_rate"]:
                    print(key, ":", ode.get_val(key))
                raise ValueError("Ascent could not satisfy all constraints")

        return alpha

    def get_prob(self, t, x):
        a_key = (t,) + tuple(x)
        if a_key not in self.prob_cache:
            self.get_alpha(t, x)
        return self.prob_cache[a_key]

    def state_equation_function(self, t, x, u=None):
        if np.any(np.isnan(x)):
            return np.ones(self.dim_output) * np.nan
        alpha = self.get_alpha(t, x)
        prob = self.get_prob(t, x)
        prob.set_val("alpha", alpha)
        return prob.state_equation_function(t, x)

    @property
    def compute_totals(self):
        return self.get_prob(self.time, self.state).compute_totals

    def output_equation_function(self, t, x):
        if np.any(np.isnan(x)) or self.output_nan:
            return np.ones(self.dim_output) * np.nan
        alpha = self.get_alpha(t, x)
        prob = self.get_prob(t, x)
        # I think always need to use ode0 and set alpha --
        # caching source problem doesn't necessarily save anything since you don't know
        # it was the last compute. but maybe it's worth a shot?
        # using solver may introduce slight variations depending on how it's walking or
        # not? and need to have a real compute before compute totals - or does that mean
        # use problem?
        prob.set_val("alpha", alpha)
        self.time = t
        self.state = x
        return prob.output_equation_function(t, x)


class SGMAccel(SimuPyProblem):
    '''
    This creates a subproblem for the acceleration phase of the trajectory that will
    be solved using SGM
    '''

    def __init__(
        self,
        phase_name='accel',
        VC_value=250.0,
        VC_units="kn",
        ode_args={},
        simupy_args={},
    ):
        ode = AccelODE(analysis_scheme=AnalysisScheme.SHOOTING, **ode_args)
        super().__init__(
            ode,
            outputs=["EAS", "mach", "alpha"],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
            ],
            # state_units=['lbm','nmi','ft'],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args,
        )
        self.phase_name = phase_name
        self.add_trigger("EAS", VC_value, units=VC_units)


class SGMClimb(SimuPyProblem):
    '''
    This creates a subproblem for the climb phase of the trajectory that will
    be solved using SGM
    '''

    def __init__(
        self,
        phase_name='climb',
        input_speed_type=SpeedType.EAS,
        input_speed_units="kn",
        speed_trigger_units=None,
        alt_trigger_units="ft",
        ode_args={},
        simupy_args={},
    ):
        ode = ClimbODE(
            analysis_scheme=AnalysisScheme.SHOOTING,
            input_speed_type=input_speed_type,
            speed_trigger_units=speed_trigger_units,
            alt_trigger_units=alt_trigger_units,
            **ode_args,
        )
        self.input_speed_type = input_speed_type
        self.input_speed_units = input_speed_units
        if input_speed_type is SpeedType.EAS:
            self.speed_trigger_name = "mach"
        elif input_speed_type is SpeedType.MACH:
            self.speed_trigger_name = "TAS_violation"
        else:
            raise ValueError("bad speed type")
        self.speed_trigger_units = speed_trigger_units
        self.alt_trigger_units = alt_trigger_units
        super().__init__(
            ode,
            outputs=[
                "alpha",
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                "required_lift",
                "lift",
                "mach",
                "EAS",
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.THRUST_TOTAL,
                "drag",
                Dynamic.Mission.ALTITUDE_RATE,
            ],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
            ],
            # state_units=['lbm','nmi','ft'],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args,
        )
        self.phase_name = phase_name
        self.add_trigger(Dynamic.Mission.ALTITUDE, "alt_trigger",
                         units=self.alt_trigger_units)
        self.add_trigger(self.speed_trigger_name, "speed_trigger",
                         units=self.speed_trigger_units)


class SGMCruise(SimuPyProblem):
    '''
    This creates a subproblem for the cruise phase of the trajectory that will
    be solved using SGM
    '''

    def __init__(
        self,
        phase_name='cruise',
        alpha_mode=AlphaModes.DEFAULT,
        input_speed_type=SpeedType.MACH,
        input_speed_units="kn",
        distance_trigger_units='ft',
        ode_args={},
        simupy_args={},
    ):
        ode = FlightPathODE(
            analysis_scheme=AnalysisScheme.SHOOTING,
            alpha_mode=alpha_mode,
            input_speed_type=input_speed_type,
            clean=True,
            **ode_args,)

        self.distance_trigger_units = distance_trigger_units

        super().__init__(
            ode,
            outputs=[
                "alpha",  # ?
                "lift",
                "EAS",
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.THRUST_TOTAL,
                "drag",
                Dynamic.Mission.ALTITUDE_RATE,
            ],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
            ],
            # state_units=['lbm','nmi','ft'],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args,
        )

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Mission.MASS, 'mass_trigger.mass_trigger', units='lbm')
        self.add_trigger(Dynamic.Mission.DISTANCE, "distance_trigger",
                         units=self.distance_trigger_units)


class SGMDescent(SimuPyProblem):
    '''
    This creates a subproblem for the descent phase of the trajectory that will
    be solved using SGM
    '''

    def __init__(
        self,
        phase_name='descent',
        input_speed_type=SpeedType.EAS,
        input_speed_units="kn",
        speed_trigger_units=None,
        alt_trigger_units="ft",
        ode_args={},
        simupy_args={},
    ):
        ode = DescentODE(
            analysis_scheme=AnalysisScheme.SHOOTING,
            input_speed_type=input_speed_type,
            speed_trigger_units=speed_trigger_units,
            alt_trigger_units=alt_trigger_units,
            **ode_args,
        )
        self.input_speed_type = input_speed_type
        self.input_speed_units = input_speed_units
        if input_speed_type is SpeedType.MACH:
            self.speed_trigger_name = "EAS"
        elif input_speed_type is SpeedType.EAS:
            self.speed_trigger_name = "TAS_violation"
        else:
            raise ValueError("bad speed type")
        self.speed_trigger_units = speed_trigger_units
        self.alt_trigger_units = alt_trigger_units
        super().__init__(
            ode,
            outputs=[
                "alpha",
                "required_lift",
                "lift",
                "EAS",
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.THRUST_TOTAL,
                "drag",
                Dynamic.Mission.ALTITUDE_RATE,
            ],
            states=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
            ],
            # state_units=['lbm','nmi','ft'],
            alternate_state_rate_names={
                Dynamic.Mission.MASS: Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL},
            **simupy_args,
        )

        self.phase_name = phase_name
        self.add_trigger(Dynamic.Mission.ALTITUDE, "alt_trigger",
                         units=self.alt_trigger_units)
        self.add_trigger(self.speed_trigger_name, "speed_trigger",
                         units=self.speed_trigger_units)
