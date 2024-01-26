from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessTime, InitialGuessControl
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.ascent_ode import AscentODE
from aviary.variable_info.variable_meta_data import _MetaData

import numpy as np


class AscentPhase(PhaseBuilderBase):
    default_name = 'ascent_phase'
    default_ode_class = AscentODE

    __slots__ = ('external_subsystems', 'meta_data')

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

    default_meta_data = _MetaData

    def __init__(
        self, name=None, subsystem_options=None, user_options=None, initial_guesses=None,
        ode_class=None, transcription=None, core_subsystems=None,
        external_subsystems=None, meta_data=None
    ):
        super().__init__(
            name=name, subsystem_options=subsystem_options, user_options=user_options,
            initial_guesses=initial_guesses, ode_class=ode_class, transcription=transcription,
            core_subsystems=core_subsystems,
        )

        if external_subsystems is None:
            external_subsystems = []

        self.external_subsystems = external_subsystems

        if meta_data is None:
            meta_data = self.default_meta_data

        self.meta_data = meta_data

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        angle_lower = user_options.get_val('angle_lower', units='rad')
        angle_upper = user_options.get_val('angle_upper', units='rad')
        angle_ref = user_options.get_val('angle_ref', units='rad')
        angle_ref0 = user_options.get_val('angle_ref0', units='rad')
        angle_defect_ref = user_options.get_val('angle_defect_ref', units='rad')
        alt_lower = user_options.get_val('alt_lower', units='ft')
        alt_upper = user_options.get_val('alt_upper', units='ft')
        alt_ref = user_options.get_val('alt_ref', units='ft')
        alt_ref0 = user_options.get_val('alt_ref0', units='ft')
        alt_defect_ref = user_options.get_val('alt_defect_ref', units='ft')
        alt_constraint_eq = user_options.get_val('alt_constraint_eq', units='ft')
        alt_constraint_ref = user_options.get_val('alt_constraint_ref', units='ft')
        alt_constraint_ref0 = user_options.get_val('alt_constraint_ref0', units='ft')
        TAS_lower = user_options.get_val('TAS_lower', units='kn')
        TAS_upper = user_options.get_val('TAS_upper', units='kn')
        TAS_ref = user_options.get_val('TAS_ref', units='kn')
        TAS_ref0 = user_options.get_val('TAS_ref0', units='kn')
        TAS_defect_ref = user_options.get_val('TAS_defect_ref', units='kn')
        mass_lower = user_options.get_val('mass_lower', units='lbm')
        mass_upper = user_options.get_val('mass_upper', units='lbm')
        mass_ref = user_options.get_val('mass_ref', units='lbm')
        mass_ref0 = user_options.get_val('mass_ref0', units='lbm')
        mass_defect_ref = user_options.get_val('mass_defect_ref', units='lbm')
        distance_lower = user_options.get_val('distance_lower', units='ft')
        distance_upper = user_options.get_val('distance_upper', units='ft')
        distance_ref = user_options.get_val('distance_ref', units='ft')
        distance_ref0 = user_options.get_val('distance_ref0', units='ft')
        distance_defect_ref = user_options.get_val('distance_defect_ref', units='ft')
        pitch_constraint_lower = user_options.get_val(
            'pitch_constraint_lower', units='deg')
        pitch_constraint_upper = user_options.get_val(
            'pitch_constraint_upper', units='deg')
        pitch_constraint_ref = user_options.get_val('pitch_constraint_ref', units='deg')
        alpha_constraint_lower = user_options.get_val(
            'alpha_constraint_lower', units='rad')
        alpha_constraint_upper = user_options.get_val(
            'alpha_constraint_upper', units='rad')
        alpha_constraint_ref = user_options.get_val('alpha_constraint_ref', units='rad')

        phase.set_time_options(
            units="s",
            targets="t_curr",
            input_initial=True,
            input_duration=True,
        )

        phase.add_state(
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            fix_initial=True,
            fix_final=False,
            lower=angle_lower,
            upper=angle_upper,
            units="rad",
            rate_source=Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            ref=angle_ref,
            defect_ref=angle_defect_ref,
            ref0=angle_ref0,
        )

        phase.add_state(
            Dynamic.Mission.ALTITUDE,
            fix_initial=True,
            fix_final=False,
            lower=alt_lower,
            upper=alt_upper,
            units="ft",
            rate_source=Dynamic.Mission.ALTITUDE_RATE,
            ref=alt_ref,
            defect_ref=alt_defect_ref,
            ref0=alt_ref0,
        )

        phase.add_state(
            "TAS",
            fix_initial=user_options.get_val('fix_initial'),
            fix_final=False,
            lower=TAS_lower,
            upper=TAS_upper,
            units="kn",
            rate_source="TAS_rate",
            ref=TAS_ref,
            defect_ref=TAS_defect_ref,
            ref0=TAS_ref0,
        )

        phase.add_state(
            Dynamic.Mission.MASS,
            fix_initial=user_options.get_val('fix_initial'),
            fix_final=False,
            lower=mass_lower,
            upper=mass_upper,
            units="lbm",
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            ref=mass_ref,
            defect_ref=mass_defect_ref,
            ref0=mass_ref0,
        )

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=user_options.get_val('fix_initial'),
            fix_final=False,
            lower=distance_lower,
            upper=distance_upper,
            units="ft",
            rate_source="distance_rate",
            ref=distance_ref,
            defect_ref=distance_defect_ref,
            ref0=distance_ref0,
        )

        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc="final",
            equals=alt_constraint_eq,
            units="ft",
            ref=alt_constraint_ref,
            ref0=alt_constraint_ref0,
        )

        phase.add_path_constraint(
            "load_factor",
            upper=1.10,
            lower=0.0
        )

        phase.add_path_constraint(
            "fuselage_pitch",
            "theta",
            lower=pitch_constraint_lower,
            upper=pitch_constraint_upper,
            units="deg",
            ref=pitch_constraint_ref,
        )

        phase.add_control(
            "alpha",
            val=0,
            lower=alpha_constraint_lower,
            upper=alpha_constraint_upper,
            units="rad",
            ref=alpha_constraint_ref,
            opt=True,
        )

        phase.add_parameter("t_init_gear", units="s",
                            static_target=True, opt=False, val=38.25)

        phase.add_parameter("t_init_flaps", units="s",
                            static_target=True, opt=False, val=48.21)

        phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("normal_force")
        phase.add_timeseries_output(Dynamic.Mission.MACH)
        phase.add_timeseries_output("EAS", units="kn")
        phase.add_timeseries_output(Dynamic.Mission.LIFT)
        phase.add_timeseries_output("CL")
        phase.add_timeseries_output("CD")

        return phase


# Adding metadata for the AscentPhase
AscentPhase._add_meta_data('fix_initial', val=False)
AscentPhase._add_meta_data('angle_lower', val=-15 * np.pi / 180, units='rad')
AscentPhase._add_meta_data('angle_upper', val=25 * np.pi / 180, units='rad')
AscentPhase._add_meta_data('angle_ref', val=np.deg2rad(1), units='rad')
AscentPhase._add_meta_data('angle_ref0', val=0, units='rad')
AscentPhase._add_meta_data('angle_defect_ref', val=0.01, units='rad')
AscentPhase._add_meta_data('alt_lower', val=0, units='ft')
AscentPhase._add_meta_data('alt_upper', val=700, units='ft')
AscentPhase._add_meta_data('alt_ref', val=100, units='ft')
AscentPhase._add_meta_data('alt_ref0', val=0, units='ft')
AscentPhase._add_meta_data('alt_defect_ref', val=100, units='ft')
AscentPhase._add_meta_data('alt_constraint_eq', val=500, units='ft')
AscentPhase._add_meta_data('alt_constraint_ref', val=100, units='ft')
AscentPhase._add_meta_data('alt_constraint_ref0', val=0, units='ft')
AscentPhase._add_meta_data('TAS_lower', val=0, units='kn')
AscentPhase._add_meta_data('TAS_upper', val=1000, units='kn')
AscentPhase._add_meta_data('TAS_ref', val=1e2, units='kn')
AscentPhase._add_meta_data('TAS_ref0', val=0, units='kn')
AscentPhase._add_meta_data('TAS_defect_ref', val=None, units='kn')
AscentPhase._add_meta_data('mass_lower', val=0, units='lbm')
AscentPhase._add_meta_data('mass_upper', val=190_000, units='lbm')
AscentPhase._add_meta_data('mass_ref', val=100_000, units='lbm')
AscentPhase._add_meta_data('mass_ref0', val=0, units='lbm')
AscentPhase._add_meta_data('mass_defect_ref', val=1e2, units='lbm')
AscentPhase._add_meta_data('distance_lower', val=0, units='ft')
AscentPhase._add_meta_data('distance_upper', val=10.e3, units='ft')
AscentPhase._add_meta_data('distance_ref', val=3000, units='ft')
AscentPhase._add_meta_data('distance_ref0', val=0, units='ft')
AscentPhase._add_meta_data('distance_defect_ref', val=3000, units='ft')
AscentPhase._add_meta_data('pitch_constraint_lower', val=0, units='deg')
AscentPhase._add_meta_data('pitch_constraint_upper', val=15, units='deg')
AscentPhase._add_meta_data('pitch_constraint_ref', val=1, units='deg')
AscentPhase._add_meta_data('alpha_constraint_lower', val=np.deg2rad(-30), units='rad')
AscentPhase._add_meta_data('alpha_constraint_upper', val=np.deg2rad(30), units='rad')
AscentPhase._add_meta_data('alpha_constraint_ref', val=np.deg2rad(5), units='rad')
AscentPhase._add_meta_data('num_segments', val=None, units='unitless')
AscentPhase._add_meta_data('order', val=None, units='unitless')

# Adding initial guess metadata
AscentPhase._add_initial_guess_meta_data(
    InitialGuessTime(),
    desc='initial guess for time options')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('flight_path_angle'),
    desc='initial guess for flight path angle state')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for altitude state')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('TAS'),
    desc='initial guess for true airspeed state')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass state')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for distance state')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessControl('alpha'),
    desc='initial guess for angle of attack control')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessControl('tau_gear'),
    desc='when the gear is retracted')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessControl('tau_flaps'),
    desc='when the flaps are retracted')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
