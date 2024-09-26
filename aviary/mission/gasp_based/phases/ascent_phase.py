import numpy as np

from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.ascent_ode import AscentODE


class AscentPhase(PhaseBuilderBase):
    """
    A phase builder for an ascent phase in a 2-degree of freedom mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the ascent phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the ascent phase are included.
    """

    default_name = 'ascent_phase'
    default_ode_class = AscentODE

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options

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

        self.add_flight_path_angle_state(user_options)
        self.add_altitude_state(user_options)
        self.add_velocity_state(user_options)
        self.add_mass_state(user_options)
        self.add_distance_state(user_options, units='ft')

        self.add_altitude_constraint(user_options)

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
AscentPhase._add_meta_data(
    'analytic', val=False, desc='this is an analytic phase (no states).')
AscentPhase._add_meta_data(
    'reserve', val=False, desc='this phase is part of the reserve mission.')
AscentPhase._add_meta_data(
    'target_distance', val={}, desc='the amount of distance traveled in this phase added as a constraint')
AscentPhase._add_meta_data(
    'target_duration', val={}, desc='the amount of time taken by this phase added as a constraint')
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
AscentPhase._add_meta_data('final_altitude', val=500, units='ft')
AscentPhase._add_meta_data('alt_constraint_ref', val=100, units='ft')
AscentPhase._add_meta_data('alt_constraint_ref0', val=0, units='ft')
AscentPhase._add_meta_data('velocity_lower', val=0, units='kn')
AscentPhase._add_meta_data('velocity_upper', val=1000, units='kn')
AscentPhase._add_meta_data('velocity_ref', val=1e2, units='kn')
AscentPhase._add_meta_data('velocity_ref0', val=0, units='kn')
AscentPhase._add_meta_data('velocity_defect_ref', val=None, units='kn')
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
    InitialGuessIntegrationVariable(),
    desc='initial guess for time options')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('flight_path_angle'),
    desc='initial guess for flight path angle state')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for altitude state')

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'),
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
