import numpy as np

from aviary.mission.gasp_based.ode.rotation_ode import RotationODE
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class RotationPhase(PhaseBuilderBase):
    """
    A phase builder for a rotation phase in a 2-degree of freedom mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the rotation phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the rotation phase are included.
    """

    default_name = 'rotation_phase'
    default_ode_class = RotationODE

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        fix_initial = user_options.get_val('fix_initial')
        angle_lower = user_options.get_val('angle_lower', units='rad')
        angle_upper = user_options.get_val('angle_upper', units='rad')
        angle_ref = user_options.get_val('angle_ref', units='rad')
        angle_ref0 = user_options.get_val('angle_ref0', units='rad')
        angle_defect_ref = user_options.get_val('angle_defect_ref', units='rad')
        distance_lower = user_options.get_val('distance_lower', units='ft')
        distance_upper = user_options.get_val('distance_upper', units='ft')
        distance_ref = user_options.get_val('distance_ref', units='ft')
        distance_ref0 = user_options.get_val('distance_ref0', units='ft')
        distance_defect_ref = user_options.get_val('distance_defect_ref', units='ft')
        normal_ref = user_options.get_val('normal_ref', units='lbf')
        normal_ref0 = user_options.get_val('normal_ref0', units='lbf')

        # Add states
        phase.add_state(
            "alpha",
            fix_initial=True,
            fix_final=False,
            lower=angle_lower,
            upper=angle_upper,
            units="rad",
            rate_source="alpha_rate",
            ref=angle_ref,
            ref0=angle_ref0,
            defect_ref=angle_defect_ref,
        )

        self.add_velocity_state(user_options)

        self.add_mass_state(user_options)

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=fix_initial,
            fix_final=False,
            lower=distance_lower,
            upper=distance_upper,
            units="ft",
            rate_source="distance_rate",
            ref=distance_ref,
            ref0=distance_ref0,
            defect_ref=distance_defect_ref,
        )

        # Add parameters
        phase.add_parameter("t_init_gear", units="s",
                            static_target=True, opt=False, val=100)
        phase.add_parameter("t_init_flaps", units="s",
                            static_target=True, opt=False, val=100)

        # Add boundary constraints
        phase.add_boundary_constraint(
            "normal_force",
            loc="final",
            equals=0,
            units="lbf",
            ref=normal_ref,
            ref0=normal_ref0,
        )

        # Add timeseries outputs
        phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("normal_force")
        phase.add_timeseries_output(Dynamic.Mission.MACH)
        phase.add_timeseries_output("EAS", units="kn")
        phase.add_timeseries_output(Dynamic.Mission.LIFT)
        phase.add_timeseries_output("CL")
        phase.add_timeseries_output("CD")
        phase.add_timeseries_output("fuselage_pitch", output_name="theta", units="deg")

        return phase


# Adding metadata for the RotationPhase
RotationPhase._add_meta_data(
    'analytic', val=False, desc='this is an analytic phase (no states).')
RotationPhase._add_meta_data(
    'reserve', val=False, desc='this phase is part of the reserve mission.')
RotationPhase._add_meta_data(
    'target_distance', val={}, desc='the amount of distance traveled in this phase added as a constraint')
RotationPhase._add_meta_data(
    'target_duration', val={}, desc='the amount of time taken by this phase added as a constraint')
RotationPhase._add_meta_data('fix_initial', val=False)
RotationPhase._add_meta_data('duration_bounds', val=(1, 100), units='s')
RotationPhase._add_meta_data('duration_ref', val=1, units='s')
RotationPhase._add_meta_data('angle_lower', val=0.0, units='rad')  # rad
RotationPhase._add_meta_data('angle_upper', val=25 * np.pi / 180, units='rad')  # rad
RotationPhase._add_meta_data('angle_ref', val=1, units='rad')
RotationPhase._add_meta_data('angle_ref0', val=0, units='rad')
RotationPhase._add_meta_data('angle_defect_ref', val=0.01, units='rad')
RotationPhase._add_meta_data('velocity_lower', val=0, units='kn')
RotationPhase._add_meta_data('velocity_upper', val=1000, units='kn')
RotationPhase._add_meta_data('velocity_ref', val=100, units='kn')
RotationPhase._add_meta_data('velocity_ref0', val=0, units='kn')
RotationPhase._add_meta_data('velocity_defect_ref', val=None, units='kn')
RotationPhase._add_meta_data('mass_lower', val=0, units='lbm')
RotationPhase._add_meta_data('mass_upper', val=190_000, units='lbm')
RotationPhase._add_meta_data('mass_ref', val=100_000, units='lbm')
RotationPhase._add_meta_data('mass_ref0', val=0, units='lbm')
RotationPhase._add_meta_data('mass_defect_ref', val=None, units='lbm')
RotationPhase._add_meta_data('distance_lower', val=0, units='ft')
RotationPhase._add_meta_data('distance_upper', val=10.e3, units='ft')
RotationPhase._add_meta_data('distance_ref', val=3000, units='ft')
RotationPhase._add_meta_data('distance_ref0', val=0, units='ft')
RotationPhase._add_meta_data('distance_defect_ref', val=3000, units='ft')
RotationPhase._add_meta_data('normal_ref', val=1, units='lbf')
RotationPhase._add_meta_data('normal_ref0', val=0, units='lbf')
RotationPhase._add_meta_data('t_init_gear', val=100, units='s')
RotationPhase._add_meta_data('t_init_flaps', val=100, units='s')
RotationPhase._add_meta_data('num_segments', val=None, units='unitless')
RotationPhase._add_meta_data('order', val=None, units='unitless')

# Adding initial guess metadata
RotationPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for time options')
RotationPhase._add_initial_guess_meta_data(
    InitialGuessState('alpha'),
    desc='initial guess for angle of attack state')
RotationPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'),
    desc='initial guess for true airspeed state')
RotationPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass state')
RotationPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for distance state')
RotationPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
