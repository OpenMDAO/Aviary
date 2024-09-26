from aviary.mission.gasp_based.ode.descent_ode import DescentODE
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class DescentPhase(PhaseBuilderBase):
    """
    A phase builder for an descent phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the descent phase of a 2-degree of freedom flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the descent phase are included.
    """

    default_name = 'descent_phase'
    default_ode_class = DescentODE

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        input_speed_type = user_options.get_val('input_speed_type')
        EAS_limit = user_options.get_val('EAS_limit', units='kn')

        # Add states
        self.add_altitude_state(user_options)

        self.add_mass_state(user_options)

        self.add_distance_state(user_options)

        # Add boundary constraint
        self.add_altitude_constraint(user_options)

        # Add parameter if necessary
        if input_speed_type == SpeedType.EAS:
            phase.add_parameter("EAS", opt=False, units="kn", val=EAS_limit)

        # Add timeseries outputs
        phase.add_timeseries_output(
            Dynamic.Mission.MACH, output_name=Dynamic.Mission.MACH, units="unitless")
        phase.add_timeseries_output("EAS", output_name="EAS", units="kn")
        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY, output_name=Dynamic.Mission.VELOCITY, units="kn"
        )
        phase.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                    output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE, units="deg")
        phase.add_timeseries_output("alpha", output_name="alpha", units="deg")
        phase.add_timeseries_output("theta", output_name="theta", units="deg")
        phase.add_timeseries_output("aero.CL", output_name="CL", units="unitless")
        phase.add_timeseries_output(
            Dynamic.Mission.THRUST_TOTAL, output_name=Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("aero.CD", output_name="CD", units="unitless")

        return phase

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'input_speed_type': self.user_options.get_val('input_speed_type'),
            'mach_cruise': self.user_options.get_val('mach_cruise'),
            'EAS_limit': self.user_options.get_val('EAS_limit', 'kn'),
        }


# Adding metadata for the DescentPhase
DescentPhase._add_meta_data(
    'analytic', val=False, desc='this is an analytic phase (no states).')
DescentPhase._add_meta_data(
    'reserve', val=False, desc='this phase is part of the reserve mission.')
DescentPhase._add_meta_data(
    'target_distance', val={}, desc='the amount of distance traveled in this phase added as a constraint')
DescentPhase._add_meta_data(
    'target_duration', val={}, desc='the amount of time taken by this phase added as a constraint')
DescentPhase._add_meta_data('fix_initial', val=False)
DescentPhase._add_meta_data('input_initial', val=False)
DescentPhase._add_meta_data('EAS_limit', val=0, units='kn')
DescentPhase._add_meta_data('mach_cruise', val=0)
DescentPhase._add_meta_data('input_speed_type', val=SpeedType.MACH)
DescentPhase._add_meta_data('final_altitude', val=0, units='ft')
DescentPhase._add_meta_data('duration_bounds', val=(0, 0), units='s')
DescentPhase._add_meta_data('duration_ref', val=1, units='s')
DescentPhase._add_meta_data('alt_lower', val=0, units='ft')
DescentPhase._add_meta_data('alt_upper', val=0, units='ft')
DescentPhase._add_meta_data('alt_ref', val=1, units='ft')
DescentPhase._add_meta_data('alt_ref0', val=0, units='ft')
DescentPhase._add_meta_data('alt_defect_ref', val=None, units='ft')
DescentPhase._add_meta_data('alt_constraint_ref', val=None, units='ft')
DescentPhase._add_meta_data('mass_lower', val=0, units='lbm')
DescentPhase._add_meta_data('mass_upper', val=0, units='lbm')
DescentPhase._add_meta_data('mass_ref', val=1, units='lbm')
DescentPhase._add_meta_data('mass_ref0', val=0, units='lbm')
DescentPhase._add_meta_data('mass_defect_ref', val=None, units='lbm')
DescentPhase._add_meta_data('distance_lower', val=0, units='NM')
DescentPhase._add_meta_data('distance_upper', val=0, units='NM')
DescentPhase._add_meta_data('distance_ref', val=1, units='NM')
DescentPhase._add_meta_data('distance_ref0', val=0, units='NM')
DescentPhase._add_meta_data('distance_defect_ref', val=None, units='NM')
DescentPhase._add_meta_data('num_segments', val=None, units='unitless')
DescentPhase._add_meta_data('order', val=None, units='unitless')

# Adding initial guess metadata
DescentPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for time options')
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for altitude state')
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass state')
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for distance state')
DescentPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
