import numpy as np

from aviary.mission.gasp_based.ode.rotation_ode import RotationODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class RotationPhaseOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
        )

        self.declare(
            'reserve',
            types=bool,
            default=False,
            desc='Designate this phase as a reserve phase and contributes its fuel burn '
            'towards the reserve mission fuel requirements. Reserve phases should be '
            'be placed after all non-reserve phases in the phase_info.',
        )

        self.declare(
            name='target_distance',
            default=None,
            units='m',
            desc='The total distance traveled by the aircraft from takeoff to landing '
            'for the primary mission, not including reserve missions. This value must '
            'be positive.',
        )

        self.declare(
            'time_duration',
            default=None,
            units='s',
            desc='The amount of time taken by this phase added as a constraint.',
        )

        self.declare(
            name='fix_initial',
            types=bool,
            default=False,
            desc='Fixes the initial states (mass, distance) and does not allow them to '
            'change during the optimization.',
        )

        self.declare(
            name='time_duration_bounds',
            default=(1.0, 100.0),
            units='s',
            desc='Lower and upper bounds on the phase duration, in the form of a nested tuple: '
            'i.e. ((20, 36), "min") This constrains the duration to be between 20 and 36 min.',
        )

        self.declare(
            name='time_duration_ref', default=1.0, units='s', desc='Scale factor ref for duration.'
        )

        self.declare(
            name='angle_lower', types=tuple, default=0.0, units='rad', desc='Lower bound for angle.'
        )

        self.declare(
            name='angle_upper', default=25 * np.pi / 180, units='rad', desc='Upper bound for angle.'
        )

        self.declare(name='angle_ref', default=1.0, units='rad', desc='Scale factor ref for angle.')

        self.declare(
            name='angle_ref0', default=0.0, units='rad', desc='Scale factor ref0 for angle.'
        )

        self.declare(
            name='angle_defect_ref',
            default=0.01,
            units='rad',
            desc='Scale factor ref for angle defect.',
        )

        self.declare(
            name='velocity_lower', default=0.0, units='kn', desc='Lower bound for velocity.'
        )

        self.declare(
            name='velocity_upper', default=1000.0, units='kn', desc='Upper bound for velocity.'
        )

        self.declare(
            name='velocity_ref', default=100.0, units='kn', desc='Scale factor ref for velocity.'
        )

        self.declare(
            name='velocity_ref0', default=0.0, units='kn', desc='Scale factor ref0 for velocity.'
        )

        self.declare(
            name='velocity_defect_ref',
            default=None,
            units='kn',
            desc='Scale factor ref for velocity defect.',
        )

        self.declare(
            name='mass_lower', types=tuple, default=0.0, units='lbm', desc='Lower bound for mass.'
        )

        self.declare(
            name='mass_upper', default=190_000.0, units='lbm', desc='Upper bound for mass.'
        )

        self.declare(
            name='mass_ref', default=100_000.0, units='lbm', desc='Scale factor ref for mass.'
        )

        self.declare(name='mass_ref0', default=0.0, units='lbm', desc='Scale factor ref0 for mass.')

        self.declare(
            name='mass_defect_ref',
            default=None,
            units='lbm',
            desc='Scale factor ref for mass defect.',
        )

        self.declare(
            name='distance_lower', default=0.0, units='ft', desc='Lower bound for distance.'
        )

        self.declare(
            name='distance_upper', default=10.0e3, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='distance_ref', default=3000.0, units='ft', desc='Scale factor ref for distance.'
        )

        self.declare(
            name='distance_ref0', default=0.0, units='ft', desc='Scale factor ref0 for distance.'
        )

        self.declare(
            name='distance_defect_ref',
            default=3000.0,
            units='ft',
            desc='Scale factor ref for distance defect.',
        )

        self.declare(
            name='normal_ref',
            default=1.0,
            units='lbf',
            desc='Scale factor ref for the normal force constraint.',
        )

        self.declare(
            name='normal_ref0',
            default=0.0,
            units='lbf',
            desc='Scale factor ref0 for the normal force constraint.',
        )

        self.declare(
            name='t_init_gear', default=100.0, units='s', desc='Time where landing gear is lifted.'
        )

        self.declare(
            name='t_init_flaps', default=100.0, units='s', desc='Time where flaps are retracted.'
        )

        self.declare(
            name='num_segments',
            types=int,
            default=None,
            desc='The number of segments in transcription creation in Dymos. ',
        )

        self.declare(
            name='order',
            types=int,
            default=None,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos.',
        )


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
    default_options_class = RotationPhaseOptions

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
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            fix_initial=True,
            fix_final=False,
            lower=angle_lower,
            upper=angle_upper,
            units='rad',
            rate_source='angle_of_attack_rate',
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
            units='ft',
            rate_source='distance_rate',
            ref=distance_ref,
            ref0=distance_ref0,
            defect_ref=distance_defect_ref,
        )

        # Add parameters
        phase.add_parameter('t_init_gear', units='s', static_target=True, opt=False, val=100)
        phase.add_parameter('t_init_flaps', units='s', static_target=True, opt=False, val=100)

        # Add boundary constraints
        phase.add_boundary_constraint(
            'normal_force',
            loc='final',
            equals=0,
            units='lbf',
            ref=normal_ref,
            ref0=normal_ref0,
        )

        # Add timeseries outputs
        phase.add_timeseries_output(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, units='lbf')
        phase.add_timeseries_output('normal_force')
        phase.add_timeseries_output(Dynamic.Atmosphere.MACH)
        phase.add_timeseries_output('EAS', units='kn')
        phase.add_timeseries_output(Dynamic.Vehicle.LIFT)
        phase.add_timeseries_output('CL')
        phase.add_timeseries_output('CD')
        phase.add_timeseries_output('fuselage_pitch', output_name='theta', units='deg')

        return phase


# Adding initial guess metadata
RotationPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(), desc='initial guess for time options'
)
RotationPhase._add_initial_guess_meta_data(
    InitialGuessState('angle_of_attack'), desc='initial guess for angle of attack state'
)
RotationPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'), desc='initial guess for true airspeed state'
)
RotationPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass state'
)
RotationPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for distance state'
)
RotationPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'), desc='initial guess for throttle'
)
