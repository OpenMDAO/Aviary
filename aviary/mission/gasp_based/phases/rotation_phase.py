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

        defaults = {
            'mass_ref': 100_000.0,
            'mass_bounds': (0.0, 190_000.0),
        }
        self.add_state_options('mass', units='lbm', defaults=defaults)

        # NOTE: All GASP phases before accel are in 'ft'.
        defaults = {
            'distance_ref': 3000.0,
            'distance_bounds': (0.0, 10.0e3),
        }
        self.add_state_options('distance', units='ft', defaults=defaults)

        defaults = {
            'velocity_ref': 100.0,
            'velocity_bounds': (0.0, 1000.0),
        }
        self.add_state_options('velocity', units='kn', defaults=defaults)

        defaults = {
            'angle_of_attack_defect_ref': 0.01,
            'angle_of_attack_bounds': (0.0, 25 * np.pi / 180),
        }
        self.add_state_options('angle_of_attack', units='rad', defaults=defaults)

        defaults = {
            'time_duration_bounds': (1.0, 100.0),
        }
        self.add_time_options(units='s', defaults=defaults)

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

        # The options below have not yet been revamped.

        self.declare(
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
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
        normal_ref = user_options.get_val('normal_ref', units='lbf')
        normal_ref0 = user_options.get_val('normal_ref0', units='lbf')

        # Add states
        self.add_state('angle_of_attack', Dynamic.Vehicle.ANGLE_OF_ATTACK, 'angle_of_attack_rate')
        self.add_state('velocity', Dynamic.Mission.VELOCITY, Dynamic.Mission.VELOCITY_RATE)
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        )
        self.add_state('distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE)

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
