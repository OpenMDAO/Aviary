from copy import deepcopy

import numpy as np
import openmdao.api as om
from dymos.transcriptions.transcription_base import TranscriptionBase

from aviary.mission.flight_phase_builder import FlightPhaseOptions
from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.mission.flops_based.phases.energy_phase import EnergyPhase
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.problem_configurator import ProblemConfiguratorBase
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.process_input_decks import initialization_guessing
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import AnalysisScheme, LegacyCode, Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class HeightEnergyProblemConfigurator(ProblemConfiguratorBase):
    """
    A Height-Energy specific builder that customizes AviaryProblem() for use with
    height energy phases.
    """

    def initial_guesses(self, prob):
        """
        Set any initial guesses for variables in the aviary problem.

        This is called at the end of AivaryProblem.load_inputs.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        # TODO: This should probably be moved to the set_initial_guesses() method in AviaryProblem class
        # Defines how the problem should build it's initial guesses for load_inputs()
        # this modifies mass_method, initialization_guesses, and aviary_values

        aviary_inputs = prob.aviary_inputs

        if prob.engine_builders is None:
            prob.engine_builders = [build_engine_deck(aviary_inputs)]

        prob.initialization_guesses = initialization_guessing(
            aviary_inputs, prob.initialization_guesses, prob.engine_builders
        )

        # Deal with missing defaults in phase info:
        prob.pre_mission_info.setdefault('include_takeoff', True)
        prob.pre_mission_info.setdefault('external_subsystems', [])

        prob.post_mission_info.setdefault('include_landing', True)
        prob.post_mission_info.setdefault('external_subsystems', [])

        # Commonly referenced values
        aviary_inputs.set_val(
            Mission.Summary.GROSS_MASS,
            val=prob.initialization_guesses['actual_takeoff_mass'],
            units='lbm',
        )

        if 'target_range' in prob.post_mission_info:
            aviary_inputs.set_val(
                Mission.Summary.RANGE,
                wrapped_convert_units(prob.post_mission_info['target_range'], 'NM'),
                units='NM',
            )
            prob.require_range_residual = True
            prob.target_range = wrapped_convert_units(prob.post_mission_info['target_range'], 'NM')
        else:
            prob.require_range_residual = False
            # still instantiate target_range because it is used for default guesses
            # for phase comps
            prob.target_range = aviary_inputs.get_val(Mission.Design.RANGE, units='NM')

    def get_default_phase_info(self, prob):
        """
        Return a default phase_info for this type or problem.

        The default phase_info is used in the level 1 and 2 interfaces when no
        phase_info is specified.

        This is called during load_inputs.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.

        Returns
        -------
        AviaryValues
            General default phase_info.
        """
        if prob.analysis_scheme is AnalysisScheme.COLLOCATION:
            from aviary.interface.default_phase_info.height_energy import phase_info
        else:
            raise RuntimeError('Height Energy requires that a phase_info is specified.')

        return phase_info

    def get_code_origin(self, prob):
        """
        Return the legacy of this problem configurator.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.

        Returns
        -------
        LegacyCode
            Code origin enum.
        """
        return LegacyCode.FLOPS

    def add_takeoff_systems(self, prob):
        """
        Adds takeoff systems to the model in prob.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        takeoff_options = Takeoff(
            airport_altitude=0.0,  # ft
            num_engines=prob.aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES),
        )

        # Build and add takeoff subsystem
        takeoff = takeoff_options.build_phase(False)
        prob.model.add_subsystem(
            'takeoff',
            takeoff,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'],
        )

    def get_phase_builder(self, prob, phase_name, phase_options):
        """
        Return a phase_builder for the requested phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        phase_name : str
            Name of the requested phase.
        phase_options : dict
            Phase options for the requested phase.

        Returns
        -------
        PhaseBuilderBase
            Phase builder for requested phase.
        """
        if 'phase_builder' in phase_options:
            phase_builder = phase_options['phase_builder']
            if not issubclass(phase_builder, PhaseBuilderBase):
                raise TypeError(
                    'phase_builder for the phase called '
                    '{phase_name} must be a PhaseBuilderBase object.'
                )
        else:
            phase_builder = EnergyPhase

        return phase_builder

    def set_phase_options(self, prob, phase_name, phase_idx, phase, user_options):
        """
        Set any necessary problem-related options on the phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        phase_name : str
            Name of the requested phase.
        phase_idx : int
            Phase position in prob.phases. Can be used to identify first phase.
        phase : Phase
            Instantiated phase object.
        user_options : dict
            Subdictionary "user_options" from the phase_info.
        """
        time_units = 's'
        initial = wrapped_convert_units(user_options['time_initial'], time_units)
        duration = wrapped_convert_units(user_options['time_duration'], time_units)
        initial_bounds = wrapped_convert_units(user_options['time_initial_bounds'], time_units)
        duration_bounds = wrapped_convert_units(user_options['time_duration_bounds'], time_units)
        initial_ref = wrapped_convert_units(user_options['time_initial_ref'], time_units)
        duration_ref = wrapped_convert_units(user_options['time_duration_ref'], time_units)

        fix_initial = initial is not None
        fix_duration = duration is not None

        # All follow-on phases.
        input_initial = phase_idx > 0

        # Figure out resonable refs if they aren't given.
        if initial_ref == 1.0:
            if initial_bounds[1]:
                initial_ref = initial_bounds[1]
            else:
                # TODO: Why were we using this value?
                initial_ref = 600.0

        if duration_ref == 1.0:
            # We have been using the average of the bounds if they exist.
            lower, upper = duration_bounds
            if lower is not None or upper is not None:
                if lower is None:
                    lower = 0.0
                if upper is None:
                    upper = 0.0
                duration_ref = 0.5 * (lower + upper)

        if (fix_initial or input_initial) and prob.comm.size == 1:
            # Redundant on a fixed input (unless MPI); raises a warning if specified.
            initial_options = {}
        else:
            initial_options = {
                'initial_ref': initial_ref,
                'initial_bounds': initial_bounds,
            }

        phase.set_time_options(
            fix_initial=fix_initial,
            fix_duration=fix_duration,
            units=time_units,
            duration_bounds=duration_bounds,
            duration_ref=duration_ref,
            **initial_options,
        )

    def link_phases(self, prob, phases, connect_directly=True):
        """
        Apply any additional phase linking.

        Note that some phase variables are handled in the AviaryProblem. Only
        problem-specific ones need to be linked here.

        This is called from AviaryProblem.link_phases

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        phases : Phase
            Phases to be linked.
        connect_directly : bool
            When True, then connected=True. This allows the connections to be
            handled by constraints if `phases` is a parallel group under MPI.
        """
        # connect regular_phases with each other if you are optimizing alt or mach
        prob._link_phases_helper_with_options(
            prob.regular_phases,
            'altitude_optimize',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        prob._link_phases_helper_with_options(
            prob.regular_phases, 'mach_optimize', Dynamic.Atmosphere.MACH
        )

        # connect reserve phases with each other if you are optimizing alt or mach
        prob._link_phases_helper_with_options(
            prob.reserve_phases,
            'altitude_optimize',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        prob._link_phases_helper_with_options(
            prob.reserve_phases, 'mach_optimize', Dynamic.Atmosphere.MACH
        )

        # connect mass and distance between all phases regardless of reserve /
        # non-reserve status
        prob.traj.link_phases(
            phases, ['time'], ref=None if connect_directly else 1e3, connected=connect_directly
        )
        prob.traj.link_phases(
            phases,
            [Dynamic.Vehicle.MASS],
            ref=None if connect_directly else 1e6,
            connected=connect_directly,
        )
        prob.traj.link_phases(
            phases,
            [Dynamic.Mission.DISTANCE],
            ref=None if connect_directly else 1e3,
            connected=connect_directly,
        )

        # Under MPI, the states aren't directly connected.
        if not connect_directly:
            for phase_name in phases[1:]:
                phase = prob.traj._phases[phase_name]
                phase.set_state_options(Dynamic.Vehicle.MASS, input_initial=False)
                phase.set_state_options(Dynamic.Mission.DISTANCE, input_initial=False)

        prob.model.connect(
            f'traj.{prob.regular_phases[-1]}.timeseries.distance',
            Mission.Summary.RANGE,
            src_indices=[-1],
            flat_src_indices=True,
        )

    def check_trajectory(self, prob):
        """
        Checks the phase_info user options for any inconsistency.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        phase_info = prob.phase_info
        all_phases = [name for name in phase_info]

        stems = [
            Dynamic.Vehicle.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Atmosphere.MACH,
            Dynamic.Mission.ALTITUDE,
        ]

        msg = ''
        for j in range(1, len(all_phases)):
            left_name = all_phases[j - 1]
            right_name = all_phases[j]
            left = phase_info[left_name]['user_options']
            right = phase_info[right_name]['user_options']

            for stem in stems:
                final = left[f'{stem}_final']
                initial = right[f'{stem}_initial']

                if initial[0] is None or final[0] is None:
                    continue

                if initial != final:
                    msg += '  Constraint mismatch across phase boundary:\n'
                    msg += f'    {left_name} {stem}_final: {final}\n'
                    msg += f'    {right_name} {stem}_initial: {initial}\n'

        if len(msg) > 0 and prob.verbosity > Verbosity.QUIET:
            print('\nThe following issues were detected in your phase_info options.')
            print(msg, '\n')

    def add_post_mission_systems(self, prob):
        """
        Add any post mission systems.

        These may include any post-mission take off and landing systems.

        This is called from AviaryProblem.add_post_mission_systems

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        if prob.pre_mission_info['include_takeoff']:
            self._add_post_mission_takeoff_systems(prob)
        else:
            first_flight_phase_name = list(prob.phase_info.keys())[0]

            # Since we don't have the takeoff subsystem, we need to use the gross mass as the
            # source for the mass at the beginning of the first flight phase. It turns out to be
            # more robust to use a constraint rather than connecting it directly.
            first_flight_phase = prob.traj._phases[first_flight_phase_name]
            first_flight_phase.set_state_options(
                Dynamic.Vehicle.MASS, fix_initial=False, input_initial=False
            )

            # connect summary mass to the initial guess of mass in the first phase
            eq = prob.model.add_subsystem(
                f'link_{first_flight_phase_name}_mass',
                om.EQConstraintComp(),
                promotes_inputs=[('rhs:mass', Mission.Summary.GROSS_MASS)],
            )

            # TODO: replace hard_coded ref for this constraint.
            eq.add_eq_output(
                'mass', eq_units='lbm', normalize=False, ref=100000.0, add_constraint=True
            )

            prob.model.connect(
                f'traj.{first_flight_phase_name}.states:mass',
                f'link_{first_flight_phase_name}_mass.lhs:mass',
                src_indices=[0],
                flat_src_indices=True,
            )

        if prob.post_mission_info['include_landing']:
            self._add_landing_systems(prob)

        prob.model.add_subsystem(
            'range_constraint',
            om.ExecComp(
                'range_resid = target_range - actual_range',
                target_range={'val': prob.target_range, 'units': 'NM'},
                actual_range={'val': prob.target_range, 'units': 'NM'},
                range_resid={'val': 30, 'units': 'NM'},
            ),
            promotes_inputs=[
                ('actual_range', Mission.Summary.RANGE),
                'target_range',
            ],
            promotes_outputs=[('range_resid', Mission.Constraints.RANGE_RESIDUAL)],
        )

        # TODO: replace hard_coded ref for this constraint.
        prob.post_mission.add_constraint(Mission.Constraints.MASS_RESIDUAL, equals=0.0, ref=1.0e5)

    def _add_post_mission_takeoff_systems(self, prob):
        """
        Adds residual and constraint components for the mach and alpha connections from takeoff
        to the first flight phase.
        """
        first_flight_phase_name = list(prob.phase_info.keys())[0]
        phase_options = prob.phase_info[first_flight_phase_name]['user_options']

        prob.model.connect(
            Mission.Takeoff.FINAL_MASS, f'traj.{first_flight_phase_name}.initial_states:mass'
        )
        prob.model.connect(
            Mission.Takeoff.GROUND_DISTANCE,
            f'traj.{first_flight_phase_name}.initial_states:distance',
        )

        if phase_options.get('mach_optimize', False):
            # Create an ExecComp to compute the difference in mach
            mach_diff_comp = om.ExecComp(
                'mach_resid_for_connecting_takeoff = final_mach - initial_mach'
            )
            prob.model.add_subsystem('mach_diff_comp', mach_diff_comp)

            # Connect the inputs to the mach difference component
            prob.model.connect(Mission.Takeoff.FINAL_MACH, 'mach_diff_comp.final_mach')
            prob.model.connect(
                f'traj.{first_flight_phase_name}.control_values:mach',
                'mach_diff_comp.initial_mach',
                src_indices=[0],
            )

            # Add constraint for mach difference
            prob.model.add_constraint(
                'mach_diff_comp.mach_resid_for_connecting_takeoff', equals=0.0
            )

        if phase_options.get('altitude_optimize', False):
            # Similar steps for altitude difference
            alt_diff_comp = om.ExecComp(
                'altitude_resid_for_connecting_takeoff = final_altitude - initial_altitude',
                units='ft',
            )
            prob.model.add_subsystem('alt_diff_comp', alt_diff_comp)

            prob.model.connect(Mission.Takeoff.FINAL_ALTITUDE, 'alt_diff_comp.final_altitude')
            prob.model.connect(
                f'traj.{first_flight_phase_name}.control_values:altitude',
                'alt_diff_comp.initial_altitude',
                src_indices=[0],
            )

            # Add constraint for altitude difference
            prob.model.add_constraint(
                'alt_diff_comp.altitude_resid_for_connecting_takeoff', equals=0.0
            )

    def _add_landing_systems(self, prob):
        landing_options = Landing(
            ref_wing_area=prob.aviary_inputs.get_val(Aircraft.Wing.AREA, units='ft**2'),
            Cl_max_ldg=prob.aviary_inputs.get_val(Mission.Landing.LIFT_COEFFICIENT_MAX),  # no units
        )

        landing = landing_options.build_phase(False)

        prob.model.add_subsystem(
            'landing',
            landing,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'],
        )

        last_regular_phase = prob.regular_phases[-1]
        prob.model.connect(
            f'traj.{last_regular_phase}.states:mass',
            Mission.Landing.TOUCHDOWN_MASS,
            src_indices=[-1],
        )
        prob.model.connect(
            f'traj.{last_regular_phase}.control_values:altitude',
            Mission.Landing.INITIAL_ALTITUDE,
            src_indices=[0],
        )

    def add_objective(self, prob):
        """
        Add any additional components related to objectives.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        pass

    def set_phase_initial_guesses(
        self, prob, phase_name, phase, guesses, target_prob, parent_prefix
    ):
        """
        Adds the initial guesses for each variable of a given phase to the problem.

        This method sets the initial guesses into the openmdao model for time, controls, states,
        and problem-specific variables for a given phase. If using the GASP model, it also handles
        some special cases that are not covered in the `phase_info` object. These include initial
        guesses for mass, time, and distance, which are determined based on the phase name and
        other mission-related variables.

        Parameters
        ----------
        phase_name : str
            The name of the phase for which the guesses are being added.
        phase : Phase
            The phase object for which the guesses are being added.
        guesses : dict
            A dictionary containing the initial guesses for the phase.
        target_prob : Problem
            Problem instance to apply the guesses.
        parent_prefix : str
            Location of this trajectory in the hierarchy.
        """
        control_keys = ['mach', 'altitude']
        state_keys = ['mass', Dynamic.Mission.DISTANCE]
        prob_keys = ['tau_gear', 'tau_flaps']

        options = prob.phase_info[phase_name]['user_options']

        # Let's preserve the original user-specified initial conditions.
        guess_dict = deepcopy(guesses)

        if 'mass' not in guess_dict:
            mass_guess = prob.aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm')

            guess_dict['mass'] = (mass_guess, 'lbm')

        if 'altitude' not in guess_dict:
            # Use values from fixed endpoints.
            altitude_initial = wrapped_convert_units(options['altitude_initial'], 'ft')
            altitude_final = wrapped_convert_units(options['altitude_final'], 'ft')

            if altitude_initial is None:
                # TODO: Pull from downstream phase.
                altitude_initial = wrapped_convert_units(options['altitude_bounds'], 'ft')[0]

            if altitude_final is None:
                # TODO: Pull from downstream phase.
                altitude_final = altitude_initial

            guess_dict['altitude'] = ([altitude_initial, altitude_final], 'ft')

        if 'mach' not in guess_dict:
            # Use values from fixed endpoints.
            mach_initial = wrapped_convert_units(options['mach_initial'], 'unitless')
            mach_final = wrapped_convert_units(options['mach_final'], 'unitless')

            if mach_initial is None:
                # TODO: Pull from downstream phase.
                mach_initial = wrapped_convert_units(options['mach_bounds'], 'unitless')[0]

            if mach_final is None:
                # TODO: Pull from downstream phase.
                mach_final = mach_initial

            guess_dict['mach'] = ([mach_initial, mach_final], 'unitless')

        if 'time' not in guess_dict and options['time_duration'][0] is None:
            # if time not in initial guesses, set it to the average of the
            # initial_bounds and the duration_bounds
            initial_bounds = wrapped_convert_units(options['time_initial_bounds'], 's')
            duration_bounds = wrapped_convert_units(options['time_duration_bounds'], 's')
            guess_dict['time'] = ([np.mean(initial_bounds[0]), np.mean(duration_bounds[0])], 's')

        for guess_key, guess_data in guess_dict.items():
            val, units = guess_data

            if 'time' == guess_key:
                # Set initial guess for time variables
                # Seems to be an openmdao bug. Switch to this when fixed.
                # phase.set_time_val(initial=val[0], duration=val[1], units=units)

                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.t_initial', val[0], units=units
                )
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.t_duration', val[1], units=units
                )

            elif guess_key in control_keys:
                # Set initial guess for control variables
                # Seems to be an openmdao bug. Switch to this when fixed.
                # phase.set_control_val(guess_key, val, units=units)

                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.controls:{guess_key}',
                    prob._process_guess_var(val, guess_key, phase),
                    units=units,
                )

            elif guess_key in state_keys:
                # Set initial guess for state variables
                # Seems to be an openmdao bug. Switch to this when fixed.
                # phase.set_state_val(guess_key, val, units=units)

                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.states:{guess_key}',
                    prob._process_guess_var(val, guess_key, phase),
                    units=units,
                )

            elif guess_key in prob_keys:
                target_prob.set_val(parent_prefix + guess_key, val, units=units)

            elif ':' in guess_key:
                # These may come from external subsystems.
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.{guess_key}',
                    prob._process_guess_var(val, guess_key, phase),
                    units=units,
                )
            else:
                # raise error if the guess key is not recognized
                raise ValueError(
                    f'Initial guess key {guess_key} in {phase_name} is not recognized.'
                )
