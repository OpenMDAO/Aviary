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
from aviary.variable_info.enums import LegacyCode, Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.mission.utils import process_guess_var


class HeightEnergyProblemConfigurator(ProblemConfiguratorBase):
    """
    A Height-Energy specific builder that customizes AviaryProblem() for use with
    height energy phases.
    """

    def initial_guesses(self, aviary_group):
        """
        Set any initial guesses for variables in the aviary problem.

        This is called at the end of AivaryProblem.load_inputs.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        # TODO: This should probably be moved to the set_initial_guesses() method in AviaryProblem class
        # Defines how the problem should build it's initial guesses for load_inputs()
        # this modifies mass_method, initialization_guesses, and aviary_values

        aviary_inputs = aviary_group.aviary_inputs

        if aviary_group.engine_builders is None:
            aviary_group.engine_builders = [build_engine_deck(aviary_inputs)]

        aviary_group.initialization_guesses = initialization_guessing(
            aviary_inputs, aviary_group.initialization_guesses, aviary_group.engine_builders
        )

        # Deal with missing defaults in phase info:
        aviary_group.pre_mission_info.setdefault('include_takeoff', True)
        aviary_group.pre_mission_info.setdefault('external_subsystems', [])

        aviary_group.post_mission_info.setdefault('include_landing', True)
        aviary_group.post_mission_info.setdefault('external_subsystems', [])

        # Commonly referenced values
        aviary_inputs.set_val(
            Mission.Summary.GROSS_MASS,
            val=aviary_group.initialization_guesses['actual_takeoff_mass'],
            units='lbm',
        )

        if 'target_range' in aviary_group.post_mission_info:
            aviary_inputs.set_val(
                Mission.Summary.RANGE,
                wrapped_convert_units(aviary_group.post_mission_info['target_range'], 'NM'),
                units='NM',
            )
            aviary_group.require_range_residual = True
            aviary_group.target_range = wrapped_convert_units(
                aviary_group.post_mission_info['target_range'], 'NM'
            )
        else:
            aviary_group.require_range_residual = False
            # still instantiate target_range because it is used for default guesses
            # for phase comps
            aviary_group.target_range = aviary_inputs.get_val(Mission.Design.RANGE, units='NM')

    def get_default_phase_info(self, aviary_group):
        """
        Return a default phase_info for this type or problem.

        The default phase_info is used in the level 1 and 2 interfaces when no
        phase_info is specified.

        This is called during load_inputs.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.

        Returns
        -------
        AviaryValues
            General default phase_info.
        """
        from aviary.models.missions.height_energy_default import phase_info

        return phase_info

    def get_code_origin(self, aviary_group):
        """
        Return the legacy of this problem configurator.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.

        Returns
        -------
        LegacyCode
            Code origin enum.
        """
        return LegacyCode.FLOPS

    def add_takeoff_systems(self, aviary_group):
        """
        Adds takeoff systems to the model in aviary_group.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        takeoff_options = Takeoff(
            airport_altitude=0.0,  # ft
            num_engines=aviary_group.aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES),
        )

        # Build and add takeoff subsystem
        takeoff = takeoff_options.build_phase(False)
        aviary_group.add_subsystem(
            'takeoff',
            takeoff,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'],
        )

    def get_phase_builder(self, aviary_group, phase_name, phase_options):
        """
        Return a phase_builder for the requested phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
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

    def set_phase_options(self, aviary_group, phase_name, phase_idx, phase, user_options, comm):
        """
        Set any necessary problem-related options on the phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        phase_name : str
            Name of the requested phase.
        phase_idx : int
            Phase position in aviary_group.phases. Can be used to identify first phase.
        phase : Phase
            Instantiated phase object.
        user_options : dict
            Subdictionary "user_options" from the phase_info.
        comm : MPI.Comm or <FakeComm>
            MPI Communicator from OpenMDAO problem.
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

        if (fix_initial or input_initial) and comm.size == 1:
            # Redundant on a fixed input (unless MPI); raises a warning if specified.
            extra_options = {}
        else:
            extra_options = {
                'initial_ref': initial_ref,
                'initial_bounds': initial_bounds,
            }

        if not fix_duration:
            extra_options['duration_bounds'] = duration_bounds
            extra_options['duration_ref'] = duration_ref

        phase.set_time_options(
            fix_initial=fix_initial,
            fix_duration=fix_duration,
            units=time_units,
            **extra_options,
        )

    def link_phases(self, aviary_group, phases, connect_directly=True):
        """
        Apply any additional phase linking.

        Note that some phase variables are handled in the AviaryProblem. Only
        problem-specific ones need to be linked here.

        This is called from AviaryProblem.link_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        phases : Phase
            Phases to be linked.
        connect_directly : bool
            When True, then connected=True. This allows the connections to be
            handled by constraints if `phases` is a parallel group under MPI.
        """
        # connect regular_phases with each other if you are optimizing alt or mach
        self.link_phases_helper_with_options(
            aviary_group,
            aviary_group.regular_phases,
            'altitude_optimize',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        self.link_phases_helper_with_options(
            aviary_group, aviary_group.regular_phases, 'mach_optimize', Dynamic.Atmosphere.MACH
        )

        # connect reserve phases with each other if you are optimizing alt or mach
        self.link_phases_helper_with_options(
            aviary_group,
            aviary_group.reserve_phases,
            'altitude_optimize',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        self.link_phases_helper_with_options(
            aviary_group, aviary_group.reserve_phases, 'mach_optimize', Dynamic.Atmosphere.MACH
        )

        # connect mass and distance between all phases regardless of reserve /
        # non-reserve status
        aviary_group.traj.link_phases(
            phases, ['time'], ref=None if connect_directly else 1e3, connected=connect_directly
        )
        aviary_group.traj.link_phases(
            phases,
            [Dynamic.Vehicle.MASS],
            ref=None if connect_directly else 1e6,
            connected=connect_directly,
        )
        aviary_group.traj.link_phases(
            phases,
            [Dynamic.Mission.DISTANCE],
            ref=None if connect_directly else 1e3,
            connected=connect_directly,
        )

        # Under MPI, the states aren't directly connected.
        if not connect_directly:
            for phase_name in phases[1:]:
                phase = aviary_group.traj._phases[phase_name]
                phase.set_state_options(Dynamic.Vehicle.MASS, input_initial=False)
                phase.set_state_options(Dynamic.Mission.DISTANCE, input_initial=False)

        phase = aviary_group.traj._phases[phases[0]]

        # Currently expects Distance to be an input.
        phase.set_state_options(Dynamic.Mission.DISTANCE, input_initial=True)

        if aviary_group.pre_mission_info['include_takeoff']:
            # Allow these to connect to outputs in the pre-mission takeoff system.
            phase.set_state_options(Dynamic.Vehicle.MASS, input_initial=True)

    def check_trajectory(self, aviary_group):
        """
        Checks the phase_info user options for any inconsistency.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        phase_info = aviary_group.mission_info
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

        if len(msg) > 0 and aviary_group.verbosity > Verbosity.QUIET:
            print('\nThe following issues were detected in your phase_info options.')
            print(msg, '\n')

    def add_post_mission_systems(self, aviary_group):
        """
        Add any post mission systems.

        These may include any post-mission take off and landing systems.

        This is called from AviaryProblem.add_post_mission_systems

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        if aviary_group.pre_mission_info['include_takeoff']:
            self._add_post_mission_takeoff_systems(aviary_group)
        else:
            first_flight_phase_name = list(aviary_group.mission_info.keys())[0]

            # Since we don't have the takeoff subsystem, we need to use the gross mass as the
            # source for the mass at the beginning of the first flight phase. It turns out to be
            # more robust to use a constraint rather than connecting it directly.
            first_flight_phase = aviary_group.traj._phases[first_flight_phase_name]
            first_flight_phase.set_state_options(
                Dynamic.Vehicle.MASS, fix_initial=False, input_initial=False
            )

            # connect summary mass to the initial guess of mass in the first phase
            eq = aviary_group.add_subsystem(
                f'link_{first_flight_phase_name}_mass',
                om.EQConstraintComp(),
                promotes_inputs=[('rhs:mass', Mission.Summary.GROSS_MASS)],
            )

            # TODO: replace hard_coded ref for this constraint.
            eq.add_eq_output(
                'mass', eq_units='lbm', normalize=False, ref=100000.0, add_constraint=True
            )

            aviary_group.connect(
                f'traj.{first_flight_phase_name}.states:mass',
                f'link_{first_flight_phase_name}_mass.lhs:mass',
                src_indices=[0],
                flat_src_indices=True,
            )

        if aviary_group.post_mission_info['include_landing']:
            self._add_landing_systems(aviary_group)

        aviary_group.add_subsystem(
            'range_constraint',
            om.ExecComp(
                'range_resid = target_range - actual_range',
                target_range={'val': aviary_group.target_range, 'units': 'NM'},
                actual_range={'val': aviary_group.target_range, 'units': 'NM'},
                range_resid={'val': 30, 'units': 'NM'},
            ),
            promotes_inputs=[
                ('actual_range', Mission.Summary.RANGE),
                'target_range',
            ],
            promotes_outputs=[('range_resid', Mission.Constraints.RANGE_RESIDUAL)],
        )

        # TODO: replace hard_coded ref for this constraint.
        aviary_group.post_mission.add_constraint(
            Mission.Constraints.MASS_RESIDUAL, equals=0.0, ref=1.0e5
        )

    def _add_post_mission_takeoff_systems(self, aviary_group):
        """
        Adds residual and constraint components for the mach and alpha connections from takeoff
        to the first flight phase.
        """
        first_flight_phase_name = list(aviary_group.mission_info.keys())[0]
        phase_options = aviary_group.mission_info[first_flight_phase_name]['user_options']

        aviary_group.connect(
            Mission.Takeoff.FINAL_MASS, f'traj.{first_flight_phase_name}.initial_states:mass'
        )
        aviary_group.connect(
            Mission.Takeoff.GROUND_DISTANCE,
            f'traj.{first_flight_phase_name}.initial_states:distance',
        )

        if phase_options.get('mach_optimize', False):
            # Create an ExecComp to compute the difference in mach
            mach_diff_comp = om.ExecComp(
                'mach_resid_for_connecting_takeoff = final_mach - initial_mach'
            )
            aviary_group.add_subsystem('mach_diff_comp', mach_diff_comp)

            # Connect the inputs to the mach difference component
            aviary_group.connect(Mission.Takeoff.FINAL_MACH, 'mach_diff_comp.final_mach')
            aviary_group.connect(
                f'traj.{first_flight_phase_name}.control_values:mach',
                'mach_diff_comp.initial_mach',
                src_indices=[0],
            )

            # Add constraint for mach difference
            aviary_group.add_constraint(
                'mach_diff_comp.mach_resid_for_connecting_takeoff', equals=0.0
            )

        if phase_options.get('altitude_optimize', False):
            # Similar steps for altitude difference
            alt_diff_comp = om.ExecComp(
                'altitude_resid_for_connecting_takeoff = final_altitude - initial_altitude',
                units='ft',
            )
            aviary_group.add_subsystem('alt_diff_comp', alt_diff_comp)

            aviary_group.connect(Mission.Takeoff.FINAL_ALTITUDE, 'alt_diff_comp.final_altitude')
            aviary_group.connect(
                f'traj.{first_flight_phase_name}.control_values:altitude',
                'alt_diff_comp.initial_altitude',
                src_indices=[0],
            )

            # Add constraint for altitude difference
            aviary_group.add_constraint(
                'alt_diff_comp.altitude_resid_for_connecting_takeoff', equals=0.0
            )

    def _add_landing_systems(self, aviary_group):
        landing_options = Landing(
            ref_wing_area=aviary_group.aviary_inputs.get_val(Aircraft.Wing.AREA, units='ft**2'),
            Cl_max_ldg=aviary_group.aviary_inputs.get_val(
                Mission.Landing.LIFT_COEFFICIENT_MAX
            ),  # no units
        )

        landing = landing_options.build_phase(False)

        aviary_group.add_subsystem(
            'landing',
            landing,
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'],
        )

        last_regular_phase = aviary_group.regular_phases[-1]
        aviary_group.connect(
            f'traj.{last_regular_phase}.states:mass',
            Mission.Landing.TOUCHDOWN_MASS,
            src_indices=[-1],
        )
        aviary_group.connect(
            f'traj.{last_regular_phase}.control_values:altitude',
            Mission.Landing.INITIAL_ALTITUDE,
            src_indices=[0],
        )

    def set_phase_initial_guesses(
        self, aviary_group, phase_name, phase, guesses, target_prob, parent_prefix
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

        options = aviary_group.mission_info[phase_name]['user_options']

        if options['throttle_enforcement'] == 'control':
            control_keys.append('throttle')

        # Let's preserve the original user-specified initial conditions.
        guess_dict = deepcopy(guesses)

        if 'mass' not in guess_dict:
            mass_guess = aviary_group.aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm')

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

        if 'time' not in guess_dict:
            # if time not in initial guesses, use the midpoints of any declared bounds.
            initial = wrapped_convert_units(options['time_initial'], 's')
            if initial is None:
                initial_bounds = wrapped_convert_units(options['time_initial_bounds'], 's')
                initial = np.mean(initial_bounds[0])
            duration = wrapped_convert_units(options['time_duration'], 's')
            if duration is None:
                duration_bounds = wrapped_convert_units(options['time_duration_bounds'], 's')
                duration = np.mean(duration_bounds[0])

            guess_dict['time'] = ([initial, duration], 's')

        for guess_key, guess_data in guess_dict.items():
            val, units = guess_data

            if 'time' == guess_key:
                # Set initial guess for time variables
                # Seems to be an openmdao bug. Switch to this when fixed.
                # phase.set_time_val(initial=val[0], duration=val[1], units=units)

                if val[0] is not None:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.t_initial', val[0], units=units
                    )
                if val[1] is not None:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.t_duration', val[1], units=units
                    )

            elif guess_key in control_keys:
                # Set initial guess for control variables
                # Seems to be an openmdao bug. Switch to this when fixed.
                # phase.set_control_val(guess_key, val, units=units)

                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.controls:{guess_key}',
                    process_guess_var(val, guess_key, phase),
                    units=units,
                )

            elif guess_key in state_keys:
                # Set initial guess for state variables
                # Seems to be an openmdao bug. Switch to this when fixed.
                # phase.set_state_val(guess_key, val, units=units)

                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.states:{guess_key}',
                    process_guess_var(val, guess_key, phase),
                    units=units,
                )

            elif guess_key in prob_keys:
                target_prob.set_val(parent_prefix + guess_key, val, units=units)

            elif ':' in guess_key:
                # These may come from external subsystems.
                target_prob.set_val(
                    parent_prefix + f'traj.{phase_name}.{guess_key}',
                    process_guess_var(val, guess_key, phase),
                    units=units,
                )
            else:
                # raise error if the guess key is not recognized
                raise ValueError(
                    f'Initial guess key {guess_key} in {phase_name} is not recognized.'
                )
