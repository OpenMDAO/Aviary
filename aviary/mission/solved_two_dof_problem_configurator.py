from aviary.mission.flops_based.phases.groundroll_phase import (
    GroundrollPhase as GroundrollPhaseVelocityIntegrated,
)
from aviary.mission.gasp_based.phases.twodof_phase import TwoDOFPhase
from aviary.mission.problem_configurator import ProblemConfiguratorBase
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Dynamic, Mission


class SolvedTwoDOFProblemConfigurator(ProblemConfiguratorBase):
    """The Solved 2DOF builder is used for detailed take-off and landing."""

    def initial_guesses(self, prob):
        """
        Set any initial guesses for variables in the aviary problem.

        This is called at the end of AivaryProblem.load_inputs.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        if prob.engine_builders is None:
            prob.engine_builders = [build_engine_deck(prob.aviary_inputs)]

        # This doesn't really have much value, but is needed for initializing
        # an objective-related component that still lives in level 2.
        prob.target_range = prob.aviary_inputs.get_val(Mission.Design.RANGE, units='NM')

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
        raise RuntimeError('Solved 2DOF requires that a phase_info is specified.')

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
        pass

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
        if (
            phase_options['user_options']['ground_roll']
            and phase_options['user_options']['fix_initial']
        ):
            phase_builder = GroundrollPhaseVelocityIntegrated
        else:
            phase_builder = TwoDOFPhase

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
        try:
            fix_initial = user_options.get_val('fix_initial')
        except KeyError:
            fix_initial = False

        try:
            fix_duration = user_options.get_val('fix_duration')
        except KeyError:
            fix_duration = False

        input_initial = False
        time_units = phase.time_options['units']

        # Make a good guess for a reasonable intitial time scaler.
        try:
            initial_bounds = user_options.get_val('initial_bounds', units=time_units)
        except KeyError:
            initial_bounds = (None, None)

        if initial_bounds[0] is not None and initial_bounds[1] != 0.0:
            # Upper bound is good for a ref.
            user_options.set_val('initial_ref', initial_bounds[1], units=time_units)
        else:
            user_options.set_val('initial_ref', 600.0, time_units)

        duration_bounds = user_options.get_val('duration_bounds', time_units)
        user_options.set_val(
            'duration_ref', (duration_bounds[0] + duration_bounds[1]) / 2.0, time_units
        )
        if phase_idx > 0:
            input_initial = True

        if fix_initial or input_initial:
            if prob.comm.size > 1:
                # Phases are disconnected to run in parallel, so initial ref is
                # valid.
                initial_ref = user_options.get_val('initial_ref', time_units)
            else:
                # Redundant on a fixed input; raises a warning if specified.
                initial_ref = None

            phase.set_time_options(
                fix_initial=fix_initial,
                fix_duration=fix_duration,
                units=time_units,
                duration_bounds=user_options.get_val('duration_bounds', time_units),
                duration_ref=user_options.get_val('duration_ref', time_units),
                initial_ref=initial_ref,
            )
        else:  # TODO: figure out how to handle this now that fix_initial is dict
            phase.set_time_options(
                fix_initial=fix_initial,
                fix_duration=fix_duration,
                units=time_units,
                duration_bounds=user_options.get_val('duration_bounds', time_units),
                duration_ref=user_options.get_val('duration_ref', time_units),
                initial_bounds=initial_bounds,
                initial_ref=user_options.get_val('initial_ref', time_units),
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
            'optimize_altitude',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        prob._link_phases_helper_with_options(
            prob.regular_phases, 'optimize_mach', Dynamic.Atmosphere.MACH
        )

        # connect reserve phases with each other if you are optimizing alt or mach
        prob._link_phases_helper_with_options(
            prob.reserve_phases,
            'optimize_altitude',
            Dynamic.Mission.ALTITUDE,
            ref=1.0e4,
        )
        prob._link_phases_helper_with_options(
            prob.reserve_phases, 'optimize_mach', Dynamic.Atmosphere.MACH
        )

        prob.traj.link_phases(phases, [Dynamic.Vehicle.MASS], connected=True)
        prob.traj.link_phases(
            phases, [Dynamic.Mission.DISTANCE], units='ft', ref=1.0e3, connected=False
        )
        prob.traj.link_phases(phases, ['time'], connected=False)

        if len(phases) > 2:
            prob.traj.link_phases(
                phases[1:],
                [Dynamic.Vehicle.ANGLE_OF_ATTACK],
                units='rad',
                connected=False,
            )

    def add_post_mission_systems(self, prob, include_landing=True):
        """
        Add any post mission systems.

        These may include any post-mission take off and landing systems.

        This is called from AviaryProblem.add_post_mission_systems

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        include_landing : bool
            When True, include the landing systems.
        """
        pass

    def add_objective(self, prob):
        """
        Add any additional components related to objectives.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        pass

    def add_guesses(self, prob, phase_name, phase, guesses, target_prob, parent_prefix):
        """
        Adds the initial guesses for each variable of a given phase to the problem.
        This method sets the initial guesses for time, control, state, and problem-specific
        variables for a given phase. If using the GASP model, it also handles some special
        cases that are not covered in the `phase_info` object. These include initial guesses
        for mass, time, and distance, which are determined based on the phase name and other
        mission-related variables.

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

        # for the simple mission method, use the provided initial and final mach
        # and altitude values from phase_info
        initial_altitude = wrapped_convert_units(
            prob.phase_info[phase_name]['user_options']['initial_altitude'], 'ft'
        )
        final_altitude = wrapped_convert_units(
            prob.phase_info[phase_name]['user_options']['final_altitude'], 'ft'
        )
        initial_mach = prob.phase_info[phase_name]['user_options']['initial_mach']
        final_mach = prob.phase_info[phase_name]['user_options']['final_mach']

        guesses['mach'] = ([initial_mach[0], final_mach[0]], 'unitless')
        guesses['altitude'] = ([initial_altitude, final_altitude], 'ft')

        for guess_key, guess_data in guesses.items():
            val, units = guess_data

            # Set initial guess for control variables
            if guess_key in control_keys:
                try:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.controls:{guess_key}',
                        prob._process_guess_var(val, guess_key, phase),
                        units=units,
                    )

                except KeyError:
                    try:
                        target_prob.set_val(
                            parent_prefix + f'traj.{phase_name}.polynomial_controls:{guess_key}',
                            prob._process_guess_var(val, guess_key, phase),
                            units=units,
                        )

                    except KeyError:
                        target_prob.set_val(
                            parent_prefix + f'traj.{phase_name}.bspline_controls:',
                            {guess_key},
                            prob._process_guess_var(val, guess_key, phase),
                            units=units,
                        )
