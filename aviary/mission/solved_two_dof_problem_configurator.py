from aviary.mission.flops_based.phases.groundroll_phase import (
    GroundrollPhase as GroundrollPhaseVelocityIntegrated,
)
from aviary.mission.gasp_based.phases.twodof_phase import TwoDOFPhase
from aviary.mission.problem_configurator import ProblemConfiguratorBase
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Dynamic, Mission
from aviary.mission.utils import process_guess_var


class SolvedTwoDOFProblemConfigurator(ProblemConfiguratorBase):
    """The Solved 2DOF builder is used for detailed take-off and landing."""

    def initial_guesses(self, aviary_group):
        """
        Set any initial guesses for variables in the aviary problem.

        This is called at the end of AivaryProblem.load_inputs.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        if aviary_group.engine_builders is None:
            aviary_group.engine_builders = [build_engine_deck(aviary_group.aviary_inputs)]

        # This doesn't really have much value, but is needed for initializing
        # an objective-related component that still lives in level 2.
        aviary_group.target_range = aviary_group.aviary_inputs.get_val(
            Mission.Design.RANGE, units='NM'
        )

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
        raise RuntimeError('Solved 2DOF requires that a phase_info is specified.')

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
        pass

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
        if phase_options['user_options'].get('ground_roll') and not phase_options[
            'user_options'
        ].get('rotation'):
            phase_builder = GroundrollPhaseVelocityIntegrated
        else:
            phase_builder = TwoDOFPhase

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
        time_units = phase.time_options['units']
        initial = wrapped_convert_units(user_options['time_initial'], time_units)
        duration = wrapped_convert_units(user_options['time_duration'], time_units)
        initial_bounds = wrapped_convert_units(user_options['time_initial_bounds'], time_units)
        duration_bounds = wrapped_convert_units(user_options['time_duration_bounds'], time_units)
        initial_ref = wrapped_convert_units(user_options['time_initial_ref'], time_units)
        duration_ref = wrapped_convert_units(user_options['time_duration_ref'], time_units)

        fix_duration = duration is not None
        fix_initial = initial is not None

        if initial_ref is None:
            # Make a good guess for a reasonable initial time scaler.
            if initial_bounds[0] is not None and initial_bounds[1] != 0.0:
                # Upper bound is good for a ref.
                initial_ref = initial_bounds[1]
            else:
                initial_ref = 600.0

        if duration_ref is None:
            duration_ref = (duration_bounds[0] + duration_bounds[1]) / 2.0

        extra_options = {}
        if not fix_initial:
            extra_options['initial_bounds'] = initial_bounds

        if comm.size == 1 or fix_initial:
            # Redundant on a fixed input; raises a warning if specified.
            extra_options['initial_ref'] = None
        else:
            extra_options['initial_ref'] = initial_ref

        phase.set_time_options(
            fix_initial=fix_initial,
            fix_duration=fix_duration,
            units=time_units,
            duration_bounds=duration_bounds,
            duration_ref=duration_ref,
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
            aviary_group,
            aviary_group.regular_phases,
            'mach_optimize',
            Dynamic.Atmosphere.MACH,
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
            aviary_group,
            aviary_group.reserve_phases,
            'mach_optimize',
            Dynamic.Atmosphere.MACH,
        )

        aviary_group.traj.link_phases(phases, [Dynamic.Vehicle.MASS], connected=True)
        aviary_group.traj.link_phases(
            phases, [Dynamic.Mission.DISTANCE], units='ft', ref=1.0e3, connected=False
        )
        aviary_group.traj.link_phases(phases, ['time'], connected=False)

        if len(phases) > 2:
            aviary_group.traj.link_phases(
                phases[1:],
                [Dynamic.Vehicle.ANGLE_OF_ATTACK],
                units='rad',
                connected=False,
            )

    def check_trajectory(self, aviary_group):
        """
        Checks the phase_info user options for any inconsistency.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        pass

    def add_post_mission_systems(self, model):
        """
        Add any post mission systems.

        These may include any post-mission take off and landing systems.

        This is called from AviaryProblem.add_post_mission_systems

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        """
        pass

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
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
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
            aviary_group.mission_info[phase_name]['user_options']['altitude_initial'], 'ft'
        )
        final_altitude = wrapped_convert_units(
            aviary_group.mission_info[phase_name]['user_options']['altitude_final'], 'ft'
        )
        initial_mach = aviary_group.mission_info[phase_name]['user_options']['mach_initial']
        final_mach = aviary_group.mission_info[phase_name]['user_options']['mach_final']

        guesses['mach'] = ([initial_mach[0], final_mach[0]], 'unitless')
        guesses['altitude'] = ([initial_altitude, final_altitude], 'ft')

        for guess_key, guess_data in guesses.items():
            val, units = guess_data

            if val[0] is None or val[1] is None:
                continue

            # Set initial guess for control variables
            if guess_key in control_keys:
                try:
                    target_prob.set_val(
                        parent_prefix + f'traj.{phase_name}.controls:{guess_key}',
                        process_guess_var(val, guess_key, phase),
                        units=units,
                    )

                except KeyError:
                    try:
                        target_prob.set_val(
                            parent_prefix + f'traj.{phase_name}.polynomial_controls:{guess_key}',
                            process_guess_var(val, guess_key, phase),
                            units=units,
                        )

                    except KeyError:
                        target_prob.set_val(
                            parent_prefix + f'traj.{phase_name}.bspline_controls:',
                            {guess_key},
                            process_guess_var(val, guess_key, phase),
                            units=units,
                        )
