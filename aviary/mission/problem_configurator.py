# TODO: This is very much a conceptual prototype, and needs fine tuning.


class ProblemConfiguratorBase:
    """Base class for a problem configurator in Aviary."""

    def initial_guesses(self, aviary_group):
        """
        Set any initial guesses for variables in the aviary problem.

        This is called at the end of AivaryProblem.load_inputs.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this builder.
        """
        pass

    def get_default_phase_info(self, aviary_group):
        """
        Return a default phase_info for this type or problem.

        The default phase_info is used in the level 1 and 2 interfaces when no
        phase_info is specified.

        This is called during load_inputs.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this builder.

        Returns
        -------
        AviaryValues
            General default phase_info.
        """
        msg = 'This pmethod must be defined in your problem configurator.'
        raise NotImplementedError(msg)

    def get_code_origin(self):
        """
        Return the legacy of this problem configurator.

        Returns
        -------
        LegacyCode
            Code origin enum.
        """
        pass

    def add_takeoff_systems(self, aviary_group):
        """
        Adds takeoff systems to the model in aviary_group.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this builder.
        """
        pass

    def get_phase_builder(self, aviary_group, phase_name, phase_options):
        """
        Return a phase_builder for the requested phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this builder.
        phase_name : str
            Name of the requested phase.
        phase_options : dict
            Phase options for the requested phase.

        Returns
        -------
        PhaseBuilder
            Phase builder for requested phase.
        """
        msg = 'This pmethod must be defined in your problem configurator.'
        raise NotImplementedError(msg)

    def set_phase_options(self, aviary_group, phase_name, phase_idx, phase, user_options, comm):
        """
        Set any necessary problem-related options on the phase.

        This is called from _get_phase in AviaryProblem.add_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this builder.
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
        pass

    def configure_trajectory(self, aviary_group, phases):
        """
        Link or configure phase connections to other upstream or downstream components.

        This is called from AviaryProblem.link_phases

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this configurator.
        phases : list[Phase]
            List of all phases in the trajectory.
        """
        pass

    def check_trajectory(self, aviary_group):
        """
        Checks the phase_info user options for any inconsistency.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this builder.
        """
        pass

    def add_post_mission_systems(self, aviary_group):
        """
        Add any post mission systems.

        These may include any post-mission take off and landing systems.

        This is called from AviaryProblem.add_post_mission_systems

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this builder.
        """
        pass

    def add_objective(self, aviary_group):
        """
        Add any additional components related to objectives.

        Parameters
        ----------
        aviary_group : AviaryGroup
            Aviary model that owns this builder.
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
            Aviary model that owns this builder.
        phase : Phase
            The phase object for which the guesses are being added.
        guesses : dict
            A dictionary containing the initial guesses for the phase.
        target_prob : Problem
            Problem instance to apply the guesses.
        parent_prefix : str
            Location of this trajectory in the hierarchy.
        """
        pass
