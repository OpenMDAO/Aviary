# TODO: This is very much a conceptual prototype, and needs fine tuning.


class ProblemConfiguratorBase:
    """Base class for a problem configurator in Aviary."""

    def initial_guesses(self, prob):
        """
        Set any initial guesses for variables in the aviary problem.

        This is called at the end of AivaryProblem.load_inputs.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        pass

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
        msg = 'This pmethod must be defined in your problem configurator.'
        raise NotImplementedError(msg)

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
        pass

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
        msg = 'This pmethod must be defined in your problem configurator.'
        raise NotImplementedError(msg)

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
        pass

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
        pass

    def check_trajectory(self, prob):
        """
        Checks the phase_info user options for any inconsistency.

        Parameters
        ----------
        prob : AviaryProblem
            Problem that owns this builder.
        """
        pass

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
        pass
