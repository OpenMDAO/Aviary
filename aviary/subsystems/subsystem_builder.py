from abc import ABC

from openmdao.core.system import System

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData


class SubsystemBuilder(ABC):
    """
    Base class of subsystem builder.

    Attributes
    ----------
    name : string
        Name of subsystem, as it will appear when added to the OpenMDAO problem
    meta_data : dict
        Dictionary containing the variable metadata used by this Aviary problem
    """

    __slots__ = ('name', 'meta_data')

    # derived type customization point
    _default_name = 'default_subsystem_name'
    _default_metadata = _MetaData

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = self._default_name
        self.name = name

        if meta_data is None:
            meta_data = self._default_metadata
        self.meta_data = meta_data

    def needs_mission_solver(self, aviary_inputs, subsystem_options):
        """
        Return True if the mission subsystem needs to be in the solver loop in mission, otherwise
        return False. Aviary will only place it in the solver loop when True. The default is
        True.

        Parameters
        ----------
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        """
        return True

    def build_pre_mission(self, aviary_inputs, subsystem_options=None) -> None | System:
        """
        Build an OpenMDAO System for the pre-mission computations of the subsystem.

        Required for subsystems with pre-mission computations.

        Used in level3.py to build the pre-mission system.

        Parameters
        ----------
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in pre-mission.

        Returns
        -------
        pre_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen in the pre-mission
            part of the Aviary problem. This includes sizing, design, and other non-mission
            parameters.
        """
        return None

    def get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Return a dictionary of dynamic states defined by this subsystem (Optional).

        Required for subsystems with mission-based dynamics.

        Note that there must be outputs provided by the user's model that provide the time
        derivative of each state. The recommended convention is to name the output
        f"{state_name}_rate", where `state_name` is the name of the state variable.

        Parameters
        ----------
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        dict
            A dictionary where the keys are the names of the state variables
            and the values are dictionaries with the following keys:

            - 'rate_source': a string indicating the name of the output that provides the time
               derivative of the state variable
            - 'units': a string indicating the units of the state variable
            - any additional keyword arguments required by Dymos for the state variable.
        """
        return {}

    def get_controls(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Return a dictionary of control variables for the subsystem (Optional).

        Parameters
        ----------
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        dict
            A dictionary where the keys are the names of the control variables
            and the values are dictionaries with the following keys:

            - 'units': str
                The units for the control variable.
            - 'opt': bool
                When True, this control becomes an optimizer design variable.
            - any additional keyword arguments required by OpenMDAO for the control variable.
        """
        return {}

    def get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Return a dictionary of parameters for the subsystem (Optional).

        A parameter is a value that does not vary over the trajectory. Adding a variable name to
        this list promotes the input to the top of the Aviary model, where it is either implicitly
        connected to any pre-mission component that produces it, or it assumes the value set in
        the csv file.

        Parameters
        ----------
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        dict
            A dictionary where the keys are the names of the fixed parameters and the values are
            dictionaries with the following keys:

            - 'value': float or array
                The fixed value for the variable.
            - 'units': str
                The units for the fixed value (optional).
            - any additional keyword arguments required by OpenMDAO for the fixed variable.
        """
        return {}

    def get_constraints(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Return a dictionary of constraints for the subsystem.

        Use when subsystems have path or boundary constraints in the phases.

        Parameters
        ----------
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        dict
            A dictionary where the keys are the names of the constraint variables and the values are
            dictionaries with the following keys:

            - type : str
                The type of constraint. Must be one of 'path' or 'boundary'.
            - any additional keyword arguments required by OpenMDAO for the constraint
              variable.
        """
        return {}

    def get_linked_variables(self, aviary_inputs=None):
        """
        Return a list of variable names that will be linked between phases.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.

        Returns
        -------
        linked_vars : list of variables to link between phases
        """
        return []

    def get_bus_variables(self, aviary_inputs=None):
        # This is an error instead of a warning because it has potential to cause your model to
        # fail in ways that are difficult to debug.
        raise RuntimeError(
            '"get_bus_variables()" has been renamed to "get_pre_mission_bus_variables()" to '
            'differentiate it from "get_post_mission_bus_variables()". Please rename this method '
            'in your subsytem builders.'
        )

    def get_pre_mission_bus_variables(self, aviary_inputs=None, mission_info=None):
        """
        Return a dictionary of variables that will be passed from the pre-mission
        to mission and post-mission systems.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.
        mission_info : dict
            The mission_info dict containing the phase_info for each phase.

        Returns
        -------
        bus_variables : dict
            A dictionary where the keys are the names of the variables output in the pre-mission
            subsystem and the values are dictionaries with the following keys:

            - 'mission_name': str or list
                Names of the input variable to be connected in the mission subsystem (optional).
            - 'post_mission_name': str or list
                Names of the input variable to be connected in the post-mission subsystem
                (optional).
            - 'units' : str
                This is temporary and will be removed, however, it is currently a requirement.
            - 'src_indices': int or list of ints or tuple of ints or int ndarray or Iterable or None
                Indices of the pre-mission variable for connection.
        """
        return {}

    def build_mission(
        self, num_nodes, aviary_inputs, user_options, subsystem_options
    ) -> None | System:
        """
        Build an OpenMDAO System for the mission computations of the subsystem.

        Required for subsystems with mission-based dynamics.

        Used in the ODE class definition (e.g. mission_ODE.py) to build the mission system.

        Parameters
        ----------
        num_nodes : int
            Number of nodes present in the current Dymos phase of mission analysis.
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        mission_sys : openmdao.core.System
            An OpenMDAO System containing all computations that need to happen during the mission.
            This includes time-dependent states that are being integrated as well as any other
            variables that vary during the mission.
        """
        return None

    def mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Returns list of mission inputs to be promoted out of the external subsystem. By default, all
        inputs are promoted. Used when only a subset of inputs should be promoted.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        list
            List of all mission inputs.
        """
        return ['*']

    def mission_outputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Returns list of mission outputs to be promoted out of the external subsystem. By default,
        all outputs are promoted. Used when only a subset of outputs should be promoted.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        list
            List of all mission outputs.
        """
        return ['*']

    def get_design_vars(self, aviary_inputs=None):
        """
        Return a dictionary of design variables for the subsystem.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.

        Returns
        -------
        design_vars : dict
            A dictionary where the keys are the names of the design variables and the values are
            dictionaries with the following keys:

            - any additional keyword arguments required by OpenMDAO for the design variable.
        """
        return {}

    def get_initial_guesses(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Return a dictionary of initial guesses for the subsystem.

        Parameters
        ----------
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        dict
            A dictionary where the keys are the names of the variables and the values are
            dictionaries with the following keys:

            - 'val': float or array
                The initial guess for the variable.
            - 'units': str
                The units for the initial guess (optional).
            - any additional keyword arguments required by OpenMDAO for the initial guess.
        """
        return {}

    def get_mass_names(self, aviary_inputs=None):
        """
        Return a list of names of the mass variables for the subsystem.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.

        Returns
        -------
        mass_names : list of str
            A list of the names of the mass variables for the subsystem.
        """
        return []

    def preprocess_inputs(self, aviary_inputs=None) -> AviaryValues:
        """
        Preprocess the inputs to the subsystem, returning a modified AviaryValues object.

        This method is called after the inputs are passed to the subsystem. It can be used to modify
        the inputs before they are used in the subsystem.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.
        """
        return aviary_inputs

    def get_outputs(self):
        DeprecationWarning(
            '"get_outputs()" has been renamed to "get_timeseries()" for more clarity on the '
            'function\'s purpose. "get_outputs()" will be removed in a future version of Aviary.'
        )
        return self.get_timeseries()

    def get_timeseries(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Return a list of outputs to add to the Dymos timeseries outputs for use when graphing or
        post-processing the mission.

        Parameters
        ----------
        aviary_inputs : dict
            Dictionary containing the aircraft definition.
        user_options : dict
            Dictionary of user options for this phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in this phase.

        Returns
        -------
        list
            List of outputs to add to the timeseries.
        """
        return []

    def get_post_mission_bus_variables(self, aviary_inputs=None, mission_info=None):
        """
        Return a dict mapping phase names to a dict mapping mission variable names to (a list of)
        post-mission variable names.

        Mission variables local to a given external subsystem should be prefixed with that
        subsystem's name. For example, to connect a variable 'bar' that is an output of the external
        subsystem "foo"'s mission to the post-mission variable "cruise_foo", map "foo.bar" to
        "cruise_foo".

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.
        mission_info : dict
            The mission_info dict containing the phase_info for each phase.

        Returns
        -------
        bus_variables : dict
            A dictionary where the keys are the phase name and the values are dictionaries with the keys being the mission variable names to connect from,
            and values are dictionaries with the following keys:
            - 'post_mission_name': str or list
                Names of the input variable to be connected in the post-mission subsystem
            - 'src_indices': int or list of ints or tuple of ints or int ndarray or Iterable or None
                Indices of the mission variable for connection
        """
        return {}

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ) -> None | System:
        """
        Build an OpenMDAO System for the post-mission computations of the subsystem.

        Required for subsystems with post-mission-based analyses.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.
        mission_info : dict
            The mission_info dict containing the phase_info for each phase.
        subsystem_options : dict
            Dictionary of optional arguments for this subsystem in post_mission.
        phase_mission_bus_lengths : dict
            Mapping from phase names to the lengths of the phase's "mission_bus_variables"
            timeseries

        Returns
        -------
        post_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen after the mission.
            This includes time-dependent states that are being integrated as well as any other
            variables that vary after the mission.
        """
        return None

    def report(self, prob, reports_folder):
        """
        Generates report file for this subsystem. If this subsystem doesn't need a
        report, do nothing.

        Parameters
        ----------
        prob : AviaryProblem
            The AviaryProblem that will be used to generate the report
        reports_folder : Path
            Location of the subsystems_report folder this report will be placed in
        """
        return
