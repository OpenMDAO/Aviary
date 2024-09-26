from abc import ABC

from aviary.variable_info.variable_meta_data import _MetaData


class SubsystemBuilderBase(ABC):
    """
    Base class of subsystem builder.

    Attributes
    ----------
    name : string
        Name of subsystem, as it will appear when added to the OpenMDAO problem
    meta_data : dict
        Dictionary containing the variable metadata used by this Aviary problem
    ```
    """

    __slots__ = ('name', 'meta_data')

    # derived type customization point
    default_name = 'default_subsystem_name'

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = self.default_name
        self.name = name

        if meta_data is None:
            meta_data = _MetaData
        self.meta_data = meta_data

    def build_pre_mission(self, aviary_inputs, **kwargs):
        """
        Build an OpenMDAO System for the pre-mission computations of the subsystem.

        Required for subsystems with pre-mission computations.

        Used in level3.py to build the pre-mission system.

        Returns
        -------
        pre_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen in
            the pre-mission part of the Aviary problem. This
            includes sizing, design, and other non-mission parameters.
        """
        return None

    def get_states(self):
        """
        Return a dictionary of states defined by this subsystem.

        Required for subsystems with mission-based dynamics.

        Note that there must be outputs provided by the user's model that provide
        the time derivative of each state. The recommended convention is to name the output
        f"{state_name}_rate", where `state_name` is the name of the state variable.

        Use in the phase builders (e.g. cruise_phase.py) when other states are added to the phase.

        Returns
        -------
        states : dict
            A dictionary where the keys are the names of the state variables
            and the values are dictionaries with the following keys:

            - 'rate_source': a string indicating the name of the output that provides the time derivative of the state variable
            - 'units': a string indicating the units of the state variable
            - any additional keyword arguments required by Dymos for the state variable.
        """
        return {}

    def get_controls(self, phase_name=None):
        """
        Return a dictionary of control variables for the subsystem.

        Parameters
        ----------
        phase_name : str, optional
            Name of the flight phase. This allows for different control variables to be used in different flight phases.
            You can add branching logic however you want based on the phase_name within a builder.

        Notes
        -----
        This method is optional, used if subsystems have control variables.
        Used in the phase builders (e.g. cruise_phase.py) when other controls are added to the phase.

        Returns
        -------
        controls : dict
            A dictionary where the keys are the names of the control variables
            and the values are dictionaries with the following keys:

            - 'units': str
                The units for the control variable.
            - any additional keyword arguments required by OpenMDAO for the control
            variable.
        """
        return {}

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        """
        Return a dictionary of fixed values for the subsystem.

        Optional, used if subsystems have fixed values.

        Used in the phase builders (e.g. cruise_phase.py) when other parameters are added to the phase.

        This is distinct from `get_design_vars` in a nuanced way. Design variables
        are variables that are optimized by the problem that are not at the phase level.
        An example would be something that occurs in the pre-mission level of the problem.
        Parameters are fixed values that are held constant throughout a phase, but if
        `opt=True`, they are able to change during the optimization.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.
        phase_info : dict
            The phase_info subdict for this phase.

        Returns
        -------
        fixed_values : dict
            A dictionary where the keys are the names of the fixed variables
            and the values are dictionaries with the following keys:

            - 'value': float or array
                The fixed value for the variable.
            - 'units': str
                The units for the fixed value (optional).
            - any additional keyword arguments required by OpenMDAO for the fixed
              variable.
        """
        return {}

    def get_constraints(self):
        """
        Return a dictionary of constraints for the subsystem.

        Optional, used if subsystems have path or boundary constraints.

        Used in the phase builders (e.g. cruise_phase.py) when other constraints are added to the phase.

        Returns
        -------
        constraints : dict
            A dictionary where the keys are the names of the constraint variables
            and the values are dictionaries with the following keys:

            - type : str
                The type of constraint. Must be one of 'path' or 'boundary'.
            - any additional keyword arguments required by OpenMDAO for the constraint
              variable.
        """
        return {}

    def get_linked_variables(self):
        """
        Return a list of variable names that will be linked between phases.

        Optional

        Returns
        -------
        linked_vars : list of variables to link between phases
        """
        return []

    def get_bus_variables(self):
        """
        Return a dictionary of variables that will be passed from the pre-mission
        to mission systems.

        Returns
        -------
        bus_variables : dict
            A dictionary where the keys are the names of the variables
            output in the pre-mission subsystem and the values are
            dictionaries with the following keys:

            - 'mission_name': str or list
                Names of the input variable to be connected in the mission
                subsystem (optional).
            - 'post_mission_name': str or list
                Names of the input variable to be connected in the post-
                mission subsystem (optional).
            - 'units' : str
                This is temporary and will be removed, however, it is
                currently a requirement.

        """
        return {}

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        """
        Build an OpenMDAO System for the mission computations of the subsystem.

        Required for subsystems with mission-based dynamics.

        Used in the ODE class definition (e.g. mission_ODE.py) to build the mission system.

        Returns
        -------
        mission_sys : openmdao.core.System
            An OpenMDAO System containing all computations that need to happen
            during the mission. This includes time-dependent states that are
            being integrated as well as any other variables that vary during
            the mission.
        """
        return None

    def define_order(self):
        """
        Return a list of subsystem names that must be defined before this one. E.g., must go before or after aero or prop.

        Optional

        This will be made easier once an upcoming POEM about reordering systems in OM is integrated.

        Not needed yet, but will be useful when subsystem order matters, like for certain aero-prop cases.

        Returns
        -------
        order : list of str
            A list of subsystem names that this subsystem depends on.
        """
        return []

    def get_design_vars(self):
        """
        Return a dictionary of design variables for the subsystem.

        Optional

        Not currently used.

        Returns
        -------
        design_vars : dict
            A dictionary where the keys are the names of the design variables
            and the values are dictionaries with the following keys:

            - any additional keyword arguments required by OpenMDAO for the design
              variable.
        """
        return {}

    def get_initial_guesses(self):
        """
        Return a dictionary of initial guesses for the subsystem.

        Optional

        Returns
        -------
        initial_guesses : dict
            A dictionary where the keys are the names of the variables
            and the values are dictionaries with the following keys:

            - 'val': float or array
                The initial guess for the variable.
            - 'units': str
                The units for the initial guess (optional).
            - any additional keyword arguments required by OpenMDAO for the initial
              guess.
        """
        return {}

    def get_mass_names(self):
        """
        Return a list of names of the mass variables for the subsystem.

        Returns
        -------
        mass_names : list of str
            A list of the names of the mass variables for the subsystem.
        """
        return []

    def preprocess_inputs(self, aviary_inputs):
        """
        Preprocess the inputs to the subsystem, returning a modified AviaryValues object.

        This method is called after the inputs are passed to the subsystem. It
        can be used to modify the inputs before they are used in the subsystem.

        Parameters
        ----------
        aviary_inputs : dict
            A dictionary containing the inputs to the subsystem.
        """
        return aviary_inputs

    def get_outputs(self):
        """
        Return a list of output variable names for the subsystem. Also adds
        all of these outputs to the Dymos timeseries outputs for use when graphing
        or post-processing the mission.

        Useful for knowing what a subsystem provides that other subsystems can use.
        """
        return []

    def build_post_mission(self, aviary_inputs, **kwargs):
        """
        Build an OpenMDAO System for the post-mission computations of the subsystem.

        Required for subsystems with post-mission-based analyses.

        Returns
        -------
        post_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen
            after the mission. This includes time-dependent states that are
            being integrated as well as any other variables that vary after
            the mission.
        """
        return None

    def report(self, prob, reports_folder, **kwargs):
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
