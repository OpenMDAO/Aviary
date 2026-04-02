"""Builder for a simple drag calculation that replaces Aviary's calculation."""

from aviary.models.external_subsystems.simple_aero.simple_drag import SimpleAeroGroup
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.variable_info.variables import Aircraft, Dynamic


class SimpleAeroBuilder(SubsystemBuilder):
    """A basic subsystem that provides a simplified equation for drag computation."""

    _default_name = 'simple_aero'

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        """
        Build an OpenMDAO system for the mission computations of the subsystem.

        Returns
        -------
        aero_group : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen during the mission.
            This includes time-dependent states that are being integrated as well as any other
            variables that vary during the mission.
        """
        aero_group = SimpleAeroGroup(
            num_nodes=num_nodes,
        )
        return aero_group

    def mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        promotes = [
            Dynamic.Atmosphere.STATIC_PRESSURE,
            Dynamic.Atmosphere.MACH,
            Dynamic.Vehicle.MASS,
            'aircraft:*',
        ]
        return promotes

    def mission_outputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        promotes = [
            Dynamic.Vehicle.DRAG,
            Dynamic.Vehicle.LIFT,
        ]
        return promotes

    def get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        """
        Return a dictionary of parameters for the subsystem. (Optional)

        A parameter is a value that does not vary over the trajectory. Adding a variable name to
        this list promotes the input to the top of the Aviary model, where it is either implicitly
        connected to any pre-mission component that produces it, or it assumes the value set in
        the csv file. Parameters can be constant or they can be computed in premission. They can
        also be declared as design variables set by the optimizer if you pass `opt=True` in the
        parameter's dictionary.

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
        params = {}
        params[Aircraft.Wing.AREA] = {
            'shape': (1,),
            'static_target': True,
            'units': 'ft**2',
        }
        return params

    def needs_mission_solver(self, aviary_inputs, subsystem_options):
        """
        Return True if the mission subsystem needs to be in the solver loop in mission, otherwise
        return False. Aviary will only place it in the solver loop when True. The default is
        True.
        """
        return False
