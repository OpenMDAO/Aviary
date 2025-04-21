"""Builder for a simple drag calculation that replaces Aviary's calculation."""

from aviary.examples.external_subsystems.custom_aero.simple_drag import SimpleAeroGroup
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.variables import Aircraft, Dynamic


class CustomAeroBuilder(SubsystemBuilderBase):
    """
    Prototype of a subsystem that overrides an aviary internally computed var.

    It also provides a method to build OpenMDAO systems for the pre-mission and mission computations of the subsystem.

    Attributes
    ----------
    name : str ('simple_aero')
        object label
    """

    def __init__(self, name='simple_aero'):
        super().__init__(name)

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        """
        Build an OpenMDAO system for the mission computations of the subsystem.

        Returns
        -------
        mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen
            during the mission. This includes time-dependent states that are
            being integrated as well as any other variables that vary during
            the mission.
        """
        aero_group = SimpleAeroGroup(
            num_nodes=num_nodes,
        )
        return aero_group

    def mission_inputs(self, **kwargs):
        promotes = [
            Dynamic.Atmosphere.STATIC_PRESSURE,
            Dynamic.Atmosphere.MACH,
            Dynamic.Vehicle.MASS,
            'aircraft:*',
        ]
        return promotes

    def mission_outputs(self, **kwargs):
        promotes = [
            Dynamic.Vehicle.DRAG,
            Dynamic.Vehicle.LIFT,
        ]
        return promotes

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
        params = {}
        params[Aircraft.Wing.AREA] = {
            'shape': (1,),
            'static_target': True,
            'units': 'ft**2',
        }
        return params

    def needs_mission_solver(self, aviary_inputs):
        """
        Return True if the mission subsystem needs to be in the solver loop in mission, otherwise
        return False. Aviary will only place it in the solver loop when True. The default is
        True.
        """
        return False
