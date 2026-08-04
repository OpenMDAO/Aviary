"""Builder for a simple drag calculation that replaces Aviary's calculation."""

from aviary.subsystems.aerodynamics.UAV_Aero.simple_drag import SimpleAeroGroup
from aviary.subsystems.subsystem_builder import SubsystemBuilder as SubsystemBuilderBase
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
            # Core aero's get_timeseries() registers these unconditionally (even with
            # method='external'), so the external aero must supply them too.
            Dynamic.Vehicle.DRAG_COEFFICIENT,
            Dynamic.Vehicle.LIFT_COEFFICIENT,
        ]
        return promotes

    def get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None, **kwargs):
        """
        Return fixed parameter definitions for this subsystem.

        Use this when the subsystem has values that stay constant within a
        phase. These parameters can still be optimized when `opt=True`.

        This differs from `get_design_vars`: design variables are optimized
        outside the phase-level parameter setup.

        Parameters
        ----------
        phase_info : dict
            Phase-specific settings.

        Returns
        -------
        fixed_values : dict
            Mapping of variable names to OpenMDAO metadata:

            - 'value': float or array
                Fixed value for the variable.
            - 'units': str
                Units for the value (optional).
            - any additional keyword arguments required by OpenMDAO.
        """
        params = {}
        params[Aircraft.Wing.AREA] = {
            'shape': (1,),
            'static_target': True,
            'units': 'ft**2',
        }
        return params

    def needs_mission_solver(self, aviary_inputs=None, subsystem_options=None, **kwargs):
        """
        Return True if the mission subsystem needs to be in the solver loop in mission, otherwise
        return False. Aviary will only place it in the solver loop when True. The default is
        True.
        """
        return False
