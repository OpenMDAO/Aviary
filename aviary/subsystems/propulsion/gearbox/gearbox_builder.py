from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.gearbox.model.gearbox_premission import GearboxPreMission
from aviary.subsystems.propulsion.gearbox.model.gearbox_mission import GearboxMission
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class GearboxBuilder(SubsystemBuilderBase):
    """
    Define the builder for a single gearbox subsystem that provides methods
    to define the gearbox subsystem's states, design variables, fixed values,
    initial guesses, and mass names. It also provides methods to build OpenMDAO
    systems for the pre-mission and mission computations of the subsystem,
    to get the constraints for the subsystem, and to preprocess the inputs for
    the subsystem.

    This is meant to be computations for a single gearbox, so there is no notion
    of "num_gearboxes" in this code.

    This is a reduction gearbox, so gear ratio is input_RPM/output_RPM.
    """

    def __init__(self, name='gearbox', include_constraints=True):
        """Initializes the GearboxBuilder object with a given name."""
        self.include_constraints = include_constraints
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        """Builds an OpenMDAO system for the pre-mission computations of the subsystem."""
        return GearboxPreMission(aviary_inputs=aviary_inputs, simple_mass=True)

    def build_mission(self, num_nodes, aviary_inputs):
        """Builds an OpenMDAO system for the mission computations of the subsystem."""
        return GearboxMission(num_nodes=num_nodes, aviary_inputs=aviary_inputs)

    def get_design_vars(self):
        """
        Design vars are only tested to see if they exist in pre_mission
        Returns a dictionary of design variables for the gearbox subsystem, where the keys are the 
        names of the design variables, and the values are dictionaries that contain the units for 
        the design variable, the lower and upper bounds for the design variable, and any 
        additional keyword arguments required by OpenMDAO for the design variable.
        """

        DVs = {
            Aircraft.Engine.Gearbox.GEAR_RATIO: {
                'opt': True,
                'units': 'unitless',
                'lower': 1.0,
                'upper': 20.0,
                'val':  10  # initial value
            },
            # This var appears in both mission and pre-mission
            Aircraft.Engine.Gearbox.SHAFT_POWER_DESIGN: {
                'val': 10000,
                'units': 'kW',
                'lower': 1.0,
                'upper': None,
            }
        }
        return DVs

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        """
        Parameters are only tested to see if they exist in mission.
        A value the doesn't change throught the mission mission
        Returns a dictionary of fixed values for the gearbox subsystem, where the keys are the names 
        of the fixed values, and the values are dictionaries that contain the fixed value for the 
        variable, the units for the variable, and any additional keyword arguments required by 
        OpenMDAO for the variable.

        Returns
        -------
        parameters : list
        A list of names for the gearbox subsystem.
        """
        parameters = {
            Aircraft.Engine.Gearbox.EFFICIENCY: {
                'val': 0.98,
                'units': 'unitless',
            },
        }

        return parameters

    def get_mass_names(self):
        return [Aircraft.Engine.Gearbox.MASS]

    def get_outputs(self):
        return [
            Dynamic.Mission.SHAFT_POWER + '_out',
            Dynamic.Mission.SHAFT_POWER_MAX + '_out',
            Dynamic.Mission.RPM + '_out',
            Dynamic.Mission.TORQUE + '_out',
            Mission.Constraints.GEARBOX_SHAFT_POWER_RESIDUAL,
        ]

    def get_constraints(self):
        if self.include_constraints:
            constraints = {
                Mission.Constraints.GEARBOX_SHAFT_POWER_RESIDUAL: {
                    'lower': 0.0,
                    'type': 'path',
                    'units': 'kW',
                }
            }
        else:
            constraints = {}
        return constraints
