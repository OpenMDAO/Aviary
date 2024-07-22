from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.gearbox.model.gearbox_premission import GearboxPreMission
from aviary.subsystems.propulsion.gearbox.model.gearbox_mission import GearboxMission
from aviary.variable_info.variables import Aircraft, Dynamic


class GearboxBuilder(SubsystemBuilderBase):
    '''
    Define the builder for a single gearbox subsystem that provides methods to define the gearbox subsystem's states, design variables, fixed values, initial guesses, and mass names.

    It also provides methods to build OpenMDAO systems for the pre-mission and mission computations of the subsystem, to get the constraints for the subsystem, and to preprocess the inputs for the subsystem.

    This is meant to be computations for a single gearbox, so there is no notion of "num_gearboxs" in this code.
    '''

    def __init__(self, name='gearbox', include_constraints=True):
        '''Initializes the GearboxBuilder object with a given name.'''
        self.include_constraints = include_constraints
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        '''Builds an OpenMDAO system for the pre-mission computations of the subsystem.'''
        return GearboxPreMission(aviary_inputs=aviary_inputs)

    def build_mission(self, num_nodes, aviary_inputs):
        '''Builds an OpenMDAO system for the mission computations of the subsystem.'''
        return GearboxMission(num_nodes=num_nodes, aviary_inputs=aviary_inputs)

    def get_design_vars(self):
        '''
        Design vars are only tested to see if they exist in pre_mission
        Returns a dictionary of design variables for the gearbox subsystem, where the keys are the 
        names of the design variables, and the values are dictionaries that contain the units for 
        the design variable, the lower and upper bounds for the design variable, and any 
        additional keyword arguments required by OpenMDAO for the design variable.

                        '''

        DVs = {
            Aircraft.Engine.Gearbox.GEAR_RATIO: {
                'opt': True,
                'units': None,
                'lower': 1.0,
                'upper': 20.0,
                'val':  10  # initial value
            },
            Aircraft.Engine.Gearbox.SPECIFIC_TORQUE: {
                'lower': 100,
                'upper': 100,
                'opt': False,
                'val': 100,
                'units': 'N*m/kg',
            }
        }
        return DVs

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        '''
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
        '''
        parameters = {
            Aircraft.Engine.Gearbox.EFFICIENCY: {
                'val': 0.98,
                'units': None,
            },
        }

        return parameters

    def get_mass_names(self):
        return [Aircraft.Engine.Gearbox.MASS]

    def get_outputs(self):
        return [
            Dynamic.Mission.RPM_GEAR,
            Dynamic.Mission.SHAFT_POWER_GEAR,
            Dynamic.Mission.SHAFT_POWER_MAX_GEAR,
            Dynamic.Mission.TORQUE_GEAR,
        ]
