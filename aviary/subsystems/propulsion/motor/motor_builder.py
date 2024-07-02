from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.motor.model.motor_premission import MotorPreMission
from aviary.subsystems.propulsion.motor.model.motor_mission import MotorMission
from aviary.variable_info.variables import Aircraft, Dynamic


class MotorBuilder(SubsystemBuilderBase):
    '''
    Define the builder for a single motor subsystem that provides methods to define the motor subsystem's states, design variables, fixed values, initial guesses, and mass names.

    It also provides methods to build OpenMDAO systems for the pre-mission and mission computations of the subsystem, to get the constraints for the subsystem, and to preprocess the inputs for the subsystem.

    This is meant to be computations for a single motor, so there is no notion of "num_motors" in this code.

    Attributes
    ----------
    name : str ('motor')
        object label

    Methods
    -------
    __init__(self, name='motor'):
        Initializes the MotorBuilder object with a given name.
    get_states(self) -> dict:
        Returns a dictionary of the subsystem's states, where the keys are the names of the state variables, and the values are dictionaries that contain the units for the state variable and any additional keyword arguments required by OpenMDAO for the state variable.
    get_linked_variables(self) -> list:
        Links voltage input from the battery subsystem
    build_pre_mission(self) -> openmdao.core.System:
        Builds an OpenMDAO system for the pre-mission computations of the subsystem.
    build_mission(self, num_nodes, aviary_inputs) -> openmdao.core.System:
        Builds an OpenMDAO system for the mission computations of the subsystem.
    get_constraints(self) -> dict:
        Returns a dictionary of constraints for the motor subsystem, where the keys are the names of the variables to be constrained, and the values are dictionaries that contain the lower and upper bounds for the constraint and any additional keyword arguments accepted by Dymos for the constraint.
    get_design_vars(self) -> dict:
        Returns a dictionary of design variables for the motor subsystem, where the keys are the names of the design variables, and the values are dictionaries that contain the units for the design variable, the lower and upper bounds for the design variable, and any additional keyword arguments required by OpenMDAO for the design variable.
    get_parameters(self) -> dict:
        Returns a dictionary of fixed values for the motor subsystem, where the keys are the names of the fixed values, and the values are dictionaries that contain the fixed value for the variable, the units for the variable, and any additional keyword arguments required by OpenMDAO for the variable.
    get_initial_guesses(self) -> dict:
        Returns a dictionary of initial guesses for the motor subsystem, where the keys are the names of the initial guesses, and the values are dictionaries that contain the initial guess value, the type of variable (state or control), and any additional keyword arguments required by OpenMDAO for the variable.
    get_mass_names(self) -> list:
        Returns a list of names for the motor subsystem mass.
    preprocess_inputs(self, aviary_inputs) -> aviary_inputs:
        No preprocessing needed for the motor subsystem.
    '''

    def __init__(self, name='motor', include_constraints=True):
        self.include_constraints = include_constraints
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        return MotorPreMission(aviary_inputs=aviary_inputs)

    def build_mission(self, num_nodes, aviary_inputs):
        return MotorMission(num_nodes=num_nodes, aviary_inputs=aviary_inputs)

    # def get_constraints(self):
        # if self.include_constraints:
        #     constraints = {
        # Dynamic.Mission.Motor.TORQUE_CON: {
        #     'upper': 0.0,
        #     'type': 'path'
        # }
        # # TBD Gearbox torque constraint
        #     }
        # else:
        #     constraints = {}

        # return constraints

    def get_design_vars(self):
        DVs = {
            Aircraft.Engine.SCALE_FACTOR: {
                'units': 'unitless',
                'lower': 0.001,
                'upper': None
            },
            Aircraft.Engine.Gearbox.GEAR_RATIO: {
                'units': None,
                'lower': 1.0,
                'upper': 1.0,
            }
        }

        return DVs

    # def get_initial_guesses(self):
        # initial_guess_dict = {
        # Aircraft.Motor.RPM: {
        #     'units': 'rpm',
        #     'type': 'parameter',
        #     'val': 4000.0,  # based on our map
        # },
        # }

        # return initial_guess_dict

    def get_mass_names(self):
        '''
        Return a list of names for the motor subsystem.

        Returns
        -------
        mass_names : list
            A list of names for the motor subsystem.
        '''
        return [Aircraft.Engine.Motor.MASS,
                Aircraft.Engine.Gearbox.MASS]

    def get_outputs(self):
        '''
        Return a list of output names for the motor subsystem.

        Returns
        -------
        outputs : list
            A list of variable names for the motor subsystem.
        '''

        return [Dynamic.Mission.TORQUE,
                Dynamic.Mission.SHAFT_POWER,
                Dynamic.Mission.SHAFT_POWER_MAX,
                Dynamic.Mission.ELECTRIC_POWER_IN,
                Dynamic.Mission.THRUST,
                Dynamic.Mission.NOX_RATE,
                Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE]
