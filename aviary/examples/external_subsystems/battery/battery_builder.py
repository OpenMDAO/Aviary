from aviary.examples.external_subsystems.battery.battery_variables import Aircraft, Mission
from aviary.examples.external_subsystems.battery.battery_variable_meta_data import ExtendedMetaData
from aviary.examples.external_subsystems.battery.model.battery_mission import BatteryMission
from aviary.examples.external_subsystems.battery.model.battery_premission import BatteryPreMission
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase


class BatteryBuilder(SubsystemBuilderBase):
    '''
    Define the builder for a battery subsystem that provides methods to define the battery subsystem's states, design variables, fixed values, initial guesses, and mass names.
    It also provides methods to build OpenMDAO systems for the pre-mission and mission computations of the subsystem, to get the constraints for the subsystem, and to preprocess the inputs for the subsystem.

    As a contrast, The battery in the examples is a "simple" battery
    model which only tracks state-of-charge throughout the mission. This battery model is only an example, and is not used by any of Aviary's core subsystems (like the pycycle or OAS examples).

    Attributes
    ----------
    name : str ('battery')
        object label
    include_constraints: boolean
        flag whether to include constraints.

    Methods
    -------
    __init__(self, name='battery', include_constraints=True):
        Initializes the BatteryBuilder object with a given name.
    get_states(self) -> dict:
        Returns a dictionary of the subsystem's states, where the keys are the names of the state variables, and the values are dictionaries that contain the units for the state variable and any additional keyword arguments required by OpenMDAO for the state variable.
    get_linked_variables(self) -> list:
        Returns an empty list since no variable linking is required for the battery subsystem.
    build_pre_mission(self, aviary_inputs) -> openmdao.core.System:
        Builds an OpenMDAO system for the pre-mission computations of the subsystem.
    build_mission(self, num_nodes, aviary_inputs) -> openmdao.core.System:
        Builds an OpenMDAO system for the mission computations of the subsystem.
    get_constraints(self) -> dict:
        Returns a dictionary of constraints for the battery subsystem, where the keys are the names of the variables to be constrained, and the values are dictionaries that contain the lower and upper bounds for the constraint and any additional keyword arguments accepted by Dymos for the constraint.
    get_design_vars(self) -> dict:
        Returns a dictionary of design variables for the battery subsystem, where the keys are the names of the design variables, and the values are dictionaries that contain the units for the design variable, the lower and upper bounds for the design variable, and any additional keyword arguments required by OpenMDAO for the design variable.
    get_parameters(self, aviary_inputs=None, phase_info=None) -> dict:
        Returns a dictionary of fixed values for the battery subsystem, where the keys are the names of the fixed values, and the values are dictionaries that contain the fixed value for the variable, the units for the variable, and any additional keyword arguments required by OpenMDAO for the variable.
    get_initial_guesses(self) -> dict:
        Returns a dictionary of initial guesses for the battery subsystem, where the keys are the names of the initial guesses, and the values are dictionaries that contain the initial guess value, the type of variable (state or control), and any additional keyword arguments required by OpenMDAO for the variable.
    get_mass_names(self) -> list:
        Returns a list of names for the battery subsystem mass.
    preprocess_inputs(self, aviary_inputs) -> aviary_inputs:
        preprocesses the inputs for the battery subsystem, setting the values for battery performance based on the battery cell type.
    '''

    def __init__(self, name='battery', include_constraints=True):
        self.include_constraints = include_constraints
        super().__init__(name, meta_data=ExtendedMetaData)

    def get_states(self):
        '''
        Return a dictionary of states for the battery subsystem.

        Returns
        -------
        states : dict
            A dictionary where the keys are the names of the state variables
            and the values are dictionaries with the following keys:

            - 'units': str
                The units for the state variable.
            - any additional keyword arguments required by OpenMDAO for the state
              variable.
        '''
        states_dict = {
            Mission.Battery.STATE_OF_CHARGE: {
                'rate_source': Mission.Battery.STATE_OF_CHARGE_RATE,
                'fix_initial': True,
            },
            Mission.Battery.VOLTAGE_THEVENIN: {
                'units': 'V',
                'rate_source': Mission.Battery.VOLTAGE_THEVENIN_RATE,
                'defect_ref': 1.e5,
                'ref': 1.e5,
            },
        }

        return states_dict

    def get_linked_variables(self):
        '''
        Return the list of linked variables for the battery subsystem; in this case
        it's our two state variables.
        '''
        return [Mission.Battery.VOLTAGE_THEVENIN, Mission.Battery.STATE_OF_CHARGE]

    def build_pre_mission(self, aviary_inputs):
        '''
        Build an OpenMDAO system for the pre-mission computations of the subsystem.

        Returns
        -------
        pre_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen in
            the pre-mission part of the Aviary problem. This
            includes sizing, design, and other non-mission parameters.
        '''
        return BatteryPreMission()

    def build_mission(self, num_nodes, aviary_inputs):
        '''
        Build an OpenMDAO system for the mission computations of the subsystem.

        Returns
        -------
        mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen
            during the mission. This includes time-dependent states that are
            being integrated as well as any other variables that vary during
            the mission.
        '''
        return BatteryMission(num_nodes=num_nodes, aviary_inputs=aviary_inputs)

    def get_constraints(self):
        '''
        Return a dictionary of constraints for the battery subsystem.

        Returns
        -------
        constraints : dict
            A dictionary where the keys are the names of the variables to be constrained
            and the values are dictionaries are any accepted by Dymos for the
            constraint.

        Description
        -----------
        This method returns a dictionary of constraints for the battery subsystem. The dictionary
        contains two constraints:

        - state of charge constraint: A boundary constraint that ensures the state of charge does not fall
          below 0.2 at the final mission point.
        - thevenin voltage constraint: A path constraint that ensures the Thevenin voltage does not
          fall below 0.
        '''
        if self.include_constraints:
            constraints = {
                Mission.Battery.STATE_OF_CHARGE: {
                    'lower': 0.2,
                    'type': 'boundary',
                    'loc': 'final',
                },
                Mission.Battery.VOLTAGE_THEVENIN: {
                    'lower': 0,
                    'type': 'path',
                },
            }
        else:
            constraints = {}

        return constraints

    def get_design_vars(self):
        '''
        Return a dictionary of design variables for the battery subsystem.

        Returns
        -------
        design_vars : dict
            A dictionary where the keys are the names of the design variables
            and the values are dictionaries with the following keys:

            - 'units': str
                The units for the design variable
            - 'lower': float or None
                The lower bound for the design variable
            - 'upper': float or None
                The upper bound for the design variable
            - any additional keyword arguments required by OpenMDAO for the design
              variable
        '''

        DVs = {
            Aircraft.Battery.Cell.DISCHARGE_RATE: {
                'units': 'A',
                'lower': 0.0,
                'upper': 2.0,
            },
        }

        return DVs

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        '''
        Return a dictionary of fixed values exposed to the phases for the battery subsystem.

        Returns
        -------
        parameter_info : dict
            A dictionary where the keys are the names of the fixed values
            and the values are dictionaries with the following keys:

            - 'value': float or None
                The fixed value for the variable.
            - 'units': str or None
                The units for the variable.
            - any additional keyword arguments required by OpenMDAO for the variable.
        '''

        parameters_dict = {
            Mission.Battery.TEMPERATURE: {'val': 25.0, 'units': 'degC'},
            Mission.Battery.CURRENT: {'val': 3.25, 'units': 'A'}
        }

        return parameters_dict

    def get_initial_guesses(self):
        '''
        Return a dictionary of initial guesses for the battery subsystem.

        Returns
        -------
        initial_guesses : dict
            A dictionary where the keys are the names of the initial guesses
            and the values are dictionaries with any additional keyword
            arguments required by OpenMDAO for the variable.
        '''

        initial_guess_dict = {
            Mission.Battery.STATE_OF_CHARGE: {
                'val': 1.0,
                'type': 'state',
            },
            Mission.Battery.VOLTAGE_THEVENIN: {
                'val': 5.0,
                'units': 'V',
                'type': 'state',
            },
        }

        return initial_guess_dict

    def get_mass_names(self):
        '''
        Return a list of names for the battery subsystem.

        Returns
        -------
        mass_names : list
            A list of names for the battery subsystem.
        '''
        return [Aircraft.Battery.MASS]

    def preprocess_inputs(self, aviary_inputs):
        '''
        Preprocess the inputs for the battery subsystem.

        Description
        -----------
        This method preprocesses the inputs for the battery subsystem.
        In this case, it sets the values battery performance based on the battery cell type.
        '''
        battery_cell_info_18650 = {
            Aircraft.Battery.Cell.DISCHARGE_RATE: [2.0, 'A'],
            Aircraft.Battery.Cell.ENERGY_CAPACITY_MAX: [3.5, 'A*h'],
            Aircraft.Battery.Cell.HEAT_CAPACITY: [1020.0, 'J/(kg*K)'],
            Aircraft.Battery.Cell.MASS: [0.045, 'kg'],
            Aircraft.Battery.Cell.VOLTAGE_LOW: [2.9, 'V'],
            Aircraft.Battery.Cell.VOLUME: [1.125, 'inch**3']
        }

        for key, val in battery_cell_info_18650.items():
            aviary_inputs.set_val(key, val[0], units=val[1], meta_data=ExtendedMetaData)

        return aviary_inputs

    def get_outputs(self):
        '''
        Return a list of output names for the battery subsystem.

        Returns
        -------
        outputs : list
            A list of variable names for the battery subsystem.
        '''

        return [Mission.Battery.VOLTAGE, Mission.Battery.HEAT_OUT, Aircraft.Battery.EFFICIENCY]
