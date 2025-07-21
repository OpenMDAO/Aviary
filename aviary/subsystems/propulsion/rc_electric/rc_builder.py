from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_premission import RCPropPreMission
from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_mission import RCPropMission
from aviary.utils.aviary_values import AviaryValues
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

class RCBuilder(SubsystemBuilderBase):
    def __init__(self, name='rc_eletric'):
        """Initializes the PropellerBuilder object with a given name."""
        aviary_inputs = AviaryValues()
        super().__init__(name)

    def build_pre_mission(self, m, b, aviary_inputs):
        """Builds an OpenMDAO system for the pre-mission computations of the subsystem."""
        return RCPropPreMission(m=m, b=b, aviary_options=aviary_inputs)

    def build_mission(num_nodes, aviary_inputs):
        """Builds an OpenMDAO system for the mission computations of the subsystem."""
        return RCPropMission(num_nodes=num_nodes, aviary_options=aviary_inputs)

    def get_design_vars(self):
        """
        Design vars are only tested to see if they exist in pre_mission
        Returns a dictionary of design variables for the gearbox subsystem, where the keys are the
        names of the design variables, and the values are dictionaries that contain the units for
        the design variable, the lower and upper bounds for the design variable, and any
        additional keyword arguments required by OpenMDAO for the design variable.

        Returns
        -------
        parameters : dict
        A dict of names for the propeller subsystem.
        """
        # TODO bounds are rough placeholders
        # TODO potentially work on optimizing the voltage
        DVs = {
            Aircraft.Battery.MASS: {
                'units': 'kg',
                'lower': 0.0,
                'upper': None,
                # 'val': 100,  
            },
            Dynamic.Vehicle.Propulsion.THROTTLE: {
                'units': 'unitless',
                'lower': 0.01,
                'upper': 1.0,
                # 'val': 0.5,  
            },
            Aircraft.Engine.Motor.IDLE_CURRENT: {
                'units': 'A',
                'lower': 1.0,
                'upper': 3.6, #TODO: this placeholder can be varied
                # 'val': 2.2,  
            },
            Aircraft.Engine.Motor.PEAK_CURRENT: {
                'units': 'A',
                'lower': 1,
                'upper': 225,
                # 'val': 100,  
            },
            Aircraft.Engine.Motor.MASS: {
                'units': 'kg',
                'lower': 0.288,
                'upper': 1.701,
                # 'val': 1.0,  
            },
            Aircraft.Engine.Propeller.PITCH: {
                'units': 'inch',
                'lower': 0.0,
                'upper': None,
                # 'val': 100,  # initial value
            },
            Aircraft.Engine.Propeller.DIAMETER: {
                'units': 'm',
                'lower': 0.0,
                'upper': None,
                # 'val': 8,  # initial value
            },

            #TODO: check if velocity is to be added here? 

        }
        return DVs

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        """
        Parameters are only tested to see if they exist in mission.
        The value doesn't change throughout the mission.
        Returns a dictionary of fixed values for the propeller subsystem, where the keys
        are the names of the fixed values, and the values are dictionaries that contain
        the fixed value for the variable, the units for the variable, and any additional
        keyword arguments required by OpenMDAO for the variable.

        Returns
        -------
        parameters : dict
        A dict of names for the propeller subsystem.
        """

        #TODO add new variables, including dvs and optional inputs
        parameters = {
            Aircraft.Battery.MASS: {
                'val': 1.0, 
                'units': 'kg',
            },
            Aircraft.Battery.VOLTAGE: {
                'val': 22.2, 
                'units': 'V',
            },
            Aircraft.Battery.RESISTANCE: {
                'val': 0.05, 
                'units': 'ohm',
            },
            Aircraft.Engine.Motor.RESISTANCE: {
                'val': 0.05,  
                'units': 'ohm',
            },
            Aircraft.Engine.Motor.KV: {
                'val': 400,  
                'units': 'rpm/V',
            },
            Aircraft.Engine.Motor.IDLE_CURRENT: {
                'val': 2.2,  
                'units': 'A',
            },
            Aircraft.Engine.Motor.PEAK_CURRENT: {
                'val': 100,  
                'units': 'A',
            },
            Aircraft.Engine.Motor.MASS: {
                'val': 0.0, 
                'units': 'kg', 
            },
            Aircraft.Engine.Propeller.DIAMETER: {
                'val': 0.0,
                'units': 'm',
            },
            Aircraft.Engine.Propeller.PITCH: {
                'val': 0.0,
                'units': 'inch',
            },
            Aircraft.Engine.NUM_ENGINES: {
                'val': 1,
                'units':'unitless',
            }
        }

        return parameters

    def get_mass_names(self):
        return [Aircraft.Battery.MASS, Aircraft.Engine.Motor.MASS]
    
    #TODO add new outputs
    def get_outputs(self):
        return [
            Dynamic.Vehicle.Propulsion.PROP_POWER + '_out',
            Dynamic.Vehicle.Propulsion.RPM + '_out',
            Dynamic.Vehicle.Propulsion.THRUST + '_out',
        ]
