from aviary.subsystems.propulsion.rc_electric.model.UAV_premission import RCPropPreMission
from aviary.subsystems.propulsion.rc_electric.model.UAV_mission import RCPropMission
from aviary.utils.aviary_values import AviaryValues
from aviary.subsystems.propulsion.engine_model import EngineModel

from aviary.variable_info.dbf_variables import Aircraft, Dynamic
from aviary.variable_info.variables import Mission

""" Builder for the UAV Propulsion Subsystem (RC Electric) """

class RCBuilder(EngineModel):
    # RCPropMission computes its own max-power chain (battery_max ... prop_max),
    # so tell Aviary NOT to build a duplicate full-throttle copy of the engine.
    # The duplicate re-declared every constraint and broke the optimizer (TOO_FEW_DOF).
    compute_max_values =True

    def __init__(self, options: AviaryValues = None, name='rc_electric', power_balance_mode='feedforward'):
        """Initializes the PropellerBuilder object with a given name."""
        # aviary_inputs = AviaryValues()
        super().__init__(name, options)

        self.power_balance_mode = power_balance_mode
    def build_pre_mission(self, aviary_inputs, **kwargs):  # m, b,
        """Builds an OpenMDAO system for the pre-mission computations of the subsystem."""
        return RCPropPreMission(aviary_options=self.options)

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        """Builds an OpenMDAO system for the mission computations of the subsystem."""


        return RCPropMission(num_nodes=num_nodes, aviary_options=self.options, power_balance_mode=self.power_balance_mode)
    # def get_constraints(self):
    #     constraints = {
    #         Dynamic.Vehicle.Propulsion.CURRENT: {
    #             'lower': 0,
    #             'type': 'path',
    #         },
    #         Dynamic.Vehicle.Propulsion.CURRENT_CON: {
    #             'upper': 0, 
    #             'type': 'path',1
    #         }1
    #     }1

    #     return constraints

    def get_design_vars(self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None):
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
        # TODO Alex bounds are rough placeholders
        # TODO Alex potentially work on optimizing the voltage
        DVs = {
            Aircraft.Battery.MASS: {
                'units': 'kg',
                'lower': 0.1,
                'upper': 1.0,
                # 'val': 100,  
            },
            Aircraft.Engine.Motor.IDLE_CURRENT: {
                'units': 'A',
                'lower': 0.91,
                'upper': 3.6, #TODO: this placeholder can be varied
                # 'val': 2.2,  
            },
           
            



        
            Aircraft.Engine.Motor.MASS: {
               
                'units': 'lbm',
                'lower': 1.0362,   # 0.47 kg -> KV low enough to keep rpm_max in the prop grid
                'upper': 1.4330,   # 0.65 kg
            },
            # Aircraft.Engine.Propeller.PITCH: {
            #     'units': 'inch',
            #     'lower': 3.0,
            #     'upper': 15.0,
            #     # 'val': 100,  # initial value
            # },
            # Aircraft.Engine.Propeller.DIAMETER: {
            #     'units': 'inch',
            #     'lower': 10.0,
            #     'upper': 20.0,
            #     # 'val': 8,  # initial value
            # },

        }
        return DVs

    def get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None):
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

       
       

        parameters = {
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

            #Single value for the motor's maximum continuous current, this cannot change during the mission, but can be optimized as a design variable.
            Aircraft.Engine.Motor.MAX_CONT_CURRENT: {
                'val': 100,  
                'units': 'A',
            },
            Aircraft.Engine.Propeller.DIAMETER: {
                'val': 19,
                'units': 'inch',
            },
            Aircraft.Engine.Propeller.PITCH: {
                'val': 12,
                'units': 'inch',
            },
        }

        return parameters

    def get_controls(self, aviary_inputs = None, user_options = None, subsystem_options = None, phase_name=None):

        if self.power_balance_mode == 'feedforward':
           controls = {Dynamic.Vehicle.Propulsion.CURRENT: {
                'targets': Dynamic.Vehicle.Propulsion.CURRENT,
                'units': 'A',
                'opt': True,
                # 1 A floor, not 10: at light cruise the balanced current is only a
                # few amps; a 10 A floor forces ~250 W through the powertrain and
                # makes the thrust=drag constraint infeasible (IPOPT local infeas).
                'lower': 1.0,
                'upper': 100.0,
                'ref': 1.0e2,
            },


            # SAND: the RPM the prop table sees. Bounds keep it inside the
            # training data (16.7 - 183 rev/s), so ct/cp are never extrapolated.
            'rpm_slack': {
                'targets': 'rpm_slack',
                'units': 'rev/s',
                'opt': True,
                'lower': 20.0,
                'upper': 180.0,
                'ref': 1.0e2,
            },
            # 'rpm_slack_max': {
            #     'targets': 'rpm_slack_max',
            #     'units': 'rev/s',
            #     'opt': True,
            #     'lower': 20.0,
            #     'upper': 180.0,
            #     'ref': 1.0e2,
            # },
            }

           return controls
        # Solver mode computes current/current_max internally in RCPropMission.
        # Declaring them as Dymos controls creates duplicate connections.

        return {}
        
       
    def get_mass_names(self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None):

        
        return [Aircraft.Battery.MASS, Aircraft.Engine.Motor.MASS]#, Aircraft.Engine.MASS]
    

   




    
    #TODO add new outputs
    def mission_outputs(self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None):
        return [
            #TODO: Alex see why this is an issue 
            # Dynamic.Vehicle.Propulsion.THROTTLE,
            # Dynamic.Vehicle.Propulsion.SHAFT_POWER + '_out',
            # Dynamic.Vehicle.Propulsion.RPM + '_out',
            # Dynamic.Vehicle.Propulsion.THRUST + '_out',
        ]