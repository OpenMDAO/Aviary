from aviary.models.external_subsystems.UAV.propulsion.model.prop_premission import UAVPropPreMission
from aviary.models.external_subsystems.UAV.propulsion.model.prop_mission import UAVPropMission
from aviary.utils.aviary_values import AviaryValues
from aviary.subsystems.propulsion.engine_model import EngineModel

from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft, Dynamic
from aviary.variable_info.variables import Mission

""" Builder for the UAV Propulsion Subsystem (RC Electric) """


class UAVBuilder(EngineModel):
    # compute_max_values is set to True to prevent a rein-instantiation of the whole 
    # propulsion model and all it's constraints. We don't need to calculate an excess 
    # power constraint for this model so not knowing what the max thrust could be is alright. 
    compute_max_values = True

    def __init__(self, options: AviaryValues = None, name='rc_electric'):
        """Initializes the PropellerBuilder object with a given name."""
        # aviary_inputs = AviaryValues()
        super().__init__(name, options)

    def build_pre_mission(self, aviary_inputs, **kwargs):  # m, b,
        """Builds an OpenMDAO system for the pre-mission computations of the subsystem."""
        return UAVPropPreMission(aviary_options=self.options)

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        """Builds an OpenMDAO system for the mission computations of the subsystem."""

        return UAVPropMission(num_nodes=num_nodes, aviary_options=self.options)

    def get_design_vars(
        self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None
    ):
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
                'upper': 5.0,
                # 'val': 100,
            },
            Aircraft.Engine.Motor.IDLE_CURRENT: {
                'units': 'A',
                'lower': 0.91,
                'upper': 3.6,  # TODO: this placeholder can be varied
                # 'val': 2.2,
            },
            Aircraft.Engine.Motor.MASS: {
                'units': 'lbm',
                'lower': 1.0362,  # 0.47 kg -> KV low enough to keep rpm_max in the prop grid
                'upper': 1.4330,  # 0.65 kg
            },
        }
        return DVs

    def get_parameters(
        self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None
    ):
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
            Aircraft.Battery.ENERGY_CAPACITY: {
                'val': 0.0,
                'units': 'W*h',
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

    def get_states(
        self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None
    ):
        states = {
            'energy_used': {
                'units': 'W*h',
                'rate_source': Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
                'targets': 'energy_used',
                'val': 0.0,
                'fix_initial': True,
                'lower': 0.0,
                'ref': 100.0,
            },
        }

        return states

    def get_controls(
        self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_name=None
    ):
        controls = {
            # Rpm slack variable the optimizer chooses to keep the propeller RPM within the bounds of the training data. The motor RPM is forced to match this value at the optimum.
            'rpm_slack': {
                'targets': 'rpm_slack',
                'units': 'rpm',
                'opt': True,
                'lower': 2800,
                'upper': 10800,
                'ref': 10800,
                'continuity_ref': 10800,
                'rate_continuity_ref': 10800,
            },
        }

        # Solver mode computes current/current_max internally in UAVPropMission.
        # Declaring them as Dymos controls creates duplicate connections.
        return controls

    def needs_mission_solver(
        self, aviary_inputs=None, user_options=None, subsystem_options=None, **kwargs
    ):
        return False

    def get_mass_names(
        self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None
    ):
        return [Aircraft.Battery.MASS, Aircraft.Engine.Motor.MASS]  # , Aircraft.Engine.MASS]

    # TODO add new outputs
    def mission_outputs(
        self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None
    ):
        return ['*']
