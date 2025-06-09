from aviary.subsystems.propulsion.propeller.propeller_performance import PropellerPerformance
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.named_values import NamedValues
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class PropellerBuilder(SubsystemBuilderBase):
    """
    Define the builder for a propeller model using the Hamilton Standard methodology that provides
    methods to define the propeller subsystem's states, design variables, fixed values, initial
    guesses, and mass names. It also provides methods to build OpenMDAO systems for the pre-mission
    and mission computations of the subsystem, to get the constraints for the subsystem, and to
    preprocess the inputs for
    the subsystem.

    Attributes
    ----------
    name : str ('propeller')
        Object label.
    data : NamedVaues (<empty>), optional
        Propeller performance data (optional). If provided, used instead of tabular data file
        (Aircraft.Engine.Propeller.DATA_FILE).
    """

    def __init__(self, name='propeller', data: NamedValues = None):
        """Initializes the PropellerBuilder object with a given name."""
        super().__init__(name)
        self.data = data

    def build_pre_mission(self, aviary_inputs):
        """Builds an OpenMDAO system for the pre-mission computations of the subsystem."""
        return

    def build_mission(self, num_nodes, aviary_inputs):
        """Builds an OpenMDAO system for the mission computations of the subsystem."""
        return PropellerPerformance(
            num_nodes=num_nodes, aviary_options=aviary_inputs, propeller_data=self.data
        )

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
        DVs = {
            Aircraft.Engine.Propeller.ACTIVITY_FACTOR: {
                'units': 'unitless',
                'lower': 100,
                'upper': 200,
                # 'val': 100,  # initial value
            },
            Aircraft.Engine.Propeller.DIAMETER: {
                'units': 'ft',
                'lower': 0.0,
                'upper': None,
                # 'val': 8,  # initial value
            },
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT: {
                'units': 'unitless',
                'lower': 0.0,
                'upper': 0.5,
                # 'val': 0.5,
            },
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
        parameters = {
            Aircraft.Engine.Propeller.TIP_MACH_MAX: {
                'val': 1.0,
                'units': 'unitless',
            },
            Aircraft.Engine.Propeller.TIP_SPEED_MAX: {
                'val': 0.0,
                'units': 'ft/s',
            },
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT: {
                'val': 0.0,
                'units': 'unitless',
            },
            Aircraft.Engine.Propeller.ACTIVITY_FACTOR: {
                'val': 0.0,
                'units': 'unitless',
            },
            Aircraft.Engine.Propeller.DIAMETER: {
                'val': 0.0,
                'units': 'ft',
            },
            Aircraft.Nacelle.AVG_DIAMETER: {
                'val': 0.0,
                'units': 'ft',
            },
        }

        return parameters

    def get_mass_names(self):
        return [Aircraft.Engine.Gearbox.MASS]

    def get_outputs(self):
        return [
            Dynamic.Vehicle.Propulsion.SHAFT_POWER + '_out',
            Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX + '_out',
            Dynamic.Vehicle.Propulsion.RPM + '_out',
            Dynamic.Vehicle.Propulsion.TORQUE + '_out',
            Mission.Constraints.GEARBOX_SHAFT_POWER_RESIDUAL,
        ]
