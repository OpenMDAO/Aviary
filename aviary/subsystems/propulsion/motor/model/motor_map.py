import numpy as np
import openmdao.api as om
from aviary.utils.csv_data_file import read_data_file
from aviary.utils.named_values import get_items

from aviary.variable_info.variables import Aircraft, Dynamic


class MotorMap(om.Group):
    """
    This function takes in 0-1 values for electric motor throttle,
    scales those values into 0 to max_torque on the motor map
    this also allows us to solve for motor efficiency
    then we scale the torque up based on the actual scale factor of the motor.
    This avoids the need to rescale the map values, and still allows for the motor scale to be optimized.
    Scaling only effects torque (and therefore shaft power production, and electric power consumption).
    RPM is not scaled and is assumed to be maxed at 6,000 rpm.
    The original maps were put together for a 746kw (1,000 hp) electric motor published in the TTBW paper:
    https://ntrs.nasa.gov/api/citations/20230016987/downloads/TTBW_SciTech_2024_Final_12_5_2023.pdf
    The map is shown in Figure 4.

    Inputs
    ----------
    Dynamic.Vehicle.Propulsion.THROTTLE : float (unitless) (0 to 1)
        The throttle command which will be translated into torque output from the engine
    Aircraft.Engine.SCALE_FACTOR : float (unitless) (positive)
    Aircraft.Motor.RPM : float (rpm) (0 to 6000)

    Outputs
    ----------
    Dynamic.Vehicle.Propulsion.TORQUE : float (positive)
    Dynamic.Mission.Motor.EFFICIENCY : float (positive)

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('motor_model', types=str) # default='aviary/models/motors/electric_motor_1800Nm_6000rpm.csv'

    def setup(self):
        n = self.options['num_nodes']
        motor_model = self.options['motor_model']

        # Read the CSV file
        # Data must be on a regular, structured, grid
        read_data, _, _, = read_data_file(motor_model)

        # Extract data with units using get_items
        data_dict = {}
        units_dict = {}
        for name, (values, units) in get_items(read_data):
            data_dict[name] = values
            units_dict[name] = units

        # Reshape to 2D array
        rpm_vals = np.unique(data_dict['rpm_vals'])
        rpm_units= units_dict['rpm_vals']
        torque_vals = np.unique(data_dict['torque_vals'])
        torque_units = units_dict['torque_vals']
        efficiency = data_dict['efficiency'].reshape(len(torque_vals), len(rpm_vals)).T
        efficiency_units = units_dict['efficiency']

        motor = om.MetaModelStructuredComp(method='slinear', vec_size=n, extrapolate=False)
        motor.add_input(
            Dynamic.Vehicle.Propulsion.RPM,
            val=np.ones(n),
            training_data=rpm_vals,
            units=rpm_units,
        )
        motor.add_input(
            'torque_unscaled',
            val=np.ones(n),  # unscaled torque
            training_data=torque_vals,
            units=torque_units,
        )
        motor.add_output(
            'motor_efficiency',
            val=np.ones(n),
            training_data=efficiency,
            units=efficiency_units,
        )

        self.add_subsystem(
            'throttle_to_torque',
            om.ExecComp(
                'torque_unscaled = torque_max * throttle',
                torque_unscaled={'val': np.ones(n), 'units': 'N*m'},
                torque_max={'val': torque_vals[-1], 'units': 'N*m'}, # only used as input because we need to dynamically assign the last value of torque as the max torque
                throttle={'val': np.ones(n), 'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes=[('throttle', Dynamic.Vehicle.Propulsion.THROTTLE)],
        )

        self.add_subsystem(
            name='motor_efficiency',
            subsys=motor,
            promotes_inputs=[Dynamic.Vehicle.Propulsion.RPM],
            promotes_outputs=['motor_efficiency'],
        )

        # Now that we know the efficiency, scale up the torque correctly for the engine
        #   size selected
        # Note: This allows the optimizer to optimize the motor size if desired
        self.add_subsystem(
            'scale_motor_torque',
            om.ExecComp(
                'torque = torque_unscaled * scale_factor',
                torque={'val': np.ones(n), 'units': 'N*m'},
                torque_unscaled={'val': np.ones(n), 'units': 'N*m'},
                scale_factor={'val': 1.0, 'units': 'unitless', 'desc':'Scales the motor by increasing or decreasing the maximum torque value.'},
                has_diag_partials=True,
            ),
            promotes=[
                ('torque', Dynamic.Vehicle.Propulsion.TORQUE),
                ('scale_factor', Aircraft.Engine.SCALE_FACTOR),
            ],
        )

        self.connect(
            'throttle_to_torque.torque_unscaled',
            ['motor_efficiency.torque_unscaled', 'scale_motor_torque.torque_unscaled'],
        )
