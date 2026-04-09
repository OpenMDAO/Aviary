import numpy as np
import openmdao.api as om
from aviary.utils.csv_data_file import read_data_file
from aviary.utils.named_values import get_items
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.functions import get_path


class MotorMap(om.Group):
    """
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
        add_aviary_option(self, Aircraft.Engine.Motor.DATA_FILE)

    def setup(self):
        n = self.options['num_nodes']
        motor_model = get_path(self.options[Aircraft.Engine.Motor.DATA_FILE])

        # Read the CSV file
        # Data must be on a regular, structured, grid
        (
            read_data,
            _,
            _,
        ) = read_data_file(motor_model)

        # Extract data with units using get_items
        data_dict = {}
        units_dict = {}
        for name, (values, units) in get_items(read_data):
            data_dict[name] = values
            units_dict[name] = units

        # Reshape to 2D array
        rotations_per_minute = np.unique(data_dict['rotations_per_minute'])
        rpm_units = units_dict['rotations_per_minute']
        torque_unscaled = np.unique(data_dict['torque_unscaled'])
        torque_max = np.max(torque_unscaled)
        torque_units = units_dict['torque_unscaled']
        efficiency = (
            data_dict['efficiency'].reshape(len(torque_unscaled), len(rotations_per_minute)).T
        )
        efficiency_units = units_dict['efficiency']

        motor = om.MetaModelStructuredComp(method='slinear', vec_size=n, extrapolate=True)
        motor.add_input(
            Dynamic.Vehicle.Propulsion.RPM,
            val=np.ones(n),
            training_data=rotations_per_minute,
            units=rpm_units,
        )
        motor.add_input(
            'torque_unscaled',
            val=np.ones(n),  # unscaled torque
            training_data=torque_unscaled,
            units=torque_units,
        )
        motor.add_output(
            'efficiency',
            val=np.ones(n),
            training_data=efficiency,
            units=efficiency_units,
        )

        self.add_subsystem(
            'throttle_to_torque',
            om.ExecComp(
                'torque_unscaled = torque_max * throttle',
                torque_unscaled={'val': np.ones(n), 'units': 'N*m'},
                torque_max={
                    'val': torque_max,
                    'units': 'N*m',
                },  # only used as input because we need to dynamically assign the last value of torque as the max torque
                throttle={'val': np.ones(n), 'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes=[('throttle', Dynamic.Vehicle.Propulsion.THROTTLE)],
        )

        self.add_subsystem(
            name='motor_efficiency',
            subsys=motor,
            promotes_inputs=[Dynamic.Vehicle.Propulsion.RPM],
            promotes_outputs=['efficiency'],
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
                scale_factor={
                    'val': 1.0,
                    'units': 'unitless',
                    'desc': 'Scales the motor by increasing or decreasing the maximum torque value.',
                },
                has_diag_partials=True,
            ),
            promotes=[
                ('torque', Dynamic.Vehicle.Propulsion.TORQUE),
                ('scale_factor', Aircraft.Engine.SCALE_FACTOR),
            ],
        )

        # Connect Torque
        self.connect(
            'throttle_to_torque.torque_unscaled',
            ['motor_efficiency.torque_unscaled', 'scale_motor_torque.torque_unscaled'],
        )
