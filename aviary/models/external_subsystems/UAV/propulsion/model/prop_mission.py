import numpy as np
import openmdao.api as om

from aviary.models.external_subsystems.UAV.propulsion.model.prop_performance import (
    Throttle,
    Battery,
    ElectronicSpeedController,
    Motor,
    PropCoefficients,
    Propeller,
    Vectorization,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft, Dynamic


class UAVPropMission(om.Group):
    """Calculates the mission performance (ODE) of a single electric RCMotor."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )

    def setup(self):
        nn = self.options['num_nodes']

        # constraint ties the motor to the prop load; in solver mode the solver does.
        motor_load_factor = 1.0

        rpm_in = [(Dynamic.Vehicle.Propulsion.RPM, 'rpm_slack')]
        self.set_input_defaults('rpm_slack', val=np.ones(nn) * 60.0, units='rev/s')

        self.add_subsystem(
            'throttle',
            Throttle(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.THROTTLE,
            ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.CURRENT,
            ],
        )

        self.add_subsystem(
            'battery',
            Battery(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Battery.VOLTAGE,
                Aircraft.Battery.RESISTANCE,
                Dynamic.Vehicle.Propulsion.CURRENT,
            ],
            promotes_outputs=[
                ('dt_denergy_used', Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN),
            ],
        )

        self.add_subsystem(
            'esc',
            ElectronicSpeedController(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.THROTTLE,
                Dynamic.Vehicle.Propulsion.CURRENT,
            ],
        )

        self.add_subsystem(
            'motor',
            Motor(num_nodes=nn, load_factor=motor_load_factor),
            promotes_inputs=[
                Aircraft.Engine.Motor.IDLE_CURRENT,
                Aircraft.Engine.Motor.RESISTANCE,
                Aircraft.Engine.Motor.KV,
                Dynamic.Vehicle.Propulsion.CURRENT,
            ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.RPM,
            ],
        )

        self.add_subsystem(
            'vectorize_geo',
            Vectorization(num_nodes=nn),
            promotes_inputs=[Aircraft.Engine.Propeller.DIAMETER, Aircraft.Engine.Propeller.PITCH],
            promotes_outputs=['temp_diameter', 'temp_pitch'],
        )

        self.add_subsystem(
            'propco',
            # akima would be smoother but needs >=5 points/dim; this table has 3.
            PropCoefficients(
                method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn
            ),
            promotes_inputs=[
                Dynamic.Mission.VELOCITY,
                'temp_diameter',
                'temp_pitch',
            ]
            + rpm_in,
            promotes_outputs=['ct', 'cp'],
        )

        self.add_subsystem(
            'prop',
            Propeller(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Engine.Propeller.DIAMETER,
                'ct',
                'cp',
                Dynamic.Atmosphere.DENSITY,
            ]
            + rpm_in,
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER,
                Dynamic.Vehicle.Propulsion.THRUST,
            ],
        )

        self.add_subsystem(
            'rpm_balance',
            om.ExecComp(
                'rpm_defect = rpm_slack - rpm_motor',
                rpm_defect={'val': np.zeros(nn), 'units': 'rev/s'},
                rpm_slack={'val': np.zeros(nn), 'units': 'rev/s'},
                rpm_motor={'val': np.zeros(nn), 'units': 'rev/s'},
                has_diag_partials=True,
            ),
            promotes_inputs=['rpm_slack'],
        )
        self.connect(Dynamic.Vehicle.Propulsion.RPM, 'rpm_balance.rpm_motor')

        self.add_subsystem(
            'energy_con',
            om.ExecComp(
                'energy_constraint = energy_capacity-energy_used',
                energy_constraint={'val': np.zeros(nn), 'units': 'W*h'},
                energy_capacity={'val': 1.0, 'units': 'W*h'},
                energy_used={'val': np.zeros(nn), 'units': 'W*h'},
                has_diag_partials=True,
            ),
            promotes_inputs=[('energy_capacity', Aircraft.Battery.ENERGY_CAPACITY), 'energy_used'],
            promotes_outputs=['energy_constraint'],
        )

        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_out', 'motor.current')

        """Constraints"""
        # ref 400: d(defect)/d(rpm_slack)=1 rpm/rpm with rpm_slack ref 10800 made
        # this row ~270 in the scaled jacobian at ref=40
        self.add_constraint('rpm_balance.rpm_defect', equals=0.0, ref=400, units='rpm')

        self.add_constraint('energy_constraint', lower=0.0, indices=[-1], ref=100, units='W*h')

        self.options['auto_order'] = True
