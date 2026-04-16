import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.motor.model.motor_map import MotorMap
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic


class MotorMission(om.Group):
    """Calculates the mission performance (ODE) of a single electric motor."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        add_aviary_option(self, Aircraft.Engine.RPM_DESIGN, units='rpm')
        add_aviary_option(self, Aircraft.Engine.FIXED_RPM, val=0.0, units='rpm')

    def setup(self):
        nn = self.options['num_nodes']
        rpm_design = self.options[Aircraft.Engine.RPM_DESIGN][0]
        fixed_rpm = self.options[Aircraft.Engine.FIXED_RPM][0]

        ivc = om.IndepVarComp()
        ivc.add_output('max_throttle', val=np.ones(nn), units='unitless')
        ivc.add_output('max_RPM', val=np.ones(nn) * rpm_design, units='rpm')

        self.add_subsystem('ivc', ivc, promotes=['*'])

        motor_group = om.Group()

        # NOTE: this relies on the option default for this group for FIXED_RPM being 0.0
        if fixed_rpm != 0.0:
            use_fixed_rpm = True
        else:
            use_fixed_rpm = False

        if not use_fixed_rpm:
            # Adjust RPM with throttle. This is because motor model does not capture the physics of RPM
            # with changing torque and shaft power, and prevents very high RPM with little to no motor
            # power (aka free energy spinning the shaft)
            motor_group.add_subsystem(
                'rpm_calc',
                om.ExecComp(
                    'RPM = max_RPM * throttle',
                    RPM={'val': np.zeros(nn), 'units': 'rpm'},
                    max_RPM={'val': np.zeros(nn), 'units': 'rpm'},
                    throttle={'val': np.zeros(nn), 'units': 'unitless'},
                ),
                promotes_inputs=[('throttle', Dynamic.Vehicle.Propulsion.THROTTLE), 'max_RPM'],
                promotes_outputs=[('RPM', Dynamic.Vehicle.Propulsion.RPM)],
            )

        motor_group.add_subsystem(
            'motor_map',
            MotorMap(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.THROTTLE,
                Aircraft.Engine.SCALE_FACTOR,
                # Dynamic.Vehicle.Propulsion.RPM,
            ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.TORQUE,
                'efficiency',
            ],
        )

        if use_fixed_rpm:
            motor_group.promotes('motor_map', inputs=[Dynamic.Vehicle.Propulsion.RPM])
        else:
            motor_group.connect(
                Dynamic.Vehicle.Propulsion.RPM, f'motor_map.{Dynamic.Vehicle.Propulsion.RPM}'
            )

        motor_group.add_subsystem(
            'power_comp',
            om.ExecComp(
                'shaft_power = torque * RPM',
                shaft_power={'val': np.ones(nn), 'units': 'kW'},
                torque={'val': np.ones(nn), 'units': 'kN*m'},
                RPM={'val': np.ones(nn), 'units': 'rad/s'},
                has_diag_partials=True,
            ),  # fixed RPM system
            # promotes_inputs=[('RPM', Dynamic.Vehicle.Propulsion.RPM)],
            promotes_outputs=[('shaft_power', Dynamic.Vehicle.Propulsion.SHAFT_POWER)],
        )

        if use_fixed_rpm:
            motor_group.promotes('power_comp', inputs=[('RPM', Dynamic.Vehicle.Propulsion.RPM)])
        else:
            motor_group.connect(Dynamic.Vehicle.Propulsion.RPM, 'power_comp.RPM')

        # Can't promote torque as an input, as it will create a feedback loop with
        # propulsion mux component. Connect it here instead
        motor_group.connect(Dynamic.Vehicle.Propulsion.TORQUE, 'power_comp.torque')

        motor_group.add_subsystem(
            'energy_comp',
            om.ExecComp(
                'power_elec = shaft_power / efficiency',
                shaft_power={'val': np.ones(nn), 'units': 'kW'},
                power_elec={'val': np.ones(nn), 'units': 'kW'},
                efficiency={'val': np.ones(nn), 'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes_inputs=['efficiency'],
            promotes_outputs=[('power_elec', Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN)],
        )

        # Can't promote shaft power as an input, as it will create a feedback loop with
        # propulsion mux component. Connect it here instead
        motor_group.connect(Dynamic.Vehicle.Propulsion.SHAFT_POWER, 'energy_comp.shaft_power')

        self.add_subsystem(
            'motor_group', motor_group, promotes_inputs=['*'], promotes_outputs=['*']
        )

        # Determine the maximum power available at this flight condition
        # this is used for excess power constraints
        motor_group_max = om.Group()

        # these two groups are the same as those above
        motor_group_max.add_subsystem(
            'motor_map_max',
            MotorMap(num_nodes=nn),
            promotes_inputs=[
                (Dynamic.Vehicle.Propulsion.THROTTLE, 'max_throttle'),
                Aircraft.Engine.SCALE_FACTOR,
                (Dynamic.Vehicle.Propulsion.RPM, 'max_RPM'),
            ],
            promotes_outputs=[
                (
                    Dynamic.Vehicle.Propulsion.TORQUE,
                    Dynamic.Vehicle.Propulsion.TORQUE_MAX,
                ),
                'efficiency',
            ],
        )

        motor_group_max.add_subsystem(
            'power_comp_max',
            om.ExecComp(
                'max_power = max_torque * pi * max_RPM / 30',
                max_power={'val': np.ones(nn), 'units': 'kW'},
                max_torque={'val': np.ones(nn), 'units': 'kN*m'},
                max_RPM={'val': np.ones(nn), 'units': 'rpm'},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('max_torque', Dynamic.Vehicle.Propulsion.TORQUE_MAX),
                'max_RPM',
            ],
            promotes_outputs=[('max_power', Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX)],
        )

        self.add_subsystem(
            'motor_group_max',
            motor_group_max,
            promotes_inputs=['*', 'max_throttle'],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX,
                Dynamic.Vehicle.Propulsion.TORQUE_MAX,
            ],
        )

        # self.set_input_defaults(Dynamic.Vehicle.Propulsion.RPM, val=np.ones(nn), units='rpm')
