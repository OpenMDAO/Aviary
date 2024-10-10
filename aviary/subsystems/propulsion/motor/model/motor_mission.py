import numpy as np

import openmdao.api as om
from aviary.utils.aviary_values import AviaryValues

from aviary.variable_info.variables import Dynamic, Aircraft
from aviary.subsystems.propulsion.motor.model.motor_map import MotorMap


class MotorMission(om.Group):

    '''
    Calculates the mission performance (ODE) of a single electric motor.
    '''

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            'aviary_inputs', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )
        self.name = 'motor_mission'

    def setup(self):
        nn = self.options["num_nodes"]

        ivc = om.IndepVarComp()
        ivc.add_output('max_throttle', val=np.ones(nn), units='unitless')

        self.add_subsystem('ivc', ivc, promotes=['*'])

        motor_group = om.Group()

        motor_group.add_subsystem(
            'motor_map',
            MotorMap(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Mission.THROTTLE,
                Aircraft.Engine.SCALE_FACTOR,
                Dynamic.Mission.RPM,
            ],
            promotes_outputs=[
                Dynamic.Mission.TORQUE,
                'motor_efficiency',
            ],
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
            promotes_inputs=[('RPM', Dynamic.Mission.RPM)],
            promotes_outputs=[('shaft_power', Dynamic.Mission.SHAFT_POWER)],
        )

        # Can't promote torque as an input, as it will create a feedback loop with
        # propulsion mux component. Connect it here instead
        motor_group.connect(Dynamic.Mission.TORQUE, 'power_comp.torque')

        motor_group.add_subsystem(
            'energy_comp',
            om.ExecComp(
                'power_elec = shaft_power / efficiency',
                shaft_power={'val': np.ones(nn), 'units': 'kW'},
                power_elec={'val': np.ones(nn), 'units': 'kW'},
                efficiency={'val': np.ones(nn), 'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes_inputs=[('efficiency', 'motor_efficiency')],
            promotes_outputs=[('power_elec', Dynamic.Mission.ELECTRIC_POWER_IN)],
        )

        # Can't promote shaft power as an input, as it will create a feedback loop with
        # propulsion mux component. Connect it here instead
        motor_group.connect(Dynamic.Mission.SHAFT_POWER, 'energy_comp.shaft_power')

        self.add_subsystem('motor_group', motor_group,
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        # Determine the maximum power available at this flight condition
        # this is used for excess power constraints
        motor_group_max = om.Group()

        # these two groups are the same as those above
        motor_group_max.add_subsystem(
            'motor_map_max',
            MotorMap(num_nodes=nn),
            promotes_inputs=[
                (Dynamic.Mission.THROTTLE, 'max_throttle'),
                Aircraft.Engine.SCALE_FACTOR,
                Dynamic.Mission.RPM,
            ],
            promotes_outputs=[
                (Dynamic.Mission.TORQUE, Dynamic.Mission.TORQUE_MAX),
                'motor_efficiency',
            ],
        )

        motor_group_max.add_subsystem(
            'power_comp_max',
            om.ExecComp(
                'max_power = max_torque * pi * RPM / 30',
                max_power={'val': np.ones(nn), 'units': 'kW'},
                max_torque={'val': np.ones(nn), 'units': 'kN*m'},
                RPM={'val': np.ones(nn), 'units': 'rpm'},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('max_torque', Dynamic.Mission.TORQUE_MAX),
                ('RPM', Dynamic.Mission.RPM),
            ],
            promotes_outputs=[('max_power', Dynamic.Mission.SHAFT_POWER_MAX)],
        )

        self.add_subsystem(
            'motor_group_max',
            motor_group_max,
            promotes_inputs=['*', 'max_throttle'],
            promotes_outputs=[
                Dynamic.Mission.SHAFT_POWER_MAX,
                Dynamic.Mission.TORQUE_MAX,
            ],
        )

        self.set_input_defaults(Dynamic.Mission.RPM, val=np.ones(nn), units='rpm')
