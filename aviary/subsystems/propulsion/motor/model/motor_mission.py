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
        n = self.options["num_nodes"]

        ivc = om.IndepVarComp()

        # TODO Remove this once engines no longer require all outputs
        # this is an artifact of allowing the motor and turbine engine
        # to swap easily between the two.
        ivc.add_output(Dynamic.Mission.THRUST, val=np.zeros(n), units='N')
        ivc.add_output(Dynamic.Mission.THRUST_MAX, val=np.zeros(n), units='N')
        ivc.add_output(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                       val=np.zeros(n), units='kg/s')
        ivc.add_output(Dynamic.Mission.NOX_RATE, val=np.zeros(n), units='kg/s')

        ivc.add_output('max_throttle', val=np.ones(n), units='unitless')

        self.add_subsystem('ivc', ivc, promotes=['*'])

        motor_group = om.Group()

        motor_group.add_subsystem('motor_map', MotorMap(num_nodes=n),
                                  promotes_inputs=[Dynamic.Mission.THROTTLE,
                                                   Aircraft.Engine.SCALE_FACTOR,
                                                   Dynamic.Mission.RPM],
                                  promotes_outputs=[(Dynamic.Mission.TORQUE, 'motor_torque'),
                                                    'motor_efficiency'])

        motor_group.add_subsystem('power_comp',
                                  om.ExecComp('shaft_power = torque * pi * RPM / 30',
                                              shaft_power={
                                                  'val': np.ones(n), 'units': 'kW'},
                                              torque={'val': np.ones(
                                                  n), 'units': 'kN*m'},
                                              RPM={'val': np.ones(n), 'units': 'rpm'}),  # fixed RPM system
                                  promotes_inputs=[('torque', 'motor_torque'),
                                                   ('RPM', Dynamic.Mission.RPM)],
                                  promotes_outputs=[('shaft_power', Dynamic.Mission.SHAFT_POWER)])

        motor_group.add_subsystem('energy_comp',
                                  om.ExecComp('power_elec = shaft_power / efficiency',
                                              shaft_power={
                                                  'val': np.ones(n), 'units': 'kW'},
                                              power_elec={'val': np.ones(
                                                  n), 'units': 'kW'},
                                              efficiency={'val': np.ones(n), 'units': 'unitless'}),
                                  promotes_inputs=[
                                      #   ('shaft_power', Dynamic.Mission.SHAFT_POWER),
                                      ('efficiency', 'motor_efficiency')],
                                  promotes_outputs=[('power_elec', Dynamic.Mission.ELECTRIC_POWER_IN)])

        motor_group.connect(Dynamic.Mission.SHAFT_POWER, 'energy_comp.shaft_power')

        # TODO Gearbox needs to be its own component separate from motor
        # is this needed???
        # this may be already covered by throttle constraints being from 0 - 1
        # motor_group.add_subsystem('torque_con',
        #                           om.ExecComp('torque_con = torque_max - torque_mission',
        #                                       torque_con={'val': np.ones(
        #                                           n), 'units': 'kN*m'},
        #                                       torque_max={'val': 1.0, 'units': 'kN*m'},
        #                                       torque_mission={'val': np.ones(n), 'units': 'kN*m'}),
        #                           promotes_inputs=[('torque_mission', Dynamic.Mission.TORQUE),
        #                                            ('torque_max', Aircraft.Motor.TORQUE_MAX)],
        #                           promotes_outputs=[('torque_con', Dynamic.Mission.Motor.TORQUE_CON)])

        self.add_subsystem('motor_group', motor_group,
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        # Determine the maximum power available at this flight condition
        # this is used for excess power constraints
        motor_group_max = om.Group()

        # these two groups are the same as those above
        motor_group_max.add_subsystem('motor_map_max', MotorMap(num_nodes=n),
                                      promotes_inputs=[(Dynamic.Mission.THROTTLE, 'max_throttle'),
                                                       Aircraft.Engine.SCALE_FACTOR,
                                                       Dynamic.Mission.RPM],
                                      promotes_outputs=[(Dynamic.Mission.TORQUE, 'motor_max_torque'),
                                                        'motor_efficiency'])

        motor_group_max.add_subsystem('power_comp_max',
                                      om.ExecComp('max_power = max_torque * pi * RPM / 30',
                                                  max_power={'val': np.ones(
                                                      n), 'units': 'kW'},
                                                  max_torque={'val': np.ones(
                                                      n), 'units': 'kN*m'},
                                                  RPM={'val': np.ones(n), 'units': 'rpm'}),
                                      promotes_inputs=[('max_torque', Aircraft.Engine.Motor.TORQUE_MAX),
                                                       ('RPM', Dynamic.Mission.RPM)],
                                      promotes_outputs=[('max_power', Dynamic.Mission.SHAFT_POWER_MAX)])

        self.add_subsystem('motor_group_max', motor_group_max,
                           promotes_inputs=['*', 'max_throttle'],
                           promotes_outputs=[Dynamic.Mission.SHAFT_POWER_MAX])

        # TODO Gearbox needs to be its own component separate from motor
        # Hamilton Standard model does not utilize torque. This can be added back in if
        # future prop models desire torque (need to also add support for torque from
        # turboshaft engine decks)
        # # determine torque available at the prop
        # self.add_subsystem('gearbox_comp',
        #                    om.ExecComp('torque = shaft_power / (pi * RPM) * 30',
        #                                shaft_power={'val': np.ones(n), 'units': 'kW'},
        #                                torque={'val': np.ones(n), 'units': 'kN*m'},
        #                                RPM={'val': np.ones(n), 'units': 'rpm'}),
        #                    promotes_inputs=[('shaft_power', Dynamic.Mission.SHAFT_POWER),
        #                                     ('RPM', Dynamic.Mission.RPM)],
        #                    promotes_outputs=[('torque', Dynamic.Mission.TORQUE),])
