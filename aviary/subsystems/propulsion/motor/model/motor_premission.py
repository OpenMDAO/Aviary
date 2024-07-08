import openmdao.api as om
import numpy as np

from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues
from aviary.subsystems.propulsion.motor.model.motor_map import MotorMap


class MotorPreMission(om.Group):
    """
    Calculate electric motor mass for a single motor
    """

    def initialize(self):
        self.options.declare(
            "aviary_inputs", types=AviaryValues,
            desc="collection of Aircraft/Mission specific options",
            default=None,
        )
        self.name = 'motor_premission'

    def setup(self):

        # Determine max torque of scaled motor

        # We create a set of default inputs for this group so that in pre-mission,
        # the group can be instantiated with only scale_factor as an input.
        # without inputs and it will return the max torque
        # based on the non-dimensional scale factor chosen by the optimizer.
        # The max torque is then used in pre-mission to determine weight of the system.
        self.set_input_defaults(Dynamic.Mission.THROTTLE, 1.0, units=None)

        self.add_subsystem('motor_map', MotorMap(num_nodes=1),
                           promotes_inputs=[Aircraft.Engine.SCALE_FACTOR,
                                            Dynamic.Mission.THROTTLE,
                                            Dynamic.Mission.RPM],
                           promotes_outputs=[(Dynamic.Mission.TORQUE,
                                              Aircraft.Engine.Motor.TORQUE_MAX)])

        # Motor mass relationship based on continuous torque rating for aerospace motors (Figure 10)
        # Propulsion Scaling Methods in the Era of Electric Flight - Duffy et. al.
        # AIAA Propulsion and Energy Forum, July 9-11, 2018
        self.add_subsystem('motor_mass',
                           om.ExecComp('motor_mass = 0.3151 * max_torque^(0.748)',
                                       motor_mass={'val': 0.0, 'units': 'kg'},
                                       max_torque={'val': 0.0, 'units': 'N*m'}),
                           promotes_inputs=[
                               ('max_torque', Aircraft.Engine.Motor.TORQUE_MAX)],
                           promotes_outputs=[('motor_mass', Aircraft.Engine.Motor.MASS)])

        # TODO Gearbox needs to be its own component separate from motor
        self.add_subsystem('power_comp',
                           om.ExecComp('power = torque * pi * RPM / 30',
                                       power={'val': 0.0, 'units': 'kW'},
                                       torque={'val': 0.0, 'units': 'kN*m'},
                                       RPM={'val': 0.0, 'units': 'rpm'}),
                           promotes_inputs=[('torque', Aircraft.Engine.Motor.TORQUE_MAX),
                                            ('RPM', Dynamic.Mission.RPM)],
                           promotes_outputs=[('power', 'shaft_power_max')])

        self.add_subsystem('gearbox_PRM',
                           om.ExecComp('RPM_out = gear_ratio * RPM_in',
                                       RPM_out={'val': 0.0, 'units': 'rpm'},
                                       gear_ratio={'val': 1.0, 'units': None},
                                       RPM_in={'val': 0.0, 'units': 'rpm'}),
                           promotes_inputs=['RPM_in',
                                            ('gear_ratio', Aircraft.Engine.Gearbox.GEAR_RATIO)],
                           promotes_outputs=['RPM_out'])

        # Gearbox mass from "An N+3 Technolgoy Level Reference Propulsion System" by Scott Jones, William Haller, and Michael Tong
        # NASA TM 2017-219501
        self.add_subsystem('gearbox_mass',
                           om.ExecComp('gearbox_mass = (power / RPM_out)^(0.75) * (RPM_in / RPM_out)^(0.15)',
                                       gearbox_mass={'val': 0.0, 'units': 'lb'},
                                       power={'val': 0.0, 'units': 'hp'},
                                       RPM_out={'val': 0.0, 'units': 'rpm'},
                                       RPM_in={'val': 0.0, 'units': 'rpm'},),
                           promotes_inputs=[('power', Dynamic.Mission.SHAFT_POWER_MAX),
                                            'RPM_out', 'RPM_in'],
                           promotes_outputs=[('gearbox_mass', Aircraft.Engine.Gearbox.MASS)])
