import openmdao.api as om
import numpy as np

from aviary.subsystems.propulsion.motor.motor_variables import Aircraft, Dynamic
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

        # TBD I'm worried that the above code won't set these in pre-mission correctly
        self.add_subsystem('motor_map', MotorMap(num_nodes=1),
                           promotes_inputs=[Aircraft.Engine.SCALE_FACTOR,
                                            Dynamic.Mission.THROTTLE,
                                            Aircraft.Motor.RPM],
                           promotes_outputs=[(Dynamic.Mission.TORQUE,
                                              Aircraft.Motor.TORQUE_MAX)])

        # Motor mass relationship based on continuous torque rating for aerospace motors (Figure 10)
        # Propulsion Scaling Methods in the Era of Electric Flight - Duffy et. al.
        # AIAA Propulsion and Energy Forum, July 9-11, 2018
        self.add_subsystem('motor_mass',
                           om.ExecComp('motor_mass = 0.3151 * max_torque^(0.748)',
                                       motor_mass={'val': 1.0, 'units': 'kg'},
                                       max_torque={'val': 1.0, 'units': 'N*m'}),
                           promotes_inputs=[('max_torque', Aircraft.Motor.TORQUE_MAX)],
                           promotes_outputs=[('motor_mass', Aircraft.Motor.MASS)])

        self.add_subsystem('power_comp',
                           om.ExecComp('P = T * pi * RPM / 30',
                                       P={'val': 1.0, 'units': 'kW'},
                                       T={'val': 1.0, 'units': 'kN*m'},
                                       RPM={'val': 1.0, 'units': 'rpm'}),
                           promotes_inputs=[("T", Aircraft.Motor.TORQUE_MAX),
                                            ("RPM", Aircraft.Motor.RPM)],
                           promotes_outputs=[("P", 'shaft_power_max')])

        self.add_subsystem('gearbox_PRM',
                           om.ExecComp('RPM_out = gear_ratio * RPM_in',
                                       RPM_out={'val': 1.0, 'units': 'rpm'},
                                       gear_ratio={'val': 1.0, 'units': None},
                                       RPM_in={'val': 1.0, 'units': 'rpm'}),
                           promotes_inputs=[('RPM_in', Aircraft.Motor.RPM),
                                            ('gear_ratio', Aircraft.Gearbox.GEAR_RATIO)],
                           promotes_outputs=[('RPM_out', Aircraft.Prop.RPM)])

        # Gearbox mass from "An N+3 Technolgoy Level Reference Propulsion System" by Schtt Jones, William Haller, and Michael Tong
        # NASA TM 2017-219501
        self.add_subsystem('gearbox_mass',
                           om.ExecComp('gearbox_mass = (P / RPM_out)^(0.75) * (RPM_in / RPM_out)^(0.15)',
                                       gearbox_mass={'val': 1.0, 'units': 'lb'},
                                       P={'val': 1.0, 'units': 'hp'},
                                       RPM_out={'val': 1.0, 'units': 'rpm'},
                                       RPM_in={'val': 1.0, 'units': 'rpm'},),
                           promotes_inputs=[("P", 'shaft_power_max'),
                                            ('RPM_out', Aircraft.Prop.RPM),
                                            ('RPM_in', Aircraft.Motor.RPM)],
                           promotes_outputs=[('gearbox_mass', Aircraft.Gearbox.MASS)])
