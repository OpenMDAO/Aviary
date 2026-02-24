import openmdao.api as om

from aviary.subsystems.propulsion.motor.model.motor_map import MotorMap
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues

class MotorPreMission(om.Group):
    """Calculate electric motor mass for a single motor."""

    def initialize(self):
        self.name = 'motor_premission'
        self.options.declare('aviary_inputs', types=AviaryValues, default=None)

    def setup(self):
        # pull the motor model from aviary options
        motor_model = self.options['aviary_inputs'].get_val(Aircraft.Engine.Motor.DATA_FILE)

        # Determine max torque of scaled motor

        # We create a set of default inputs for this group so that in pre-mission, the
        #   group can be instantiated with only scale_factor as an input.
        # Without inputs it will return the max torque based on the non-dimensional
        #   scale factor chosen by the optimizer!
        # The max torque is then used in pre-mission to determine weight of the system.

        # max_throttle is set to 1 for the purposes of determining max_torque.
        self.set_input_defaults('max_throttle', 1.0, units=None)

        # RPM HAS NO EFFECT ON TORQUE_MAX OR MOTOR_MASS.
        # We did not want to write a second MotorMap() group just to remove the RPM inputs
        # since we need that input in Motor Mission.
        self.set_input_defaults('dummy_rpm', 0.0, units='rpm')

        self.add_subsystem(
            'motor_map',
            MotorMap(num_nodes=1, motor_model=motor_model),
            promotes_inputs=[
                Aircraft.Engine.SCALE_FACTOR,
                (Dynamic.Vehicle.Propulsion.THROTTLE, 'max_throttle'),
                (Dynamic.Vehicle.Propulsion.RPM, 'dummy_rpm'),
            ],
            promotes_outputs=[
                (Dynamic.Vehicle.Propulsion.TORQUE, Aircraft.Engine.Motor.TORQUE_MAX)
            ],
        )

        # Motor mass relationship based on continuous torque rating for aerospace motors (Figure 10)
        # Propulsion Scaling Methods in the Era of Electric Flight - Duffy et. al.
        # AIAA Propulsion and Energy Forum, July 9-11, 2018
        self.add_subsystem(
            'motor_mass',
            om.ExecComp(
                'motor_mass = 0.3151 * max_torque**(0.748)',
                motor_mass={'val': 0.0, 'units': 'kg'},
                max_torque={'val': 0.0, 'units': 'N*m'},
            ),
            promotes_inputs=[('max_torque', Aircraft.Engine.Motor.TORQUE_MAX)],
            promotes_outputs=[('motor_mass', Aircraft.Engine.Motor.MASS)],
        )
