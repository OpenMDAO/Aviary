import openmdao.api as om

from aviary.subsystems.propulsion.rc_electric.model.rc_performance import RCPropGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic


class RCPropPreMission(om.Group):
    """Calculate electric motor mass for a single motor."""

    def initialize(self):
        self.options.declare('m', default = 1.3132, desc='m coefficient for kv(mass, peak_current): kv = m * peak_current/mass + b')
        self.options.declare('b', default = 0.01, desc='b coefficient for kv(mass, peak_current): kv = m * peak_current/mass + b')
        self.options.declare(
            'aviary_inputs',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )
        self.name = 'rcpropulsion_premission'

    def setup(self):
        # Determine max torque of scaled motor

        # We create a set of default inputs for this group so that in pre-mission, the
        #   group can be instantiated with only scale_factor as an input.
        # Without inputs it will return the max torque based on the non-dimensional
        #   scale factor chosen by the optimizer.
        # The max torque is then used in pre-mission to determine weight of the system.
        
        #TODO: CITE!
        self.add_subsystem(
            'energy_calc',
            om.ExecComp(
                #TODO: may add options here
                'energy = voltage_in * (battery_mass * 7.3 - 0.246)',
                energy={'val': 0.0, 'units': 'W*h'},
                voltage_in={'val': 0.0, 'units': 'V'},
                battery_mass={'val': 0.0, 'units': 'kg'},
            ),
            promotes_inputs=[('battery_mass', Aircraft.Battery.MASS), ('voltage_in', Aircraft.Battery.VOLTAGE)],
            promotes_outputs=[('energy', Aircraft.Battery.ENERGY_CAPACITY)],
        )

        self.add_subsystem(
            'motor_resistance_calc',
            om.ExecComp(
                'resistance = 0.0467 * idle_current ** -1.892', 
                idle_current={'val': 0.0, 'units': 'A'},
                resistance={'val': 0.0, 'units': 'ohm'}
            ),
            promotes_inputs=[('idle_current', Aircraft.Engine.Motor.IDLE_CURRENT)],
            promotes_outputs=[('resistance', Aircraft.Engine.Motor.RESISTANCE)]
        )

        #TODO: Cite
        self.add_subsystem(
            'motor_kv_calc',
            om.ExecComp(
                'kv = m * peak_current / motor_mass + b',
                kv={'val': 0.0, 'units': 'rpm/V'},
                peak_current={'val': 0.0, 'units': 'A'},
                motor_mass={'val': 0.0, 'units': 'kg'},
                m=self.options['m'],
                b=self.options['b'],
            ),
            promotes_inputs=[
                ('peak_current', Aircraft.Engine.Motor.PEAK_CURRENT),
                ('motor_mass', Aircraft.Engine.Motor.MASS),
            ],
            promotes_outputs=[('kv', Aircraft.Engine.Motor.KV)]
        )

