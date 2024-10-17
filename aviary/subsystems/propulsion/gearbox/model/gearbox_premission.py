import openmdao.api as om
import numpy as np

from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues


class GearboxPreMission(om.Group):
    """
    Calculate gearbox mass for a single gearbox.

    Gearbox design assumes collective control which means that RPM coming into the 
    gearbox is fixed and RPM going out of the gearbox is fixed over the whole mission.
    """

    def initialize(self, ):
        self.options.declare("simple_mass", types=bool, default=True)
        self.options.declare(
            "aviary_inputs", types=AviaryValues,
            desc="collection of Aircraft/Mission specific options",
            default=None,
        )
        self.name = 'gearbox_premission'

    def setup(self):
        self.add_subsystem('gearbox_PRM',
                           om.ExecComp('RPM_out = RPM_in / gear_ratio',
                                       RPM_out={'val': 0.0, 'units': 'rpm'},
                                       gear_ratio={'val': 1.0, 'units': 'unitless'},
                                       RPM_in={'val': 0.0, 'units': 'rpm'},
                                       has_diag_partials=True),
                           promotes_inputs=[('RPM_in', Aircraft.Engine.RPM_DESIGN),
                                            ('gear_ratio', Aircraft.Engine.Gearbox.GEAR_RATIO)],
                           promotes_outputs=['RPM_out'])

        # max torque is calculated based on input shaft power and output RPM
        self.add_subsystem('torque_comp',
                           om.ExecComp('torque_max = shaft_power / RPM_out',
                                       shaft_power={'val': 1.0, 'units': 'kW'},
                                       torque_max={'val': 1.0, 'units': 'kN*m'},
                                       RPM_out={'val': 1.0, 'units': 'rad/s'},
                                       has_diag_partials=True),
                           promotes_inputs=[('shaft_power', Aircraft.Engine.Gearbox.SHAFT_POWER_DESIGN),
                                            'RPM_out'],
                           promotes_outputs=['torque_max'])

        if self.options["simple_mass"]:
            # Simple gearbox mass will always produce positive values for mass based on a fixed specific torque
            self.add_subsystem(
                'mass_comp',
                om.ExecComp(
                    'gearbox_mass = torque_max / specific_torque',
                    gearbox_mass={'val': 0.0, 'units': 'kg'},
                    torque_max={'val': 0.0, 'units': 'N*m'},
                    specific_torque={'val': 0.0, 'units': 'N*m/kg'},
                    has_diag_partials=True,
                ),
                promotes_inputs=[
                    'torque_max',
                    ('specific_torque', Aircraft.Engine.Gearbox.SPECIFIC_TORQUE),
                ],
                promotes_outputs=[('gearbox_mass', Aircraft.Engine.Gearbox.MASS)],
            )

        else:
            # This gearbox mass calc can work for large systems but can produce negative weights for some inputs
            # Gearbox mass from "An N+3 Technolgoy Level Reference Propulsion System" by Scott Jones, William Haller, and Michael Tong
            # NASA TM 2017-219501
            self.add_subsystem('gearbox_mass',
                               om.ExecComp('gearbox_mass = (shaftpower / RPM_out)**(0.75) * (RPM_in / RPM_out)**(0.15)',
                                           gearbox_mass={'val': 0.0, 'units': 'lb'},
                                           shaftpower={'val': 0.0, 'units': 'hp'},
                                           RPM_out={'val': 0.0, 'units': 'rpm'},
                                           RPM_in={'val': 0.0, 'units': 'rpm'},
                                           has_diag_partials=True),
                               promotes_inputs=[('shaftpower', Aircraft.Engine.Gearbox.SHAFT_POWER_DESIGN),
                                                'RPM_out', ('RPM_in', Aircraft.Engine.RPM_DESIGN)],
                               promotes_outputs=[('gearbox_mass', Aircraft.Engine.Gearbox.MASS)])
