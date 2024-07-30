import numpy as np

import openmdao.api as om
from aviary.utils.aviary_values import AviaryValues

from aviary.variable_info.variables import Dynamic, Aircraft


class GearboxMission(om.Group):
    """Calculates the mission performance (ODE) of a single gearbox."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            'aviary_inputs', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )
        self.name = 'gearbox_mission'

    def setup(self):
        n = self.options["num_nodes"]

        self.add_subsystem('RPM_comp',
                           om.ExecComp('RPM_out = RPM_in / gear_ratio',
                                       RPM_out={'val': np.ones(n), 'units': 'rpm'},
                                       gear_ratio={'val': 1.0, 'units': None},
                                       RPM_in={'val': np.ones(n), 'units': 'rpm'}),
                           promotes_inputs=[('RPM_in', Dynamic.Mission.RPM),
                                            ('gear_ratio', Aircraft.Engine.Gearbox.GEAR_RATIO)],
                           promotes_outputs=[('RPM_out', Dynamic.Mission.RPM_GEAR)])

        self.add_subsystem('shaft_power_comp',
                           om.ExecComp('shaft_power_out = shaft_power_in * eff',
                                       shaft_power_in={'val': np.ones(n), 'units': 'kW'},
                                       shaft_power_out={
                                           'val': np.ones(n), 'units': 'kW'},
                                       eff={'val': 0.98, 'units': None}),
                           promotes_inputs=[('shaft_power_in', Dynamic.Mission.SHAFT_POWER),
                                            ('eff', Aircraft.Engine.Gearbox.EFFICIENCY)],
                           promotes_outputs=[('shaft_power_out', Dynamic.Mission.SHAFT_POWER_GEAR)])

        self.add_subsystem('torque_comp',
                           om.ExecComp('torque = shaft_power / (pi * RPM_out) * 30',
                                       shaft_power={'val': np.ones(n), 'units': 'kW'},
                                       torque={'val': np.ones(n), 'units': 'kN*m'},
                                       RPM_out={'val': np.ones(n), 'units': 'rpm'}),
                           promotes_inputs=[('shaft_power', Dynamic.Mission.SHAFT_POWER_GEAR),
                                            ('RPM_out', Dynamic.Mission.RPM_GEAR)],
                           promotes_outputs=[('torque', Dynamic.Mission.TORQUE_GEAR)])

        # Determine the maximum power available at this flight condition
        # this is used for excess power constraints
        self.add_subsystem('shaft_power_max_comp',
                           om.ExecComp('shaft_power_out = shaft_power_in * eff',
                                       shaft_power_in={'val': np.ones(n), 'units': 'kW'},
                                       shaft_power_out={
                                           'val': np.ones(n), 'units': 'kW'},
                                       eff={'val': 0.98, 'units': None}),
                           promotes_inputs=[('shaft_power_in', Dynamic.Mission.SHAFT_POWER_MAX),
                                            ('eff', Aircraft.Engine.Gearbox.EFFICIENCY)],
                           promotes_outputs=[('shaft_power_out', Dynamic.Mission.SHAFT_POWER_MAX_GEAR)])

        # We must ensure the maximum shaft power guess that we provided to pre-mission is enforced
        # residual needs to be 0 or larger for all cases
        self.add_subsystem('shaft_power_residual',
                           om.ExecComp('shaft_power_resid = shaft_power_design - shaft_power_max',
                                       shaft_power_max={
                                           'val': np.ones(n), 'units': 'kW'},
                                       shaft_power_design={'val': 1.0, 'units': 'kW'},
                                       shaft_power_resid={'val': np.ones(n), 'units': 'kW'}),
                           promotes_inputs=[('shaft_power_max', Dynamic.Mission.SHAFT_POWER_MAX),
                                            ('shaft_power_design', Aircraft.Engine.SHAFT_POWER_DESIGN)],
                           promotes_outputs=['shaft_power_resid'])

        # TODO max thrust from the props will depend on this max shaft power from the gearbox and the new gearbox RPM value
