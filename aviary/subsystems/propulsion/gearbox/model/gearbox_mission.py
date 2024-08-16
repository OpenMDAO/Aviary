import numpy as np

import openmdao.api as om
from aviary.utils.aviary_values import AviaryValues

from aviary.variable_info.variables import Dynamic, Aircraft, Mission


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
                                       gear_ratio={'val': 1.0, 'units': 'unitless'},
                                       RPM_in={'val': np.ones(n), 'units': 'rpm'},
                                       has_diag_partials=True),
                           promotes_inputs=[('RPM_in', Aircraft.Engine.RPM_DESIGN),
                                            ('gear_ratio', Aircraft.Engine.Gearbox.GEAR_RATIO)],
                           promotes_outputs=[('RPM_out', Dynamic.Mission.RPM_GEARBOX)])

        self.add_subsystem('shaft_power_comp',
                           om.ExecComp('shaft_power_out = shaft_power_in * eff',
                                       shaft_power_in={'val': np.ones(n), 'units': 'kW'},
                                       shaft_power_out={
                                           'val': np.ones(n), 'units': 'kW'},
                                       eff={'val': 0.98, 'units': 'unitless'},
                                       has_diag_partials=True),
                           promotes_inputs=[('shaft_power_in', Dynamic.Mission.SHAFT_POWER),
                                            ('eff', Aircraft.Engine.Gearbox.EFFICIENCY)],
                           promotes_outputs=[('shaft_power_out', Dynamic.Mission.SHAFT_POWER_GEARBOX)])

        self.add_subsystem('torque_comp',
                           om.ExecComp('torque_out = shaft_power_out / RPM_out',
                                       shaft_power_out={
                                           'val': np.ones(n), 'units': 'kW'},
                                       torque_out={'val': np.ones(n), 'units': 'kN*m'},
                                       RPM_out={'val': np.ones(n), 'units': 'rad/s'},
                                       has_diag_partials=True),
                           promotes_inputs=[('shaft_power_out', Dynamic.Mission.SHAFT_POWER_GEARBOX),
                                            ('RPM_out', Dynamic.Mission.RPM_GEARBOX)],
                           promotes_outputs=[('torque_out', Dynamic.Mission.TORQUE_GEARBOX)])

        # Determine the maximum power available at this flight condition
        # this is used for excess power constraints
        self.add_subsystem('shaft_power_max_comp',
                           om.ExecComp('shaft_power_out = shaft_power_in * eff',
                                       shaft_power_in={'val': np.ones(n), 'units': 'kW'},
                                       shaft_power_out={
                                           'val': np.ones(n), 'units': 'kW'},
                                       eff={'val': 0.98, 'units': 'unitless'},
                                       has_diag_partials=True),
                           promotes_inputs=[('shaft_power_in', Dynamic.Mission.SHAFT_POWER_MAX),
                                            ('eff', Aircraft.Engine.Gearbox.EFFICIENCY)],
                           promotes_outputs=[('shaft_power_out', Dynamic.Mission.SHAFT_POWER_MAX_GEARBOX)])

        # We must ensure the design shaft power that was provided to pre-mission is
        # larger than the maximum shaft power that could be drawn by the mission.
        # Note this is a larger value than the actual maximum shaft power drawn during the mission
        # because the aircraft might need to climb to avoid obstacles at anytime during the mission
        self.add_subsystem('shaft_power_residual',
                           om.ExecComp('shaft_power_resid = shaft_power_design - shaft_power_max',
                                       shaft_power_max={
                                           'val': np.ones(n), 'units': 'kW'},
                                       shaft_power_design={'val': 1.0, 'units': 'kW'},
                                       shaft_power_resid={
                                           'val': np.ones(n), 'units': 'kW'},
                                       has_diag_partials=True),
                           promotes_inputs=[('shaft_power_max', Dynamic.Mission.SHAFT_POWER_MAX),
                                            ('shaft_power_design', Aircraft.Engine.Gearbox.SHAFT_POWER_DESIGN)],
                           promotes_outputs=[('shaft_power_resid', Mission.Constraints.SHAFT_POWER_RESIDUAL)])

        # TODO max thrust from the props will depend on this max shaft power from the gearbox and the new gearbox RPM value
