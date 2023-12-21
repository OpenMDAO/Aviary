import numpy as np
import openmdao.api as om

from pathlib import Path

import aviary.constants as constants
from aviary.subsystems.aerodynamics.aero_common import DynamicPressure
from aviary.subsystems.aerodynamics.gasp_based.table_based import CruiseAero as TabularCruiseAero
from aviary.utils.named_values import NamedValues
from aviary.variable_info.variables import Aircraft, Dynamic

grav_metric = constants.GRAV_METRIC_FLOPS


class SolvedAlphaGroup(om.Group):
    """
    Aerodynmaics group that computes the coefficients of lift and drag on a
    structured table of altitude, mach, and angle of attack. The angle of
    attack is solved iteratively to balance the lift with the aircraft weight.

    The aerodynamics data can come from a file or can come from the output of
    a component in the aviary pre_mission group.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('num_nodes', types=int)

        self.options.declare('aero_data', types=(str, Path, NamedValues), default=None,
                             desc='Data file or NamedValues object containing lift and '
                                  'drag coefficient table as a function of altitude, '
                                  'Mach, and angle of attack')

        self.options.declare('training_data', default=False,
                             desc='When True, the aero tables will be passed as '
                                  'OpenMDAO variables')

        self.options.declare('structured', types=bool, default=True,
                             desc='Flag that sets if data is a structured grid')

        self.options.declare('extrapolate', default=True, desc='Flag that sets if drag '
                                                               'data can be extrapolated')

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        aero_data = options['aero_data']
        training_data = options['training_data']
        structured = options['structured']
        extrapolate = options['extrapolate']

        self.add_subsystem(
            'DynamicPressure', DynamicPressure(num_nodes=nn),
            promotes_inputs=[Dynamic.Mission.MACH, Dynamic.Mission.STATIC_PRESSURE],
            promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE])

        aero = TabularCruiseAero(num_nodes=nn,
                                 aero_data=aero_data,
                                 training_data=training_data,
                                 structured=structured,
                                 extrapolate=extrapolate)

        if training_data:
            extra_promotes = [Aircraft.Design.DRAG_POLAR,
                              Aircraft.Design.LIFT_POLAR]
        else:
            extra_promotes = []

        self.add_subsystem("tabular_aero", aero,
                           promotes_inputs=[Dynamic.Mission.ALTITUDE, Dynamic.Mission.MACH,
                                            Aircraft.Wing.AREA, Dynamic.Mission.MACH,
                                            Dynamic.Mission.DYNAMIC_PRESSURE]
                           + extra_promotes,
                           promotes_outputs=[Dynamic.Mission.LIFT, Dynamic.Mission.DRAG])

        balance = self.add_subsystem('balance', om.BalanceComp())
        balance.add_balance('alpha', val=np.ones(nn), units='deg')

        self.connect('balance.alpha', 'tabular_aero.alpha')
        self.connect('needed_lift.lift_resid', 'balance.lhs:alpha')

        self.add_subsystem('needed_lift',
                           om.ExecComp('lift_resid = mass * grav_metric - computed_lift',
                                       grav_metric={'val': grav_metric},
                                       mass={'units': 'kg', 'shape': nn},
                                       computed_lift={'units': 'N', 'shape': nn},
                                       lift_resid={'shape': nn},
                                       ),
                           promotes_inputs=[('mass', Dynamic.Mission.MASS),
                                            ('computed_lift', Dynamic.Mission.LIFT)]
                           )

        self.linear_solver = om.DirectSolver()
        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        newton.options['iprint'] = 2
        newton.options['atol'] = 1e-9
        newton.options['rtol'] = 1e-12
