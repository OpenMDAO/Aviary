import numpy as np
import openmdao.api as om

from aviary import constants
from aviary.variable_info.variables import Dynamic


class SolveAlphaGroup(om.Group):
    """
    Group that contains components needed to determine angle of attack. Must be paired with
    aerodynamics method and a solver to properly balance lift and weight.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        self.add_subsystem(
            'required_lift',
            om.ExecComp(
                'required_lift = mass * grav_metric',
                grav_metric={'val': constants.GRAV_METRIC_GASP},
                mass={'units': 'kg', 'shape': num_nodes},
                required_lift={'units': 'N', 'shape': num_nodes},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('mass', Dynamic.Vehicle.MASS),
            ],
        )

        balance = self.add_subsystem(
            'balance',
            om.BalanceComp(),
            promotes=[Dynamic.Vehicle.ANGLE_OF_ATTACK, Dynamic.Vehicle.LIFT],
        )
        balance.add_balance(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            val=np.ones(num_nodes),
            units='deg',
            res_ref=1.0e6,
            lhs_name=Dynamic.Vehicle.LIFT,
            rhs_name='required_lift',
            eq_units='lbf',
        )

        self.connect(
            'required_lift.required_lift',
            'balance.required_lift',
        )
