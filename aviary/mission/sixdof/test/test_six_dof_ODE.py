import unittest 
import openmdao.api as om
import numpy as np

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.sixdof.six_dof_ODE import SixDOF_ODE
from aviary.variable_info.variables import Aircraft, Dynamic


class SixDOFODETestCase(unittest.TestCase):
    """
    Test 6-degree of freedom ODE.

    """

    def setUp(self):
        self.prob = om.Problem()

        self.sys = self.prob.model = SixDOF_ODE(
            num_nodes=1, 
            
        )

