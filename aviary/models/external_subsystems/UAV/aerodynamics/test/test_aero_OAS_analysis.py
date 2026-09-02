import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from aviary.models.external_subsystems.UAV.aerodynamics.model.aero_OAS_analysis import AeroConditions, LiftBalanceComp
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Dynamic

class TestAeroConditions(unittest.TestCase):

    def test_partials(self):
        nn = 3

        prob = om.Problem()

        prob.model.add_subsystem(
            'aero_conditions',
            AeroConditions(num_nodes=nn),
            promotes=['*'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(
            Dynamic.Mission.VELOCITY,
            np.array([100.0, 150.0, 200.0]),
            units='m/s',
        )

        prob.set_val(
            Dynamic.Atmosphere.DENSITY,
            np.array([1.225, 1.0, 0.8]),
            units='kg/m**3',
        )

        prob.set_val(
            Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
            np.array([1.8e-5, 1.7e-5, 1.6e-5]),
            units='Pa*s',
        )

        prob.run_model()

        partials = prob.check_partials(
            method='cs',
            out_stream=None,
        )

        assert_check_partials(
            partials,
            atol=1.0e-10,
            rtol=1.0e-10,
        )

class TestLiftBalanceComp(unittest.TestCase):

    def test_partials(self):
        nn = 3

        prob = om.Problem()

        prob.model.add_subsystem(
            'liftbalance',
            LiftBalanceComp(num_nodes=nn),
            promotes=['*'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(
            Dynamic.Vehicle.LIFT,
            np.array([10000.0, 12000.0, 15000.0]),
            units='N',
        )

        prob.set_val(
            Dynamic.Vehicle.MASS,
            np.array([1000.0, 1100.0, 1200.0]),
            units='kg',
        )

        prob.run_model()

        partials = prob.check_partials(
            method='cs',
            out_stream=None,
        )

        assert_check_partials(
            partials,
            atol=1.0e-10,
            rtol=1.0e-10,
        )


if __name__ == '__main__':
    unittest.main()