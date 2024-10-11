import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.mass_to_weight import MassToWeight


class MassToWeightTest(unittest.TestCase):
    """
    Test computation of weight from mass.
    """

    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):

        prob = om.Problem()

        prob.model.add_subsystem(
            "calc_weight",
            MassToWeight(),
            promotes=['mass', 'weight']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('mass', 120_000, 'lbm')

        prob.run_model()
        prob.model.list_inputs()
        prob.model.list_outputs()
        assert_near_equal(prob.get_val('weight', 'lbf'),
                          120_000 * GRAV_ENGLISH_LBM, 1.0e-12)
        partial_data = prob.check_partials(compact_print=True, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-12)

        nn = 3
        mass = np.full(nn, 120_000)

        prob = om.Problem()

        prob.model.add_subsystem(
            "calc_weight",
            MassToWeight(num_nodes=nn),
            promotes=['mass', 'weight']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('mass', mass, 'lbm')

        prob.run_model()
        prob.model.list_inputs()
        prob.model.list_outputs()
        assert_near_equal(prob.get_val('weight', 'lbf'),
                          mass * GRAV_ENGLISH_LBM, 1.0e-12)
        partial_data = prob.check_partials(compact_print=True, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-12)


class MassToWeightTest2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.subsystems.mass.mass_to_weight as m_to_w
        m_to_w.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.mass_to_weight as m_to_w
        m_to_w.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "calc_weight",
            MassToWeight(),
            promotes=['mass', 'weight']
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('mass', 120_000, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

        nn = 2
        mass = np.full(nn, 120_000)
        prob = om.Problem()
        prob.model.add_subsystem(
            "calc_weight",
            MassToWeight(num_nodes=nn),
            promotes=['mass', 'weight']
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('mass', mass, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
