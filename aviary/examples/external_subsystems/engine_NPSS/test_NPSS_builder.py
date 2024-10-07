"""
Benchmark test for aviary mission with an NPSS defined engine.
"""

import unittest
import os as os

from openmdao.utils.assert_utils import assert_near_equal

from aviary.examples.external_subsystems.engine_NPSS.define_simple_engine_problem import define_aviary_NPSS_problem


class AviaryNPSSTestCase(unittest.TestCase):
    """
    Test NPSS engine builder from table by building an Aviary model with NPSS engine and run
    """

    @unittest.skipUnless(os.environ.get('NPSS_TOP', False), 'environment does not contain NPSS')
    def bench_test_aviary_NPSS(self):
        prob = define_aviary_NPSS_problem()
        prob.run_aviary_problem(suppress_solver_print=True)

        rtol = 0.01

        # There are no truth values for these.
        assert_near_equal(prob.get_val('aircraft:engine:design_mass_flow'),
                          315.1648646, tolerance=rtol)

        assert_near_equal(prob.get_val('aircraft:engine:scaled_sls_thrust'),
                          35045.993119, tolerance=rtol)

        assert_near_equal(prob.get_val('traj.cruise.rhs_all.NPSS_prop_system.fuel_flow_rate_negative')[0],
                          -1.13552634, tolerance=rtol)
        assert_near_equal(prob.get_val('traj.cruise.rhs_all.NPSS_prop_system.thrust_net')[0],
                          4253.95759421, tolerance=rtol)


if __name__ == '__main__':
    test = AviaryNPSSTestCase()
    test.bench_test_aviary_NPSS()
