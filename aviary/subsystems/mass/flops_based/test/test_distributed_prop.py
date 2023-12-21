import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor, distributed_nacelle_diam_factor,
    distributed_nacelle_diam_factor_deriv, distributed_thrust_factor)


class DistributedPropulsionFactorsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_case_dist_eng_count(self):
        tol = 1e-12

        for num_eng in [1, 2, 3, 4]:
            num_eng_fact = distributed_engine_count_factor(num_eng)
            assert_near_equal(num_eng_fact, num_eng, tol)

        num_eng = 5
        num_eng_fact = distributed_engine_count_factor(num_eng)
        assert_near_equal(num_eng_fact, 4.643501108793284, tol)

    def test_case_dist_thrust(self):
        tol = 1e-12
        max_thrust = 10000.0

        for num_eng in [1, 2, 3, 4]:
            thrust_fact = distributed_thrust_factor(max_thrust, num_eng)
            assert_near_equal(thrust_fact, max_thrust/num_eng, tol)

        num_eng = 5
        thrust_fact = distributed_thrust_factor(max_thrust, num_eng)
        assert_near_equal(thrust_fact, max_thrust/4.643501108793284, tol)

    def test_case_dist_nacelle_diam(self):
        tol = 1e-12
        diam_nacelle = 10.0

        for num_eng in [1, 2, 3, 4]:
            diam_fact = distributed_nacelle_diam_factor(diam_nacelle, num_eng)
            assert_near_equal(diam_fact, diam_nacelle, tol)

        num_eng = 5
        diam_fact = distributed_nacelle_diam_factor(diam_nacelle, num_eng)
        assert_near_equal(diam_fact, diam_nacelle*1.11803398875, tol)

    def test_case_dist_nacelle_diam_deriv(self):
        tol = 1e-12

        for num_eng in [1, 2, 3, 4]:
            diam_fact = distributed_nacelle_diam_factor_deriv(num_eng)
            assert_near_equal(diam_fact, 1.0, tol)

        num_eng = 5
        diam_fact = distributed_nacelle_diam_factor_deriv(num_eng)
        assert_near_equal(diam_fact, 1.11803398875, tol)


if __name__ == "__main__":
    unittest.main()
