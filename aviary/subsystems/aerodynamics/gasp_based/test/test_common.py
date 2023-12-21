import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.common import (CLFromLift,
                                                              TanhRampComp,
                                                              TimeRamp)


class TestTimeRamp(unittest.TestCase):
    def test_single_ramp_up(self):
        k = 1
        nn = 11
        t = np.linspace(0, 10, nn)
        x = np.ones(nn)
        y_exp = np.array([[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]])

        tr = TimeRamp(num_nodes=nn, num_inputs=k, ramp_up=True)

        prob = om.Problem()
        prob.model.add_subsystem("comp", tr, promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val("t_init", 3)
        prob.set_val("t_curr", t)
        prob.set_val("duration", 4)
        prob.set_val("x", x)

        prob.run_model()

        y_exp = np.array([[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]])
        assert_near_equal(prob["y"], y_exp)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

    def test_multi_ramp_down(self):
        k = 2
        nn = 11
        t = np.linspace(0, 10, nn)
        x = np.tile([[1], [10]], nn)
        y_exp = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 0.0, 0.0],
            ]
        )

        tr = TimeRamp(num_nodes=nn, num_inputs=k, ramp_up=False)

        prob = om.Problem()
        prob.model.add_subsystem("comp", tr, promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val("t_init", 3)
        prob.set_val("t_curr", t)
        prob.set_val("duration", 5)
        prob.set_val("x", x)

        prob.run_model()

        assert_near_equal(prob["y"], y_exp)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class TestCLFromLift(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model.add_subsystem("comp", CLFromLift(num_nodes=2), promotes=["*"])
        prob.setup(check=False, force_alloc_complex=True)
        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class TestTanhRampComp(unittest.TestCase):

    def test_tanh_ramp_up(self):
        p = om.Problem()

        nn = 1000

        c = TanhRampComp(time_units='s', num_nodes=nn)

        c.add_ramp('thruput', output_units='kg/s', initial_val=30,
                   final_val=40, t_init_val=25, t_duration_val=5)

        p.model.add_subsystem('tanh_ramp', c)

        p.setup(force_alloc_complex=True)

        p.set_val('tanh_ramp.time', val=np.linspace(0, 100, nn))

        p.run_model()

        cpd = p.check_partials(compact_print=True, method='cs')

        thruput = p.get_val('tanh_ramp.thruput')

        assert_near_equal(thruput[250], desired=30, tolerance=0.01)
        assert_near_equal(thruput[275], desired=35, tolerance=0.01)
        assert_near_equal(thruput[300], desired=40, tolerance=0.01)

        assert_near_equal(thruput[500:], desired=40*np.ones((500, 1)), tolerance=0.01)
        assert_near_equal(thruput[:200], desired=30*np.ones((200, 1)), tolerance=0.01)

        self.assertTrue(np.all(thruput >= 30))
        self.assertTrue(np.all(thruput <= 40))

        assert_check_partials(cpd, atol=1.0E-9, rtol=1.0E-12)

    def test_tanh_ramp_down(self):
        p = om.Problem()

        nn = 1000

        c = TanhRampComp(time_units='s', num_nodes=nn)

        c.add_ramp('thruput', output_units='kg/s', initial_val=40,
                   final_val=30, t_init_val=25, t_duration_val=5)

        p.model.add_subsystem('tanh_ramp', c)

        p.setup(force_alloc_complex=True)

        p.set_val('tanh_ramp.time', val=np.linspace(0, 100, nn))

        p.run_model()

        cpd = p.check_partials(compact_print=True, method='cs', out_stream=None)

        thruput = p.get_val('tanh_ramp.thruput')[:, 0]

        assert_near_equal(thruput[250], desired=40, tolerance=0.01)
        assert_near_equal(thruput[275], desired=35, tolerance=0.01)
        assert_near_equal(thruput[300], desired=30, tolerance=0.01)

        assert_near_equal(thruput[500:], desired=30*np.ones(500), tolerance=0.01)
        assert_near_equal(thruput[:200], desired=40*np.ones(200), tolerance=0.01)

        self.assertTrue(np.all(thruput >= 30))
        self.assertTrue(np.all(thruput <= 40))

        assert_check_partials(cpd, atol=1.0E-9, rtol=1.0E-12)


if __name__ == "__main__":
    unittest.main()
