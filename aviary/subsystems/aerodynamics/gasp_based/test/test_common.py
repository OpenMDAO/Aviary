import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs


from aviary.subsystems.aerodynamics.gasp_based.common import (
    AeroForces,
    CLFromLift,
    TanhRampComp,
    TimeRamp,
)
from aviary.variable_info.variables import Aircraft, Dynamic


@use_tempdirs
class TestTanhRampComp(unittest.TestCase):
    def test_tanh_ramp_up(self):
        p = om.Problem()

        nn = 1000

        c = TanhRampComp(time_units='s', num_nodes=nn)

        c.add_ramp(
            'thruput',
            output_units='kg/s',
            initial_val=30,
            final_val=40,
            t_init_val=25,
            t_duration_val=5,
        )

        p.model.add_subsystem('tanh_ramp', c)

        p.setup(force_alloc_complex=True)

        p.set_val('tanh_ramp.time', val=np.linspace(0, 100, nn))

        p.run_model()

        cpd = p.check_partials(compact_print=True, method='cs', out_stream=None)

        thruput = p.get_val('tanh_ramp.thruput')

        assert_near_equal(thruput[250], desired=30, tolerance=0.01)
        assert_near_equal(thruput[275], desired=35, tolerance=0.01)
        assert_near_equal(thruput[300], desired=40, tolerance=0.01)

        assert_near_equal(thruput[500:], desired=40 * np.ones((500, 1)), tolerance=0.01)
        assert_near_equal(thruput[:200], desired=30 * np.ones((200, 1)), tolerance=0.01)

        self.assertTrue(np.all(thruput >= 30))
        self.assertTrue(np.all(thruput <= 40))

        #assert_check_partials(cpd, atol=1.0e-9, rtol=1.0e-12)


if __name__ == '__main__':
    unittest.main()
