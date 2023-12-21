import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

from aviary.subsystems.aerodynamics.flops_based.buffet_lift import BuffetLift
from aviary.variable_info.variables import Aircraft, Mission


class TestBuffetLift(unittest.TestCase):

    def test_derivs(self):
        mach = [.2, .25, .3, .35, .4, .45, .5, .55, .6, .7, .75, .775, .8, .825, .85, .875,
                0.9, 0.95, 1.05, 1.1]
        nn = len(mach)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('BUF', BuffetLift(num_nodes=nn),
                            promotes_inputs=[Aircraft.Wing.ASPECT_RATIO,
                                             Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
                                             Aircraft.Wing.SWEEP,
                                             Aircraft.Wing.THICKNESS_TO_CHORD,
                                             Mission.Design.MACH])

        prob.setup(force_alloc_complex=True)

        prob.set_val('BUF.mach', mach)

        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.13)  # 0.1632
        prob.set_val(Aircraft.Wing.SWEEP, 25)
        prob.set_val(Aircraft.Wing.ASPECT_RATIO, 11.232936003236246)
        prob.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 0.0)
        prob.set_val(Mission.Design.MACH, 0.801)

        prob.run_model()
        derivs = prob.check_partials(compact_print=True, method="cs")

        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)

        truth = np.array([0.86693558,  0.83439662,  0.79510222,  0.74915741,  0.6965622,
                          0.63256819,  0.57123245,  0.51727722,  0.45729095,  0.31903298,
                          0.2212182,  0.15774801,  0.08446832, -0.00190504, -0.09177745,
                          -0.18507041, -0.28122907, -0.4820969, -0.91803465, -1.15310457])

        assert_near_equal(prob['BUF.DELCLB'], truth, 1e-7)


if __name__ == "__main__":
    unittest.main()
