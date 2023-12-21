import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.subsystems.aerodynamics.flops_based.compressibility_drag import \
    CompressibilityDrag
from aviary.variable_info.variables import Aircraft, Mission


class CompressibilityDragTest(unittest.TestCase):

    def test_derivs(self):

        # Nudge the mach and diam/wingspan values off of the table points to prevent problem with
        # linear interp.
        delta = 1e-5

        mach = [.2, .25, .3, .35, .4, .45, .5, .55, .6, .7, .75, .775, .8, .825, .85, .875, 0.9,
                0.95, 1.05, 1.1]
        mach = np.array(mach) + delta
        nn = len(mach)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('LID', CompressibilityDrag(num_nodes=nn),
                            promotes_inputs=[Aircraft.Design.BASE_AREA,
                                             Aircraft.Wing.AREA,
                                             Aircraft.Wing.ASPECT_RATIO,
                                             Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
                                             Aircraft.Wing.SWEEP,
                                             Aircraft.Wing.TAPER_RATIO,
                                             Aircraft.Wing.THICKNESS_TO_CHORD,
                                             Aircraft.Fuselage.CROSS_SECTION,
                                             Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
                                             Aircraft.Fuselage.LENGTH_TO_DIAMETER,
                                             Mission.Design.MACH])

        prob.setup(force_alloc_complex=True)

        prob.set_val('LID.mach', mach)
        prob.set_val(Mission.Design.MACH, .8-delta)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.13)
        prob.set_val(Aircraft.Fuselage.CROSS_SECTION, 128.2)
        prob.set_val(Aircraft.Design.BASE_AREA, 0.01)
        prob.set_val(Aircraft.Wing.AREA, 1370.0)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.432)
        prob.set_val(Aircraft.Wing.ASPECT_RATIO, 11.5)
        prob.set_val(Aircraft.Wing.SWEEP, 25.07)
        prob.set_val(Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, 0.15 + delta)
        prob.set_val(Aircraft.Fuselage.LENGTH_TO_DIAMETER, 10.12345)
        prob.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 0.)

        prob.run_model()
        derivs = prob.check_partials(out_stream=None, method="cs")

        # TODO: need to test outputs too
        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
