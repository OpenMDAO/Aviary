import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.subsystems.aerodynamics.flops_based.skin_friction_drag import \
    SkinFrictionDrag
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft


class SkinFrictionDragTest(unittest.TestCase):

    def test_derivs(self):
        nn = 2

        fine = np.array([0.13, .125, .1195, 10.0392, 1.5491, 1.5491])
        Re = np.array([[15.0, 11, 18.3, 183.4, 17.6, 17.6],
                      [17.0, 16, 15.3, 78.4, 23.6, 33.6]]) * 1e-6
        cf = np.array([[.00263, .00276, .00255, .00182, .00256, .00256],
                       [.00283, .00296, .00235, .00172, .00276, .00276]])
        Sw = np.array([2396.56, 592.65, 581.13, 4158.62, 273.45, 273.45])
        lam_up = np.array([6 * 0.03])
        lam_low = np.array([6 * 0.02])
        nc = len(fine)

        prob = om.Problem()
        model = prob.model

        options = AviaryValues()
        options.set_val(Aircraft.Fuselage.NUM_FUSELAGES, 1)
        options.set_val(Aircraft.Engine.NUM_ENGINES, [2])
        options.set_val(Aircraft.VerticalTail.NUM_TAILS, 1)
        options.set_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY, 1.93)

        model.add_subsystem(
            'CDf', SkinFrictionDrag(num_nodes=nn, aviary_options=options),
            promotes_inputs=[Aircraft.Wing.AREA])

        prob.setup(force_alloc_complex=True)

        prob.set_val('CDf.fineness_ratios', fine)
        prob.set_val('CDf.Re', Re)
        prob.set_val('CDf.skin_friction_coeff', cf)
        prob.set_val('CDf.wetted_areas', Sw)
        prob.set_val('CDf.laminar_fractions_upper', lam_up)
        prob.set_val('CDf.laminar_fractions_lower', lam_low)
        prob.set_val(Aircraft.Wing.AREA, 198.0)

        derivs = prob.check_partials(out_stream=None, method="cs")

        # atol is low because Re magnitude is 1e8.
        # TODO: need to test outputs too
        assert_check_partials(derivs, atol=1e-7, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
