import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.flops_based.skin_friction_drag import SkinFrictionDrag
from aviary.variable_info.variables import Aircraft


class SkinFrictionDragTest(unittest.TestCase):
    def test_derivs(self):
        nn = 2

        fine = np.array([0.13, 0.125, 0.1195, 10.0392, 1.5491, 1.5491])
        Re = (
            np.array([[15.0, 11, 18.3, 183.4, 17.6, 17.6], [17.0, 16, 15.3, 78.4, 23.6, 33.6]])
            * 1e-6
        )
        cf = np.array(
            [
                [0.00263, 0.00276, 0.00255, 0.00182, 0.00256, 0.00256],
                [0.00283, 0.00296, 0.00235, 0.00172, 0.00276, 0.00276],
            ]
        )
        Sw = np.array([2396.56, 592.65, 581.13, 4158.62, 273.45, 273.45])
        lam_up = np.array([6 * 0.03])
        lam_low = np.array([6 * 0.02])

        prob = om.Problem()
        model = prob.model

        options = {
            Aircraft.Fuselage.NUM_FUSELAGES: 1,
            Aircraft.Engine.NUM_ENGINES: [2],
            Aircraft.VerticalTail.NUM_TAILS: 1,
            Aircraft.Wing.AIRFOIL_TECHNOLOGY: 1.93,
        }

        model.add_subsystem(
            'CDf',
            SkinFrictionDrag(num_nodes=nn, **options),
            promotes_inputs=[Aircraft.Wing.AREA],
            promotes_outputs=['skin_friction_drag_coeff'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('CDf.fineness_ratios', fine)
        prob.set_val('CDf.Re', Re)
        prob.set_val('CDf.skin_friction_coeff', cf)
        prob.set_val('CDf.wetted_areas', Sw)
        prob.set_val('CDf.laminar_fractions_upper', lam_up)
        prob.set_val('CDf.laminar_fractions_lower', lam_low)
        prob.set_val(Aircraft.Wing.AREA, 198.0)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method='cs')

        # atol is low because Re magnitude is 1e8.
        assert_check_partials(derivs, atol=1e-7, rtol=1e-12)

        assert_near_equal(prob.get_val('skin_friction_drag_coeff'), [14.91229, 15.01284], 1e-6)

    def test_derivs_multiengine(self):
        nn = 2

        fine = np.array([0.13, 0.125, 0.1195, 10.0392, 1.5491, 1.5491, 1.125, 1.125, 1.125, 1.125])
        Re = (
            np.array(
                [
                    [15.0, 11, 18.3, 183.4, 17.6, 17.6, 5.55, 5.55, 5.55, 5.55],
                    [17.0, 16, 15.3, 78.4, 23.6, 33.6, 22.7, 22.7, 22.7, 22.7],
                ]
            )
            * 1e-6
        )
        cf = np.array(
            [
                [
                    0.00263,
                    0.00276,
                    0.00255,
                    0.00182,
                    0.00256,
                    0.00256,
                    0.0184,
                    0.0184,
                    0.0184,
                    0.0184,
                ],
                [
                    0.00283,
                    0.00296,
                    0.00235,
                    0.00172,
                    0.00276,
                    0.00276,
                    0.0203,
                    0.0203,
                    0.0203,
                    0.0203,
                ],
            ]
        )
        Sw = np.array(
            [2396.56, 592.65, 581.13, 4158.62, 273.45, 273.45, 239.1, 239.1, 239.1, 239.1]
        )
        lam_up = np.array([6 * 0.03])
        lam_low = np.array([6 * 0.02])

        prob = om.Problem()
        model = prob.model

        options = {
            Aircraft.Fuselage.NUM_FUSELAGES: 1,
            Aircraft.Engine.NUM_ENGINES: [2, 4],
            Aircraft.VerticalTail.NUM_TAILS: 1,
            Aircraft.Wing.AIRFOIL_TECHNOLOGY: 1.93,
        }

        model.add_subsystem(
            'CDf',
            SkinFrictionDrag(num_nodes=nn, **options),
            promotes_inputs=[Aircraft.Wing.AREA],
            promotes_outputs=['skin_friction_drag_coeff'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('CDf.fineness_ratios', fine)
        prob.set_val('CDf.Re', Re)
        prob.set_val('CDf.skin_friction_coeff', cf)
        prob.set_val('CDf.wetted_areas', Sw)
        prob.set_val('CDf.laminar_fractions_upper', lam_up)
        prob.set_val('CDf.laminar_fractions_lower', lam_low)
        prob.set_val(Aircraft.Wing.AREA, 198.0)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method='cs')

        # atol is low because Re magnitude is 1e8.
        assert_check_partials(derivs, atol=1e-7, rtol=1e-12)

        assert_near_equal(prob.get_val('skin_friction_drag_coeff'), [24.27078, 19.82677], 1e-6)


if __name__ == '__main__':
    unittest.main()
    # test = SkinFrictionDragTest()
    # test.test_derivs_multiengine()
