import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.flops_based.design import Design
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Mission


class DesignMCLTest(unittest.TestCase):

    def test_derivs_supersonic(self):
        prob = om.Problem()
        model = prob.model

        options = {}
        options[Aircraft.Wing.AIRFOIL_TECHNOLOGY] = (1.0, 'unitless')
        options[Mission.Constraints.MAX_MACH] = (0.9, 'unitless')

        model.add_subsystem(
            'design',
            Design(aviary_options=AviaryValues(options)),
            promotes_inputs=['*'],
            promotes_outputs=[Mission.Design.MACH, Mission.Design.LIFT_COEFFICIENT]
        )
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.ASPECT_RATIO, val=11.05)
        prob.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, val=1.0)
        prob.set_val(Aircraft.Wing.SWEEP, val=2.0191)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.12)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)

        assert_near_equal(
            prob.get_val(Mission.Design.MACH), [0.671088], 1e-6)
        assert_near_equal(
            prob.get_val(Mission.Design.LIFT_COEFFICIENT), [0.683078], 1e-6)

    def test_derivs_subsonic(self):
        prob = om.Problem()
        model = prob.model

        options = {}
        options[Aircraft.Wing.AIRFOIL_TECHNOLOGY] = (1.0, 'unitless')
        options[Mission.Constraints.MAX_MACH] = (0.9, 'unitless')

        model.add_subsystem(
            'design',
            Design(aviary_options=AviaryValues(options)),
            promotes_inputs=['*'],
            promotes_outputs=[Mission.Design.MACH, Mission.Design.LIFT_COEFFICIENT],
        )
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.ASPECT_RATIO, val=11.05)
        prob.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, val=1.0)
        prob.set_val(Aircraft.Wing.SWEEP, val=2.191)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.12)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)

        assert_near_equal(
            prob.get_val(Mission.Design.MACH), [0.671145], 1e-6)
        assert_near_equal(
            prob.get_val(Mission.Design.LIFT_COEFFICIENT), [0.683002], 1e-6)


if __name__ == "__main__":
    unittest.main()
