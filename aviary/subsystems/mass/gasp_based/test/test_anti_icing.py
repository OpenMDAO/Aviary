import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.anti_icing import AntiIcingMass

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


class AntiIcingTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units='ft**2'
        )  # Unknown origin
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units='ft**2'
        )  # large_single_airsle_1_FLOPS_data.py
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2'
        )  # Unknown origin

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AntiIcing.MASS], 683.46852785, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class AntiIcingTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes=['*'],
        )

        import aviary.subsystems.mass.gasp_based.anti_icing as anti_icing

        anti_icing.GRAV_ENGLISH_LBM = 1.1

        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units='ft**2'
        )  # Unknown origin
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units='ft**2'
        )  # large_single_airsle_1_FLOPS_data.py
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2'
        )  # Unknown origin

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.anti_icing as anti_icing

        anti_icing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AntiIcing.MASS], 621.33502532, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class AntiIcingTestCase3(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units='ft**2'
        )  # Unknown origin
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units='ft**2'
        )  # large_single_airsle_1_FLOPS_data.py
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2'
        )  # Unknown origin

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AntiIcing.MASS], 683.46852785, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
