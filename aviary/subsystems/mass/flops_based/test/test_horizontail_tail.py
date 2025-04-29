import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.horizontal_tail import (
    AltHorizontalTailMass,
    HorizontalTailMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission


class ExplicitHorizontalTailMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'horizontal_tail',
            HorizontalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.TAPER_RATIO,
                Mission.Design.GROSS_MASS,
                Aircraft.HorizontalTail.MASS_SCALER,
            ],
            output_keys=Aircraft.HorizontalTail.MASS,
            version=Version.TRANSPORT,
            tol=2.0e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class ExplicitHorizontalTailMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.horizontal_tail as htail

        htail.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.horizontal_tail as htail

        htail.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'horizontal_tail',
            HorizontalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.HorizontalTail.AREA, 10.0, 'ft**2')
        prob.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 10.0, 'unitless')
        prob.set_val(Mission.Design.GROSS_MASS, 1000.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class ExplicitAltHorizontalTailMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'horizontal_tail',
            AltHorizontalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.HorizontalTail.AREA, Aircraft.HorizontalTail.MASS_SCALER],
            output_keys=Aircraft.HorizontalTail.MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class ExplicitAltHorizontalTailMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.horizontal_tail as htail

        htail.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.horizontal_tail as htail

        htail.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'horizontal_tail',
            AltHorizontalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.HorizontalTail.AREA, 10.0, 'ft**2')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
