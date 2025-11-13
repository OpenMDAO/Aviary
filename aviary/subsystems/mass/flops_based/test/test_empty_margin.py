import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.empty_margin import EmptyMassMargin
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class EmptyMassMarginTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit='BWB1aFLOPS'), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'margin',
            EmptyMassMargin(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Propulsion.MASS,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS,
                Aircraft.Design.EMPTY_MASS_MARGIN_SCALER,
            ],
            output_keys=Aircraft.Design.EMPTY_MASS_MARGIN,
            tol=1e-3,
            atol=2e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class BWBEmptyMassMarginTest(unittest.TestCase):
    """Test empty mass margin calculation for BWB data."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):
        case_name = 'BWB1aFLOPS'
        prob = self.prob

        prob.model.add_subsystem(
            'margin',
            EmptyMassMargin(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Propulsion.MASS,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS,
                Aircraft.Design.EMPTY_MASS_MARGIN_SCALER,
            ],
            output_keys=Aircraft.Design.EMPTY_MASS_MARGIN,
            version=Version.BWB,
            tol=1e-3,
            atol=2e-11,
        )


if __name__ == '__main__':
    unittest.main()
