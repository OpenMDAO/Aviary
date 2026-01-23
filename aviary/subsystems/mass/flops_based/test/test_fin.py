import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.fin import FinMass
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    do_validation_test,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
    Version,
)
from aviary.variable_info.variables import Aircraft, Mission

# TODO - None of the flops cases have any fins.

fin_test_data = {}
fin_test_data['1'] = AviaryValues(
    {
        Aircraft.Fins.NUM_FINS: (1, 'unitless'),
        Mission.Design.GROSS_MASS: (100000, 'lbm'),
        Aircraft.Fins.TAPER_RATIO: (0.3300, 'unitless'),
        Aircraft.Fins.AREA: (250.00, 'ft**2'),
        Aircraft.Fins.MASS_SCALER: (1.0, 'unitless'),
        Aircraft.Fins.MASS: (917.228, 'lbm'),
    }
)

fin_data_sets = [key for key in fin_test_data]

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS']


class FinMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(fin_data_sets, name_func=print_case)
    def test_case(self, case_name):
        validation_data = fin_test_data[case_name]
        prob = self.prob

        options = {
            Aircraft.Fins.NUM_FINS: 1,
        }
        prob.model.add_subsystem(
            'fin',
            FinMass(**options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=validation_data,
            output_validation_data=validation_data,
            input_keys=[
                Mission.Design.GROSS_MASS,
                Aircraft.Fins.TAPER_RATIO,
                Aircraft.Fins.AREA,
                Aircraft.Fins.MASS_SCALER,
            ],
            output_keys=Aircraft.Fins.MASS,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class FinMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.fin as fin

        fin.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.fin as fin

        fin.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        options = {
            Aircraft.Fins.NUM_FINS: 1,
        }
        prob.model.add_subsystem(
            'fin',
            FinMass(**options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Mission.Design.GROSS_MASS, 100000.0, 'lbm')
        prob.set_val(Aircraft.Fins.TAPER_RATIO, 0.33, 'unitless')
        prob.set_val(Aircraft.Fins.AREA, 250.0, 'ft**2')
        prob.set_val(Aircraft.Fins.MASS, 1000.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class BWBFinMassTest(unittest.TestCase):
    """Tests engine mass calculation for BWB."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'engine_mass',
            FinMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Mission.Design.GROSS_MASS,
                Aircraft.Fins.TAPER_RATIO,
                Aircraft.Fins.AREA,
            ],
            output_keys=[
                Aircraft.Fins.MASS,
            ],
            list_inputs=True,
            list_outputs=True,
            version=Version.BWB,
            rtol=1e-10,
        )


if __name__ == '__main__':
    unittest.main()
