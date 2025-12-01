import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.electrical import AltElectricalMass, ElectricalMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft


class ElectricMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'electric_test',
            ElectricalMass(),
            promotes_outputs=[
                Aircraft.Electrical.MASS,
            ],
            promotes_inputs=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Electrical.MASS_SCALER,
            ],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Electrical.MASS_SCALER,
            ],
            output_keys=Aircraft.Electrical.MASS,
            version=Version.TRANSPORT,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class ElectricMassTest0(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):
        prob = self.prob

        prob.model.add_subsystem(
            'electric_test',
            ElectricalMass(),
            promotes_outputs=[
                Aircraft.Electrical.MASS,
            ],
            promotes_inputs=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Electrical.MASS_SCALER,
            ],
        )

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle', preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            'AdvancedSingleAisle',
            input_keys=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Electrical.MASS_SCALER,
            ],
            output_keys=Aircraft.Electrical.MASS,
            version=Version.TRANSPORT,
        )
        prob.list_indep_vars()


class ElectricMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.electrical as electrical

        electrical.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.electrical as electrical

        electrical.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'electric_test',
            ElectricalMass(),
            promotes_outputs=[
                Aircraft.Electrical.MASS,
            ],
            promotes_inputs=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Electrical.MASS_SCALER,
            ],
        )

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle', preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.LENGTH, 100.0, 'ft')
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 12.0, 'ft')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class AltElectricMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'electric_test',
            AltElectricalMass(),
            promotes_outputs=[
                Aircraft.Electrical.MASS,
            ],
            promotes_inputs=[Aircraft.Electrical.MASS_SCALER],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            case_name,
            input_keys=Aircraft.Electrical.MASS_SCALER,
            output_keys=Aircraft.Electrical.MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
