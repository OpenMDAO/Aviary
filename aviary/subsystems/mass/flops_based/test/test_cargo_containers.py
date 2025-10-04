import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.cargo_containers import TransportCargoContainersMass
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class CargoContainerMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'cargo_containers',
            TransportCargoContainersMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER,
                Aircraft.CrewPayload.CARGO_MASS,
                Aircraft.CrewPayload.BAGGAGE_MASS,
            ],
            output_keys=Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
            rtol=1e-10,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class CargoContainerMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.anti_icing as antiicing

        antiicing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.anti_icing as antiicing

        antiicing.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'cargo_containers',
            TransportCargoContainersMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle', preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.CrewPayload.BAGGAGE_MASS, 5000.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class BWBCargoContainersMassTest(unittest.TestCase):
    """Test BWB cargo containers mass"""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob

        prob.model.add_subsystem(
            'cargo_containers',
            TransportCargoContainersMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER, val=1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.CrewPayload.BAGGAGE_MASS, val=20592.0, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        tol = 1e-8
        assert_near_equal(self.prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 3850.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
