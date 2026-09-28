import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.air_conditioning import AltAirCondMass, TransportAirCondMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
    Version,
)
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class TransportAirCondMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'air_cond',
            TransportAirCondMass(),
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
                Aircraft.AirConditioning.MASS_SCALER,
                Aircraft.Avionics.MASS,
                Aircraft.Fuselage.MAX_HEIGHT,
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Design.MAX_MACH,
            ],
            output_keys=Aircraft.AirConditioning.MASS,
            aviary_option_keys=[Aircraft.CrewPayload.Design.NUM_PASSENGERS],
            version=Version.TRANSPORT_and_BWB,
            tol=3.0e-4,
            atol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class AltAirCondMassTest(unittest.TestCase):
    """Tests alternate air conditioning mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'air_cond',
            AltAirCondMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=Aircraft.AirConditioning.MASS_SCALER,
            output_keys=Aircraft.AirConditioning.MASS,
            aviary_option_keys=Aircraft.CrewPayload.Design.NUM_PASSENGERS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
