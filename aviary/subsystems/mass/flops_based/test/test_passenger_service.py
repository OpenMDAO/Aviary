import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.passenger_service import (
    AltPassengerServiceMass,
    PassengerServiceMass,
)
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
class PassengerServiceMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'passenger_service_weight',
            PassengerServiceMass(),
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
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER,
                Aircraft.Design.RANGE,
                Aircraft.Design.MAX_MACH,
            ],
            output_keys=Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            version=Version.TRANSPORT_and_BWB,
            tol=2e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class AlternatePassengerServiceMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'alternate_passenger_service_weight',
            AltPassengerServiceMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER,
            output_keys=Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
