import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.crew import FlightCrewMass, NonFlightCrewMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
    Version,
)
from aviary.variable_info.variables import Aircraft

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS']


@use_tempdirs
class NonFlightCrewMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'non_flight_crew',
            NonFlightCrewMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER,
            output_keys=Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
            atol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class FlightCrewMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'flight_crew',
            FlightCrewMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER,
            output_keys=Aircraft.CrewPayload.FLIGHT_CREW_MASS,
            atol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class BWBNonFlightCrewMassTest(unittest.TestCase):
    """Test non-flight crew mass calculation for BWB data."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'non_flight_crew',
            NonFlightCrewMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=False)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER,
            output_keys=Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
            version=Version.BWB,
            atol=1e-11,
        )


@use_tempdirs
class BWBFlightCrewMassTest(unittest.TestCase):
    """Test flight crew mass calculation for BWB data."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'flight_crew',
            FlightCrewMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER,
            output_keys=Aircraft.CrewPayload.FLIGHT_CREW_MASS,
            version=Version.BWB,
            atol=1e-11,
        )


if __name__ == '__main__':
    unittest.main()
