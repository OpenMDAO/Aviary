import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.payload import PayloadGroup
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
from aviary.variable_info.variables import Aircraft

cargo_test_data = {}
cargo_test_data['1'] = AviaryValues(
    {
        Aircraft.CrewPayload.MISC_CARGO: (2000.0, 'lbm'),  # custom
        Aircraft.CrewPayload.WING_CARGO: (1000.0, 'lbm'),  # custom
        Aircraft.CrewPayload.BAGGAGE_MASS: (9200.0, 'lbm'),  # custom
        Aircraft.CrewPayload.PASSENGER_MASS_TOTAL: (33120.0, 'lbm'),  # custom
        Aircraft.CrewPayload.CARGO_MASS: (3000.0, 'lbm'),  # custom
        Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS: (45320.0, 'lbm'),  # custom
    }
)

cargo_data_sets = [key for key in cargo_test_data]

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS']


@use_tempdirs
class PayloadGroupTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(cargo_data_sets, name_func=print_case)
    def test_case(self, case_name):
        validation_data = cargo_test_data[case_name]
        prob = self.prob

        prob.model.add_subsystem(
            'cargo_passenger',
            PayloadGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = {
            Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER: (50, 'lbm'),
            Aircraft.CrewPayload.MASS_PER_PASSENGER: (180.0, 'lbm'),
            Aircraft.CrewPayload.NUM_PASSENGERS: 184,  # custom
        }

        prob.setup(check=False, force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=validation_data,
            output_validation_data=validation_data,
            input_keys=[Aircraft.CrewPayload.MISC_CARGO, Aircraft.CrewPayload.WING_CARGO],
            output_keys=[
                Aircraft.CrewPayload.BAGGAGE_MASS,
                Aircraft.CrewPayload.PASSENGER_MASS_TOTAL,
                Aircraft.CrewPayload.CARGO_MASS,
                Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            ],
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class BWBCargoMassTest(unittest.TestCase):
    """Test BWB cargo mass"""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'cargo_passenger',
            PayloadGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.CrewPayload.MISC_CARGO, Aircraft.CrewPayload.WING_CARGO],
            output_keys=[
                Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
                Aircraft.CrewPayload.BAGGAGE_MASS,
                Aircraft.CrewPayload.PASSENGER_MASS_TOTAL,
                Aircraft.CrewPayload.CARGO_MASS,
                Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            ],
            version=Version.BWB,
        )


if __name__ == '__main__':
    unittest.main()
