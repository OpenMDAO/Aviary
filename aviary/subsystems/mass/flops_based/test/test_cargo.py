import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.cargo import PayloadGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import do_validation_test, print_case
from aviary.variable_info.variables import Aircraft

cargo_test_data = {}
cargo_test_data['1'] = AviaryValues(
    {
        Aircraft.CrewPayload.MISC_CARGO: (2000.0, 'lbm'),  # custom
        Aircraft.CrewPayload.WING_CARGO: (1000.0, 'lbm'),  # custom
        Aircraft.CrewPayload.BAGGAGE_MASS: (9200.0, 'lbm'),  # custom
        Aircraft.CrewPayload.PASSENGER_MASS: (33120.0, 'lbm'),  # custom
        Aircraft.CrewPayload.CARGO_MASS: (3000.0, 'lbm'),  # custom
        Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS: (45320.0, 'lbm'),  # custom
    }
)

cargo_data_sets = [key for key in cargo_test_data]


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
                Aircraft.CrewPayload.PASSENGER_MASS,
                Aircraft.CrewPayload.CARGO_MASS,
                Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            ],
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
