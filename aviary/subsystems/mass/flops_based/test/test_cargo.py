import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.cargo import CargoMass
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import do_validation_test, print_case
from aviary.variable_info.functions import setup_model_options
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


@use_tempdirs
class CargoMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(cargo_data_sets, name_func=print_case)
    def test_case(self, case_name):
        validation_data = cargo_test_data[case_name]
        prob = self.prob

        prob.model.add_subsystem(
            'cargo_passenger',
            CargoMass(),
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


@use_tempdirs
class BWBCargoMassTest(unittest.TestCase):
    """Test BWB cargo mass"""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, 44, units='lbm')
        aviary_options.set_val(Aircraft.CrewPayload.MASS_PER_PASSENGER, 165, units='lbm')
        aviary_options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, 468, units='unitless')
        prob = self.prob

        prob.model.add_subsystem(
            'cargo',
            CargoMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.CrewPayload.WING_CARGO, val=0.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.CrewPayload.MISC_CARGO, val=0.0, units='lbm')

        setup_model_options(self.prob, aviary_options)
        self.prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        tol = 1e-8
        assert_near_equal(self.prob[Aircraft.CrewPayload.PASSENGER_MASS], 77220.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.BAGGAGE_MASS], 20592.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 97812.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.CARGO_MASS], 0.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 97812.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
