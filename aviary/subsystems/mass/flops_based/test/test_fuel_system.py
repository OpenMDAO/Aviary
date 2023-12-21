import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.fuel_system import (AltFuelSystemMass,
                                                            TransportFuelSystemMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class AltFuelSystemTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "alt_fuel_sys_test",
            AltFuelSystemMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,
                        Aircraft.Fuel.TOTAL_CAPACITY],
            output_keys=Aircraft.Fuel.FUEL_SYSTEM_MASS,
            version=Version.ALTERNATE,)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class TransportFuelSystemTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "transport_fuel_sys_test",
            TransportFuelSystemMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,
                        Aircraft.Fuel.TOTAL_CAPACITY],
            output_keys=Aircraft.Fuel.FUEL_SYSTEM_MASS,
            version=Version.TRANSPORT,
            tol=8.0e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
