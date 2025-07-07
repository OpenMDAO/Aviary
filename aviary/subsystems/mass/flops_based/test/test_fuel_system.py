import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.fuel_system import (
    AltFuelSystemMass,
    TransportFuelSystemMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission


class AltFuelSystemTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Fuel.NUM_TANKS: inputs.get_val(Aircraft.Fuel.NUM_TANKS),
        }

        prob.model.add_subsystem(
            'alt_fuel_sys_test',
            AltFuelSystemMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, Aircraft.Fuel.TOTAL_CAPACITY],
            output_keys=Aircraft.Fuel.FUEL_SYSTEM_MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltFuelSystemTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.fuel_system as fuel

        fuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.fuel_system as fuel

        fuel.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        inputs = get_flops_inputs('AdvancedSingleAisle', preprocess=True)

        options = {
            Aircraft.Fuel.NUM_TANKS: inputs.get_val(Aircraft.Fuel.NUM_TANKS),
        }

        prob.model.add_subsystem(
            'alt_fuel_sys_test',
            AltFuelSystemMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 100.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class TransportFuelSystemTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Propulsion.TOTAL_NUM_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_ENGINES
            ),
            Mission.Constraints.MAX_MACH: inputs.get_val(Mission.Constraints.MAX_MACH),
        }

        prob.model.add_subsystem(
            'transport_fuel_sys_test',
            TransportFuelSystemMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, Aircraft.Fuel.TOTAL_CAPACITY],
            output_keys=Aircraft.Fuel.FUEL_SYSTEM_MASS,
            version=Version.TRANSPORT,
            tol=8.0e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class TransportFuelSystemTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.fuel_system as fuel

        fuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.fuel_system as fuel

        fuel.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        inputs = get_flops_inputs('AdvancedSingleAisle', preprocess=True)

        options = {
            Aircraft.Propulsion.TOTAL_NUM_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_ENGINES
            ),
            Mission.Constraints.MAX_MACH: inputs.get_val(Mission.Constraints.MAX_MACH),
        }

        prob.model.add_subsystem(
            'transport_fuel_sys_test',
            TransportFuelSystemMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 100.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
