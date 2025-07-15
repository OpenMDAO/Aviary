import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.air_conditioning import AltAirCondMass, TransportAirCondMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft


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

        prob.model_options['*'] = get_flops_options(case_name)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.AirConditioning.MASS_SCALER,
                Aircraft.Avionics.MASS,
                Aircraft.Fuselage.MAX_HEIGHT,
                Aircraft.Fuselage.PLANFORM_AREA,
            ],
            output_keys=Aircraft.AirConditioning.MASS,
            aviary_option_keys=[Aircraft.CrewPayload.Design.NUM_PASSENGERS],
            version=Version.TRANSPORT,
            tol=3.0e-4,
            atol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class TransportAirCondMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.air_conditioning as ac

        ac.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.air_conditioning as ac

        ac.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'air_cond',
            TransportAirCondMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle')

        prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_SCALER, val=0.98094, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=2032.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuselage.MAX_HEIGHT, val=13.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.PLANFORM_AREA, val=1537.5, units='ft**2')

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


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
            prob,
            case_name,
            input_keys=Aircraft.AirConditioning.MASS_SCALER,
            output_keys=Aircraft.AirConditioning.MASS,
            aviary_option_keys=Aircraft.CrewPayload.Design.NUM_PASSENGERS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltAirCondMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.air_conditioning as ac

        ac.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.air_conditioning as ac

        ac.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'air_cond',
            AltAirCondMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle')

        prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_SCALER, val=0.98094, units='unitless'
        )
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
