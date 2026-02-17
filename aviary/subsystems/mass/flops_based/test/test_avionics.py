import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.avionics import TransportAvionicsMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
    Version,
)
from aviary.variable_info.variables import Aircraft, Mission

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS']


@use_tempdirs
class TransportAvionicsMassTest(unittest.TestCase):
    """Tests transport/GA avionics mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'avionics',
            TransportAvionicsMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Avionics.MASS_SCALER,
                Aircraft.Fuselage.PLANFORM_AREA,
                Mission.Design.RANGE,
            ],
            output_keys=Aircraft.Avionics.MASS,
            aviary_option_keys=[Aircraft.CrewPayload.NUM_FLIGHT_CREW],
            tol=2.0e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class TransportAvionicsMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.avionics as avionics

        avionics.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.avionics as avionics

        avionics.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'avionics',
            TransportAvionicsMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle', preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.PLANFORM_AREA, 1500.0, 'ft**2')
        prob.set_val(Mission.Design.RANGE, 3500.0, 'nmi')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class BWBTransportAvionicsMassTest(unittest.TestCase):
    """Test fuselage mass calculation for BWB data."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'avionics',
            TransportAvionicsMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Avionics.MASS_SCALER,
                Aircraft.Fuselage.PLANFORM_AREA,
                Mission.Design.RANGE,
            ],
            output_keys=Aircraft.Avionics.MASS,
            version=Version.BWB,
            aviary_option_keys=[Aircraft.CrewPayload.NUM_FLIGHT_CREW],
            tol=2.0e-4,
        )


if __name__ == '__main__':
    unittest.main()
