import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.instruments import TransportInstrumentMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission


class TransportInstrumentsMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.CrewPayload.NUM_FLIGHT_CREW: inputs.get_val(
                Aircraft.CrewPayload.NUM_FLIGHT_CREW
            ),
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Mission.Constraints.MAX_MACH: inputs.get_val(Mission.Constraints.MAX_MACH),
        }

        prob.model.add_subsystem(
            'instruments_tests',
            TransportInstrumentMass(**options),
            promotes_outputs=[
                Aircraft.Instruments.MASS,
            ],
            promotes_inputs=[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Instruments.MASS_SCALER],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Instruments.MASS_SCALER],
            output_keys=Aircraft.Instruments.MASS,
            tol=1e-3,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class TransportInstrumentsMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.instruments as instruments

        instruments.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.instruments as instruments

        instruments.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        inputs = get_flops_inputs('AdvancedSingleAisle', preprocess=True)

        options = {
            Aircraft.CrewPayload.NUM_FLIGHT_CREW: inputs.get_val(
                Aircraft.CrewPayload.NUM_FLIGHT_CREW
            ),
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Mission.Constraints.MAX_MACH: inputs.get_val(Mission.Constraints.MAX_MACH),
        }

        prob.model.add_subsystem(
            'instruments_tests',
            TransportInstrumentMass(**options),
            promotes_outputs=[
                Aircraft.Instruments.MASS,
            ],
            promotes_inputs=[Aircraft.Fuselage.PLANFORM_AREA, Aircraft.Instruments.MASS_SCALER],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.PLANFORM_AREA, 1500.0, 'ft**2')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
