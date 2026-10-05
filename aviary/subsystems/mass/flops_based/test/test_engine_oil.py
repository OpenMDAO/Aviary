import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.engine_oil import AltEngineOilMass, TransportEngineOilMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class TransportEngineOilMassTest(unittest.TestCase):
    """Tests transport/GA engine oil mass calculation."""

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
        }

        prob.model.add_subsystem(
            'engine_oil',
            TransportEngineOilMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = options

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER,
                Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
            ],
            output_keys=[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS],
            version=Version.TRANSPORT_and_BWB,
            tol=4.0e-3,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class AltEngineOilMassTest(unittest.TestCase):
    """Tests alternate engine oil mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.CrewPayload.Design.NUM_PASSENGERS: inputs.get_val(
                Aircraft.CrewPayload.Design.NUM_PASSENGERS
            ),
        }

        prob.model.add_subsystem(
            'engine_oil', AltEngineOilMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )

        prob.model_options['*'] = options

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER,
            ],
            output_keys=[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS],
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
