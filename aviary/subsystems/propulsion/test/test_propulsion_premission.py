import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.models.multi_engine_single_aisle.multi_engine_single_aisle_data import (
    engine_1_inputs,
    engine_2_inputs,
)
from aviary.subsystems.propulsion.propulsion_premission import PropulsionPreMission, PropulsionSum
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.preprocessors import preprocess_options
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Settings


class PropulsionPreMissionTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):
        options = get_flops_inputs('LargeSingleAisle2FLOPS')
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))

        self.prob.model = PropulsionPreMission(
            aviary_options=options, engine_models=[build_engine_deck(options)]
        )

        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, np.ones(1))

        setup_model_options(self.prob, options)

        self.prob.setup(force_alloc_complex=True)
        # self.prob.set_val(Aircraft.Engine.SCALED_SLS_THRUST, options.get_val(
        #     Aircraft.Engine.SCALED_SLS_THRUST, units='lbf'))

        self.prob.run_model()

        sls_thrust = self.prob.get_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST)

        expected_sls_thrust = np.array([54602.0])

        assert_near_equal(sls_thrust, expected_sls_thrust, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_multi_engine(self):
        options = get_flops_inputs('MultiEngineSingleAisle')
        options.set_val(Settings.VERBOSITY, 0)

        engine1 = build_engine_deck(engine_1_inputs)
        engine2 = build_engine_deck(engine_2_inputs)
        engine_models = [engine1, engine2]
        preprocess_options(options, engine_models=engine_models)

        setup_model_options(self.prob, options, engine_models=engine_models)

        model = self.prob.model
        prop = PropulsionPreMission(aviary_options=options, engine_models=engine_models)
        model.add_subsystem('core_propulsion', prop, promotes=['*'])

        setup_model_options(self.prob, options, engine_models=engine_models)

        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, np.ones(2) * 0.5)

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(
            Aircraft.Engine.SCALED_SLS_THRUST,
            options.get_val(Aircraft.Engine.SCALED_SLS_THRUST, units='lbf'),
        )

        self.prob.run_model()

        sls_thrust = self.prob.get_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST)

        expected_sls_thrust = np.array([51128.6])

        assert_near_equal(sls_thrust, expected_sls_thrust, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_propulsion_sum(self):
        options = {
            Aircraft.Engine.NUM_ENGINES: np.array([1, 2, 5]),
        }
        self.prob.model = om.Group()
        self.prob.model.add_subsystem('propsum', PropulsionSum(**options), promotes=['*'])

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(
            Aircraft.Engine.SCALED_SLS_THRUST, np.array([1000, 3000, 13200]), units='lbf'
        )

        self.prob.run_model()

        total_thrust = self.prob.get_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, units='lbf')

        expected_thrust = 73000.0

        assert_near_equal(total_thrust, expected_thrust, tolerance=1e-12)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
    # test = PropulsionPreMissionTest()
    # test.setUp()
    # test.test_case()
