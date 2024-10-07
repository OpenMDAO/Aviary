import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.engine import EngineMass
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.utils.functions import get_path
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Settings


class EngineMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "engine_mass",
            EngineMass(aviary_options=get_flops_inputs(case_name, preprocess=True)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Engine.SCALED_SLS_THRUST,
                        Aircraft.Engine.MASS,
                        Aircraft.Engine.ADDITIONAL_MASS,
                        Aircraft.Propulsion.TOTAL_ENGINE_MASS],
            output_keys=[Aircraft.Engine.MASS,
                         Aircraft.Engine.ADDITIONAL_MASS,
                         Aircraft.Propulsion.TOTAL_ENGINE_MASS],
            list_inputs=True,
            list_outputs=True,
            rtol=1e-10)

    def test_case_2(self):
        # arbitrary case to trigger both types of scaling equations, multiengine
        prob = om.Problem()

        options = AviaryValues()

        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.REFERENCE_MASS, 6000, units='lbm')
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        options.set_val(Aircraft.Engine.SCALE_MASS, True)
        options.set_val(Aircraft.Engine.MASS_SCALER, 1.15)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.9)
        options.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
        options.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)
        options.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)

        options.set_val(Aircraft.Engine.DATA_FILE, get_path(
            'models/engines/turbofan_28k.deck'))
        engine = EngineDeck(options=options)
        options.set_val(Aircraft.Engine.SCALE_MASS, False)
        engine2 = EngineDeck(name='engine2', options=options)
        options.set_val(Aircraft.Engine.MASS_SCALER, 0.2)
        options.set_val(Aircraft.Engine.SCALE_MASS, True)
        engine3 = EngineDeck(name='engine3', options=options)
        preprocess_propulsion(options, [engine, engine2, engine3])

        prob.model.add_subsystem('engine_mass', EngineMass(
            aviary_options=options), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Engine.SCALED_SLS_THRUST,
                     np.array([28000.0, 28000.0, 28000.0]), units='lbf')
        # Pull value from the processed options.
        val, units = options.get_item(Aircraft.Engine.MASS_SCALER)
        prob.set_val(Aircraft.Engine.MASS_SCALER, val, units=units)

        prob.run_model()

        mass = prob.get_val(Aircraft.Engine.MASS, 'lbm')
        total_mass = prob.get_val(Aircraft.Propulsion.TOTAL_ENGINE_MASS, 'lbm')
        additional_mass = prob.get_val(Aircraft.Engine.ADDITIONAL_MASS, 'lbm')

        mass_expected = np.array([5779.16494294, 6000.0, 5814.38])
        total_mass_expected = np.array([35187.08988589])
        additional_mass_expected = np.array([5201.24844865, 5400., 5232.942])

        assert_near_equal(mass, mass_expected, tolerance=1e-10)
        assert_near_equal(total_mass, total_mass_expected, tolerance=1e-10)
        assert_near_equal(additional_mass, additional_mass_expected, tolerance=1e-10)

        partial_data = prob.check_partials(
            out_stream=None, compact_print=True, show_only_incorrect=True, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
