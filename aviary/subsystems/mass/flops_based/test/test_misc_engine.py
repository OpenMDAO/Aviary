import unittest

import numpy as np
import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.misc_engine import EngineMiscMass
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Settings
from aviary.utils.functions import get_path
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal


class MiscEngineMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "misc_mass",
            EngineMiscMass(aviary_options=get_flops_inputs(case_name, preprocess=True)),
            promotes_inputs=['*'],
            promotes_outputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Engine.ADDITIONAL_MASS,
                        Aircraft.Propulsion.MISC_MASS_SCALER,
                        Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
                        Aircraft.Propulsion.TOTAL_STARTER_MASS],
            output_keys=Aircraft.Propulsion.TOTAL_MISC_MASS)

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_case_multiengine(self):
        prob = om.Problem()

        options = AviaryValues()

        options.set_val(Aircraft.Engine.NUM_ENGINES, 4)
        options.set_val(Aircraft.Engine.DATA_FILE, get_path(
            'models/engines/turbofan_28k.deck'))
        options.set_val(Settings.VERBOSITY, 0)
        engineModel1 = EngineDeck(options=options)
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engineModel2 = EngineDeck(options=options)
        engineModel3 = EngineDeck(options=options)

        preprocess_propulsion(options, [engineModel1, engineModel2, engineModel3])

        prob.model.add_subsystem('misc_engine_mass', EngineMiscMass(
            aviary_options=options), promotes=['*'])
        prob.setup(force_alloc_complex=True)
        prob.set_val(Aircraft.Engine.ADDITIONAL_MASS,
                     np.array([100, 26, 30]), units='lbm')
        prob.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, 1.02, units='unitless')
        prob.set_val(Aircraft.Propulsion.TOTAL_STARTER_MASS, 50.0, units='lbm')
        prob.set_val(Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, 10.0, units='lbm')

        prob.run_model()

        total_misc_mass = prob.get_val(Aircraft.Propulsion.TOTAL_MISC_MASS, 'lbm')
        # manual computation of expected misc mass
        total_misc_mass_expected = (50 + (100*4 + 26*2 + 30*2) + 10)*1.02
        assert_near_equal(total_misc_mass, total_misc_mass_expected, tolerance=1e-10)

        partial_data = prob.check_partials(
            out_stream=None, compact_print=True, show_only_incorrect=True, form='central', method="fd")
        assert_check_partials(partial_data, atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
