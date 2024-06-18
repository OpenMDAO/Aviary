import unittest

import numpy as np
import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.nacelle import NacelleMass
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft
from aviary.utils.functions import get_path
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal


class NacelleMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "nacelle",
            NacelleMass(aviary_options=get_flops_inputs(case_name, preprocess=True)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            case_name,
            input_keys=[Aircraft.Nacelle.AVG_DIAMETER,
                        Aircraft.Nacelle.AVG_LENGTH,
                        Aircraft.Nacelle.MASS_SCALER,
                        Aircraft.Engine.SCALED_SLS_THRUST],
            output_keys=Aircraft.Nacelle.MASS)

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_case_multiengine(self):
        prob = om.Problem()

        options = AviaryValues()

        options.set_val(Aircraft.Engine.NUM_ENGINES, 4)
        options.set_val(Aircraft.Engine.DATA_FILE, get_path(
            'models/engines/turbofan_28k.deck'))
        engineModel1 = EngineDeck(options=options)
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engineModel2 = EngineDeck(options=options)
        engineModel3 = EngineDeck(options=options)

        preprocess_propulsion(options, [engineModel1, engineModel2, engineModel3])

        prob.model.add_subsystem('nacelle_mass', NacelleMass(
            aviary_options=options), promotes=['*'])
        prob.setup(force_alloc_complex=True)
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER,
                     np.array([5.0, 3.0, 8.0]), units='ft')
        prob.set_val(Aircraft.Nacelle.AVG_LENGTH,
                     np.array([14.0, 13.0, 12.0]), units='ft')
        prob.set_val(Aircraft.Nacelle.MASS_SCALER,
                     np.array([1.0] * 3), units='unitless')
        prob.set_val(Aircraft.Engine.SCALED_SLS_THRUST,
                     np.array([28000] * 3), units='lbf')

        prob.run_model()

        nacelle_mass = prob.get_val(Aircraft.Nacelle.MASS, 'lbm')
        nacelle_mass_expected = np.array([2793.02569, 778.05716, 1915.21762])
        assert_near_equal(nacelle_mass, nacelle_mass_expected, tolerance=1e-8)

        partial_data = prob.check_partials(
            out_stream=None, compact_print=True, show_only_incorrect=True, form='central', method="fd")
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
