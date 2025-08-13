import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.gasp_based.electric import CableSize
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class ElectricTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 2)

        self.prob.model.add_subsystem('cable', CableSize(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, 0.35, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, 128, units='ft'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, 10, units='ft'
        )  # not actual GASP value

        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Electrical.HYBRID_CABLE_LENGTH], 64.8, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class ElectricTestCaseMultiEngine(unittest.TestCase):
    def test_case_multiengine(self):
        prob = om.Problem()

        aviary_options = AviaryValues()
        # aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 4]))
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 6)

        prob.model.add_subsystem('cable', CableSize(), promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, np.array([0.35, 0.2, 0.6]), units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 128, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 10, units='ft')

        setup_model_options(prob, aviary_options)

        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Electrical.HYBRID_CABLE_LENGTH], 167.2, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
