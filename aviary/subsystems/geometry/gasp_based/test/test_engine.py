import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.engine import (
    BWBEngineSizeGroup,
    EngineSize,
    PercentNotInFuselage,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import extract_options, setup_model_options
from aviary.variable_info.variables import Aircraft


class TestPercentNotInFuselage(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, [3])

        self.prob.model.add_subsystem('perc', PercentNotInFuselage(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE, 0.0, units='unitless'
        )

        setup_model_options(self.prob, aviary_options)
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_zero_buried(self):
        self.prob.run_model()
        tol = 1e-7
        assert_near_equal(self.prob['percent_exposed'], 1.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_half_buried(self):
        """Test in the range (epsilon, 1.0 - epsilon)."""
        self.prob.set_val(
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE, val=0.5, units='unitless'
        )
        self.prob.run_model()
        tol = 1e-7
        assert_near_equal(self.prob['percent_exposed'], 0.5, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_left_buried(self):
        """Test in the range (0.0, epsilon)."""
        self.prob.set_val(
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE, val=0.03, units='unitless'
        )
        self.prob.run_model()
        tol = 1e-7
        assert_near_equal(self.prob['percent_exposed'], 0.89181881, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_right_buried(self):
        """Test in the range (1.0 - epsilon, 1.0)."""
        self.prob.set_val(
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE, val=0.97, units='unitless'
        )
        self.prob.run_model()
        tol = 1e-7

        assert_near_equal(self.prob['percent_exposed'], 0.10818119, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TestEngine(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))

        self.prob.model.add_subsystem('engsz', EngineSize(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, val=1.028233)
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_large_sinle_aisle_1_defaults(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_DIAMETER], 7.35163, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.70326, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.SURFACE_AREA], 339.58389, tol)

    def test_partials(self):
        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class ElectricTestCaseMultiEngine(unittest.TestCase):
    def test_case_multiengine(self):
        prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 4]))

        prob.model.add_subsystem('cable', EngineSize(), promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.Engine.REFERENCE_DIAMETER, np.array([5.8, 8.2]), units='ft'
        )
        prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, val=np.array([1.028233, 0.9]))
        prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, np.array([1.25, 1.02]), units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.FINENESS, np.array([2, 2.21]), units='unitless'
        )

        prob.model_options['*'] = extract_options(aviary_options)

        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        tol = 1e-5

        assert_near_equal(prob[Aircraft.Nacelle.AVG_DIAMETER], [7.35163, 7.9347871], tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_LENGTH], [14.70326, 17.5358795], tol)
        assert_near_equal(prob[Aircraft.Nacelle.SURFACE_AREA], [339.58389, 437.13210486], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class BWBEngineSizeGroupTestCase(unittest.TestCase):
    """this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix."""

    def setUp(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, [2])

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            BWBEngineSizeGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE, 0.0, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2205, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.3588, units='unitless')

        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        Testing GASP data case:
        Aircraft.Nacelle.AVG_DIAMETER -- DBARN = 6.95
        Aircraft.Nacelle.AVG_LENGTH -- ELN = 9.44
        Aircraft.Nacelle.SURFACE_AREA -- SN/2 = 205.965 (for one engine)
        Note: Aviary uses reference diameter which is different from GASP. So, the outpus are
        not the same.
        """
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_DIAMETER], 7.17813375, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 9.75364814, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.SURFACE_AREA], 219.95229788, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
