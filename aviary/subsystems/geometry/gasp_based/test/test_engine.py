import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.gasp_based.engine import (
    GASPEngineSize,
    GASPEngineSizeGroup,
    PercentNotInFuselage,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import extract_options, setup_model_options
from aviary.variable_info.variables import Aircraft, Mission


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
        aviary_options.set_val(Aircraft.Engine.INLET_AREA_COEFFICIENT, 0.000301265)

        self.prob.model.add_subsystem('engsz', GASPEngineSize(), promotes=['*'])

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400.0, units='lbm')
        self.prob.model.set_input_defaults('percent_exposed', 1.0)
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_large_single_aisle_1_defaults(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_DIAMETER], 7.25002007, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.50004014, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.SURFACE_AREA], 330.26175625, tol)

    def test_partials(self):
        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class BWBTestEngine(unittest.TestCase):
    """Test engine size using GASPEngineSize class and BWB data"""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))

        self.prob.model.add_subsystem('engsz', GASPEngineSize(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        self.prob.model.set_input_defaults('percent_exposed', 1.0)
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2205, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.3588, units='unitless')

        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 1e-6
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_DIAMETER], 5.33382144, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 7.24759657, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.SURFACE_AREA], 121.44575974, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class ElectricTestCaseMultiEngine(unittest.TestCase):
    def test_case_multiengine(self):
        prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 4]))
        aviary_options.set_val(Aircraft.Engine.INLET_AREA_COEFFICIENT, [0.0003, 0.0002])

        prob.model.add_subsystem('cable', GASPEngineSize(), promotes=['*'])

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400.0, units='lbm')
        prob.model.set_input_defaults('percent_exposed', [1.0, 1.0])
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

        assert_near_equal(prob[Aircraft.Nacelle.AVG_DIAMETER], [7.23478278, 3.40843509], tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_LENGTH], [14.46956556, 7.53264155], tol)
        assert_near_equal(prob[Aircraft.Nacelle.SURFACE_AREA], [328.875, 80.65888272], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class GASPEngineSizeGroupTestCase(unittest.TestCase):
    """this is the GASP BWB test case"""

    def setUp(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, [2])

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            GASPEngineSizeGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE, 0.0, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2205, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.3588, units='unitless')

        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        Testing GASP data case:
        Aircraft.Nacelle.AVG_DIAMETER -- DBARN = 5.3338151
        Aircraft.Nacelle.AVG_LENGTH -- ELN = 7.24758816
        Aircraft.Nacelle.SURFACE_AREA -- SN/2 = 121.445763 (for one engine)
        """
        self.prob.run_model()

        tol = 1e-6
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_DIAMETER], 5.33382144, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 7.24759657, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.SURFACE_AREA], 121.44575974, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
