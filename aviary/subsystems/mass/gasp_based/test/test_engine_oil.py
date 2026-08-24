import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.engine_oil import EngineOilMass
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import GASPEngineType, Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Settings


@use_tempdirs
class TestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def test_case1(self):
        options = AviaryValues()
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.TURBOJET], units='unitless'
        )  # arbitrarily set
        options.set_val(
            Aircraft.Engine.NUM_ENGINES, val=[2], units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(Settings.VERBOSITY, val=0, units='unitless')  # arbitrarily set

        prob = om.Problem()
        prob.model.add_subsystem(
            'engine_oil_mass',
            EngineOilMass(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500], units='lbf'
        )  # generic_BWB_GASP.csv - 11.45

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 342.6, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_turboprop_derivs(self):
        eng = EngineOilMass()
        eng.options[Aircraft.Engine.TYPE] = [GASPEngineType.TURBOPROP]
        eng.options[Aircraft.Engine.NUM_ENGINES] = [2]

        prob = om.Problem()
        prob.model.add_subsystem('engine_oil_mass', eng, promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500], units='lbf'
        )  # generic_BWB_GASP.csv - 11.45

        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_multiengine(self):
        eng = EngineOilMass()

        eng.options[Aircraft.Engine.TYPE] = [GASPEngineType.TURBOJET, GASPEngineType.TURBOPROP]
        eng.options[Aircraft.Engine.NUM_ENGINES] = [2, 3]

        prob = om.Problem()
        prob.model.add_subsystem('engine_oil_mass', eng, promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500, 14000], units='lbf'
        )  # generic_BWB_GASP.csv - 11.45

        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 1283.4, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class TestCase2(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        options = AviaryValues()
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.TURBOJET], units='unitless'
        )  # arbitrarily set
        options.set_val(
            Aircraft.Engine.NUM_ENGINES, val=[2], units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(Settings.VERBOSITY, val=0, units='unitless')  # arbitrarily set

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine_oil_mass',
            EngineOilMass(),
            promotes=['*'],
        )

        from aviary.subsystems.mass.gasp_based import engine_oil

        engine_oil.GRAV_ENGLISH_LBM = 1.1

        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500], units='lbf'
        )  # generic_BWB_GASP.csv - 11.45

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        from aviary.subsystems.mass.gasp_based import engine_oil

        engine_oil.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 311.45454545, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class TestCase3(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        options = AviaryValues()
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.RECIP_CARB], units='unitless'
        )  # arbitrarily set
        options.set_val(
            Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=2, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(Settings.VERBOSITY, val=0, units='unitless')  # arbitrarily set
        options.set_val(Aircraft.Engine.NUM_ENGINES, val=[2], units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine_oil_mass',
            EngineOilMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=19580.1602, units='lbf'
        )  # generic_BWB_GASP.csv - 11.45

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
