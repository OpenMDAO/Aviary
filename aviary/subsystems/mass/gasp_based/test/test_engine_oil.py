import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.engine_oil import EngineOilMass

from aviary.variable_info.enums import GASPEngineType, Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Settings


class ElectricalTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.TURBOJET], units='unitless'
        )  # arbitrarily set
        options.set_val(
            Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=2, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(Settings.VERBOSITY, val=0, units='unitless')  # arbitrarily set

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine_oil_mass',
            EngineOilMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500, units='lbf'
        )  # generic_BWB_GASP.csv - 11.45

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 24.12366, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
