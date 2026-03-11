import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.instruments import InstrumentMass

from aviary.variable_info.enums import GASPEngineType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class InstrumentTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.TURBOJET], units='unitless'
        )  # arbitrary
        options.set_val(
            Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=2, units='unitless'
        )  # large_single_aisle_1_GASP.csv

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'instruments',
            InstrumentMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units='ft'
        )  # arbitrary - 128
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400.0, units='lbm'
        )  # large_single_aisle_1_GASP.csv
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.83, units='ft')  # arbitrary
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )  # generic_bwb_gasp.csv - 0.116

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Instruments.MASS], 547.50860102, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class InstrumentTestCase2(unittest.TestCase):
    """BWB Parameters"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.RECIP_CARB], units='unitless'
        )  # arbitrary
        options.set_val(
            Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=2, units='unitless'
        )  # large_single_aisle_1_GASP.csv

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'instruments',
            InstrumentMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=71.5245514, units='ft'
        )  # arbitrary - 128
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=150000.0, units='lbm'
        )  # large_single_aisle_1_GASP.csv
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=146.38501, units='ft'
        )  # arbitrary
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.116, units='unitless'
        )  # generic_bwb_gasp.csv - 0.116

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Instruments.MASS], 917.20099314, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
