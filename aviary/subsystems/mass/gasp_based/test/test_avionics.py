import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.avionics import AvionicsMass

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class AvionicsTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )  # arbitrarily set

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'avionics',
            AvionicsMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Avionics.MASS], 1514.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class AvionicsTestCase2(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )  # arbitrarily set

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'avionics',
            AvionicsMass(),
            promotes=['*'],
        )

        import aviary.subsystems.mass.gasp_based.avionics as avionics

        avionics.GRAV_ENGLISH_LBM = 1.1

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.avionics as avionics

        avionics.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Avionics.MASS], 1376.36363636, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class AvionicsTestCase3(unittest.TestCase):
    """this is a mix of BWB conditions with lower passengers and discontinuities off test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=5, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # arbitrarily set

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'avionics',
            AvionicsMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=150000, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Avionics.MASS], 340.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
