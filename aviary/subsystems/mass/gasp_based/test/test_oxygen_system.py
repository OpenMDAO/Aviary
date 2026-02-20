import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.oxygen_system import OxygenSystemMass

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class OxygenSystemTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # arbitrary

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'oxygen_system',
            OxygenSystemMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400.0, units='lbm'
        )  # large_single_aisle_1_GASP

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.OxygenSystem.MASS], 50.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class OxygenSystemTestCase2(unittest.TestCase):
    """Gravity Modification"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # arbitrary

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'oxygen_system',
            OxygenSystemMass(),
            promotes=['*'],
        )

        import aviary.subsystems.mass.gasp_based.oxygen_system as oxygen_system

        oxygen_system.GRAV_ENGLISH_LBM = 1.1

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400.0, units='lbm'
        )  # large_single_aisle_1_GASP

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.oxygen_system as oxygen_system

        oxygen_system.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.OxygenSystem.MASS], 45.45454545, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class OxygenSystemTestCase3(unittest.TestCase):
    """BWB Parameters"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )  # arbitrary

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'oxygen_system',
            OxygenSystemMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=150000.0, units='lbm'
        )  # large_single_aisle_1_GASP

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.OxygenSystem.MASS], 50.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
