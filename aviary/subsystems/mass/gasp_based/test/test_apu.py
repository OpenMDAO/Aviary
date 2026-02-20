import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.mass.gasp_based.apu import APUMass

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


class ApuTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'apu',
            APUMass(),
            promotes=['*'],
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(
            self.prob[Aircraft.APU.MASS], 1077.969377, tol
        )  # large_single_aisle_1_GASP.csv


class ApuTestCase2(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'apu',
            APUMass(),
            promotes=['*'],
        )

        import aviary.subsystems.mass.gasp_based.apu as apu

        apu.GRAV_ENGLISH_LBM = 1.1

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.apu as apu

        apu.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(
            self.prob[Aircraft.APU.MASS], 979.97216091, tol
        )  # large_single_aisle_1_GASP.csv


class ApuTestCase3(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=5, units='unitless'
        )  # large_single_aisle_1_GASP.csv

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'apu',
            APUMass(),
            promotes=['*'],
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.APU.MASS], 0, tol)  # large_single_aisle_1_GASP.csv


if __name__ == '__main__':
    unittest.main()
