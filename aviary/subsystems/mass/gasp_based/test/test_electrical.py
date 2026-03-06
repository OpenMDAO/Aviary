import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.electrical import ElectricalMass

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class ElectricalTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=2, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'electrical',
            ElectricalMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER, val=16.0, units='lbm'
        )  # generic_BWB_GASP.csv
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400.0, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Electrical.MASS], 3050.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class ElectricalTestCase2(unittest.TestCase):
    """Gravity Modification"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=2, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'electrical',
            ElectricalMass(),
            promotes=['*'],
        )

        import aviary.subsystems.mass.gasp_based.electrical as electrical

        electrical.GRAV_ENGLISH_LBM = 1.1

        self.prob.model.set_input_defaults(
            Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER, val=16.0, units='lbm'
        )  # generic_BWB_GASP.csv
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400.0, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.electrical as electrical

        electrical.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Electrical.MASS], 3034.54545455, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class ElectricalTestCase3(unittest.TestCase):
    """BWB Parameters"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=2, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'electrical',
            ElectricalMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER, val=11.45, units='lbm'
        )  # generic_BWB_GASP.csv
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=150000.0, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Electrical.MASS], 1887.5, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
