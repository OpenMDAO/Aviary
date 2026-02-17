import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.hydraulics import HydraulicsMass

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


class HydraulicsTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless'
        )  # large single aisle GASP

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'hydraulics',
            HydraulicsMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )  # generic_BWB
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )  # generic_BWB
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7511.0, units='lbm'
        )  # Pulled from old equipment and useful load test
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm'
        )  # Pulled from old equipment and useful load test

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Hydraulics.MASS], 1487.78, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class HydraulicsTestCase2(unittest.TestCase):
    """BWB Parameters"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless'
        )  # large single aisle GASP

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'hydraulics',
            HydraulicsMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.107, units='unitless'
        )  # generic_BWB
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.135, units='unitless'
        )  # generic_BWB
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7800.0, units='lbm'
        )  # Pulled from old equipment and useful load test
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=2115.19946, units='lbm'
        )  # Pulled from old equipment and useful load test

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Hydraulics.MASS], 1279.32634222, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
