import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.energy_state.phases.build_takeoff import Takeoff
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class TakeoffPhaseTest(unittest.TestCase):
    """Test takeoff phase builder."""

    def test_case1(self):
        takeoff_options = Takeoff(airport_altitude=0)  # ft

        use_detailed = False
        takeoff = takeoff_options.build_phase(use_detailed=use_detailed)

        prob = om.Problem()
        prob.model = takeoff
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units='ft**2')
        prob.model.set_input_defaults(Mission.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 200000, units='lbf'
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val(Mission.Takeoff.LIFT_OVER_DRAG, 2)

        options = get_option_defaults()
        options.set_val(Mission.SEA_LEVEL_DENSITY, 0.0023769, units='slug/ft**3')
        setup_model_options(prob, options)
        prob.run_model()

        partial_data = prob.check_partials(
            out_stream=None, method='cs', compact_print=False, excludes=['*atmosphere*']
        )
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

        tol = 1e-5
        assert_near_equal(prob[Mission.Takeoff.GROUND_DISTANCE], 2811.50257923, tol)


if __name__ == '__main__':
    unittest.main()
