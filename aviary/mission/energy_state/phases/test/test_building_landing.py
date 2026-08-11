import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.energy_state.phases.build_landing import Landing
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class LandingPhaseTest(unittest.TestCase):
    """Test landing phase builder."""

    def test_case1(self):
        landing_options = Landing()

        use_detailed = False
        landing = landing_options.build_phase(use_detailed=use_detailed)

        prob = om.Problem()
        prob.model = landing
        prob.model.set_input_defaults(Mission.FINAL_MASS, val=150000.0, units='lbm')
        prob.model.set_input_defaults(Mission.Landing.INITIAL_ALTITUDE, val=0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.0, units='ft**2')
        prob.model.set_input_defaults(Mission.Landing.LIFT_COEFFICIENT_MAX, val=3, units='unitless')

        options = get_option_defaults()
        options.set_val(Mission.SEA_LEVEL_DENSITY, 0.0023769, units='slug/ft**3')
        setup_model_options(prob, options)
        prob.setup(force_alloc_complex=True)

        prob.set_val(
            Mission.FINAL_MASS,
            val=150_000,
        )

        prob.run_model()

        partial_data = prob.check_partials(
            out_stream=None, method='cs', compact_print=False, excludes=['*atmosphere*']
        )
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

        tol = 1e-6
        assert_near_equal(prob[Mission.Landing.GROUND_DISTANCE], 6332.13214907, tol)
        assert_near_equal(
            prob.get_val(Mission.Landing.INITIAL_VELOCITY, units='kn'), 134.97550621, tol
        )


if __name__ == '__main__':
    unittest.main()
