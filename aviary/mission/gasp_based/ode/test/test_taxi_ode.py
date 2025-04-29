import unittest

import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from packaging import version

from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.mission.gasp_based.ode.taxi_ode import TaxiSegment
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic, Mission


class TaxiTestCase(unittest.TestCase):
    """Test computation of taxi group."""

    def setUp(self):
        self.prob = om.Problem()

        options = get_option_defaults()
        options.set_val(Mission.Taxi.DURATION, 0.1677, units='h')
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(options)]
        )

        self.prob.model = TaxiSegment(
            aviary_options=options, core_subsystems=default_mission_subsystems
        )

        setup_model_options(self.prob, options)

        self.prob.model.set_input_defaults(
            Mission.Takeoff.AIRPORT_ALTITUDE,
            0.0,
        )

    @unittest.skipIf(
        version.parse(openmdao.__version__) < version.parse('3.26'),
        'Skipping due to OpenMDAO version being too low (<3.26)',
    )
    def test_taxi(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        set_params_for_unit_tests(self.prob)

        self.prob.set_val(Mission.Takeoff.AIRPORT_ALTITUDE, 0, units='ft')
        self.prob.set_val(Mission.Taxi.MACH, 0.1, units='unitless')
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            -1512,
            units='lbm/h',
        )

        self.prob.run_model()

        testvals = {
            Dynamic.Vehicle.MASS: 175190.3,  # lbm
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*atmos*', '*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
