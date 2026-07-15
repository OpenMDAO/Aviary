import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary import constants
from aviary.mission.energy_state.phases.simplified_landing import LandingCalc, LandingGroup
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class LandingCalcTest(unittest.TestCase):
    """Test computation in LandingCalc class (the simplified landing)."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'land',
            LandingCalc(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.FINAL_MASS, val=152800.0, units='lbm')
        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, val=constants.RHO_SEA_LEVEL_METRIC, units='kg/m**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.0, units='ft**2')
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=3, units='unitless'
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(self.prob[Mission.Landing.GROUND_DISTANCE], 6403.64963504, tol)
        assert_near_equal(
            self.prob.get_val(Mission.Landing.INITIAL_VELOCITY, units='kn'), 136.22914933, tol
        )

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class LandingGroupTest(unittest.TestCase):
    """Test the computation of LandingGroup."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'land',
            LandingGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.FINAL_MASS, val=152800.0, units='lbm')
        self.prob.model.set_input_defaults(Mission.Landing.INITIAL_ALTITUDE, val=35, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.0, units='ft**2')
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=3, units='unitless'
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(self.prob[Mission.Landing.GROUND_DISTANCE], 6407.65299289, tol)
        assert_near_equal(
            self.prob.get_val(Mission.Landing.INITIAL_VELOCITY, units='kn'), 136.29923391, tol
        )

        partial_data = self.prob.check_partials(
            out_stream=None, excludes=['*.standard_atmosphere'], method='cs'
        )
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
