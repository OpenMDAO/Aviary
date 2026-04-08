import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

import aviary.api as av
from aviary.models.missions.two_dof_default import phase_info
from aviary.variable_info.enums import ProblemType, Verbosity


class TwoDOFTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sized_mass = 171044.0
        self.sized_range = 3675
        self.phase_info = deepcopy(phase_info)


@use_tempdirs
class TestOffDesign(TwoDOFTestCase):
    """
    Build the model using a large single aisle commercial transport aircraft data using
    GASP mass method and TWO_DEGREES_OF_FREEDOM mission method. Run a OFF_DESIGN_MAX_RANGE mission to test off design.
    """

    @require_pyoptsparse(optimizer='IPOPT')
    def test_off_design_IPOPT(self):
        # OFF_DESIGN_MAX_RANGE Mission
        prob_off_design_max_range = av.AviaryProblem()
        prob_off_design_max_range.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            self.phase_info,
            verbosity=Verbosity.QUIET,
        )

        prob_off_design_max_range.problem_type = ProblemType.OFF_DESIGN_MAX_RANGE
        prob_off_design_max_range.aviary_inputs.set_val(
            'problem_type', ProblemType.OFF_DESIGN_MAX_RANGE, units='unitless'
        )
        prob_off_design_max_range.aviary_inputs.set_val(
            'aircraft:design:gross_mass', self.sized_mass, units='lbm'
        )
        prob_off_design_max_range.aviary_inputs.set_val(
            'mission:gross_mass', self.sized_mass, units='lbm'
        )

        prob_off_design_max_range.check_and_preprocess_inputs()

        prob_off_design_max_range.build_model()
        prob_off_design_max_range.add_driver('IPOPT', max_iter=100)
        prob_off_design_max_range.add_design_variables()
        prob_off_design_max_range.add_objective()
        prob_off_design_max_range.setup()
        prob_off_design_max_range.run_aviary_problem()

        # Alternate Mission
        prob_alternate = av.AviaryProblem()
        prob_alternate.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            self.phase_info,
            verbosity=Verbosity.QUIET,
        )
        prob_alternate.problem_type = ProblemType.OFF_DESIGN_MIN_FUEL
        prob_alternate.aviary_inputs.set_val(
            'problem_type', ProblemType.OFF_DESIGN_MIN_FUEL, units='unitless'
        )

        prob_alternate.aviary_inputs.set_val(
            'aircraft:design:gross_mass', self.sized_mass, units='lbm'
        )
        prob_alternate.aviary_inputs.set_val('mission:gross_mass', self.sized_mass, units='lbm')

        prob_alternate.check_and_preprocess_inputs()
        prob_alternate.build_model()
        prob_alternate.add_driver('IPOPT', max_iter=100)
        prob_alternate.add_design_variables()
        prob_alternate.add_objective()
        prob_alternate.setup()
        prob_alternate.run_aviary_problem()

        off_design_max_range_range = prob_off_design_max_range.get_val(av.Mission.RANGE)
        alternate_mass = prob_alternate.get_val(av.Mission.GROSS_MASS)
        assert_near_equal(off_design_max_range_range, self.sized_range, tolerance=0.02)
        assert_near_equal(alternate_mass, self.sized_mass, tolerance=0.02)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_SNOPT(self):
        # off_design_max_range Mission
        prob_off_design_max_range = av.AviaryProblem()
        prob_off_design_max_range.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            self.phase_info,
            verbosity=Verbosity.QUIET,
        )

        prob_off_design_max_range.problem_type = ProblemType.OFF_DESIGN_MAX_RANGE
        prob_off_design_max_range.aviary_inputs.set_val(
            'problem_type', ProblemType.OFF_DESIGN_MAX_RANGE, units='unitless'
        )
        prob_off_design_max_range.aviary_inputs.set_val(
            'aircraft:design:gross_mass', self.sized_mass, units='lbm'
        )
        prob_off_design_max_range.aviary_inputs.set_val(
            'mission:gross_mass', self.sized_mass, units='lbm'
        )

        prob_off_design_max_range.check_and_preprocess_inputs()
        prob_off_design_max_range.build_model()
        prob_off_design_max_range.add_driver('SNOPT', max_iter=100)
        prob_off_design_max_range.add_design_variables()
        prob_off_design_max_range.add_objective()
        prob_off_design_max_range.setup()
        prob_off_design_max_range.run_aviary_problem()

        # Alternate Mission
        prob_alternate = av.AviaryProblem()
        prob_alternate.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            self.phase_info,
            verbosity=Verbosity.QUIET,
        )
        prob_alternate.problem_type = ProblemType.OFF_DESIGN_MIN_FUEL
        prob_alternate.aviary_inputs.set_val(
            'problem_type', ProblemType.OFF_DESIGN_MIN_FUEL, units='unitless'
        )

        prob_alternate.aviary_inputs.set_val(
            'aircraft:design:gross_mass', self.sized_mass, units='lbm'
        )
        prob_alternate.aviary_inputs.set_val('mission:gross_mass', self.sized_mass, units='lbm')

        prob_alternate.check_and_preprocess_inputs()
        prob_alternate.build_model()
        prob_alternate.add_driver('SNOPT', max_iter=100)
        prob_alternate.add_design_variables()
        prob_alternate.add_objective()
        prob_alternate.setup()
        prob_alternate.run_aviary_problem()

        off_design_max_range_range = prob_off_design_max_range.get_val(av.Mission.RANGE)
        alternate_mass = prob_alternate.get_val(av.Mission.GROSS_MASS)
        assert_near_equal(off_design_max_range_range, self.sized_range, tolerance=0.02)
        assert_near_equal(alternate_mass, self.sized_mass, tolerance=0.02)


if __name__ == '__main__':
    unittest.main()
    # test = TestOffDesign()
    # test.setUp()
    # test.test_off_design_SNOPT()
    # test.test_off_design_IPOPT()
