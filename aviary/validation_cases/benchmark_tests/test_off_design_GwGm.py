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
    GASP mass method and TWO_DEGREES_OF_FREEDOM mission method. Run a fallout mission to test off design.
    """

    @require_pyoptsparse(optimizer='IPOPT')
    def test_off_design_IPOPT(self):
        # Fallout Mission
        prob_fallout = av.AviaryProblem()
        prob_fallout.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            self.phase_info,
            verbosity=Verbosity.QUIET,
        )

        prob_fallout.problem_type = ProblemType.FALLOUT
        prob_fallout.aviary_inputs.set_val('problem_type', ProblemType.FALLOUT, units='unitless')
        prob_fallout.aviary_inputs.set_val(
            'mission:design:gross_mass', self.sized_mass, units='lbm'
        )
        prob_fallout.aviary_inputs.set_val(
            'mission:summary:gross_mass', self.sized_mass, units='lbm'
        )

        prob_fallout.check_and_preprocess_inputs()

        prob_fallout.build_model()
        prob_fallout.add_driver('IPOPT', max_iter=100)
        prob_fallout.add_design_variables()
        prob_fallout.add_objective()
        prob_fallout.setup()
        prob_fallout.run_aviary_problem()

        # Alternate Mission
        prob_alternate = av.AviaryProblem()
        prob_alternate.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            self.phase_info,
            verbosity=Verbosity.QUIET,
        )
        prob_alternate.problem_type = ProblemType.ALTERNATE
        prob_alternate.aviary_inputs.set_val(
            'problem_type', ProblemType.ALTERNATE, units='unitless'
        )

        prob_alternate.aviary_inputs.set_val(
            'mission:design:gross_mass', self.sized_mass, units='lbm'
        )
        prob_alternate.aviary_inputs.set_val(
            'mission:summary:gross_mass', self.sized_mass, units='lbm'
        )

        prob_alternate.check_and_preprocess_inputs()
        prob_alternate.build_model()
        prob_alternate.add_driver('IPOPT', max_iter=100)
        prob_alternate.add_design_variables()
        prob_alternate.add_objective()
        prob_alternate.setup()
        prob_alternate.run_aviary_problem()

        fallout_range = prob_fallout.get_val(av.Mission.Summary.RANGE)
        alternate_mass = prob_alternate.get_val(av.Mission.Summary.GROSS_MASS)
        assert_near_equal(fallout_range, self.sized_range, tolerance=0.02)
        assert_near_equal(alternate_mass, self.sized_mass, tolerance=0.02)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_SNOPT(self):
        # Fallout Mission
        prob_fallout = av.AviaryProblem()
        prob_fallout.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            self.phase_info,
            verbosity=Verbosity.QUIET,
        )

        prob_fallout.problem_type = ProblemType.FALLOUT
        prob_fallout.aviary_inputs.set_val('problem_type', ProblemType.FALLOUT, units='unitless')
        prob_fallout.aviary_inputs.set_val(
            'mission:design:gross_mass', self.sized_mass, units='lbm'
        )
        prob_fallout.aviary_inputs.set_val(
            'mission:summary:gross_mass', self.sized_mass, units='lbm'
        )

        prob_fallout.check_and_preprocess_inputs()
        prob_fallout.build_model()
        prob_fallout.add_driver('SNOPT', max_iter=100)
        prob_fallout.add_design_variables()
        prob_fallout.add_objective()
        prob_fallout.setup()
        prob_fallout.run_aviary_problem()

        # Alternate Mission
        prob_alternate = av.AviaryProblem()
        prob_alternate.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            self.phase_info,
            verbosity=Verbosity.QUIET,
        )
        prob_alternate.problem_type = ProblemType.ALTERNATE
        prob_alternate.aviary_inputs.set_val(
            'problem_type', ProblemType.ALTERNATE, units='unitless'
        )

        prob_alternate.aviary_inputs.set_val(
            'mission:design:gross_mass', self.sized_mass, units='lbm'
        )
        prob_alternate.aviary_inputs.set_val(
            'mission:summary:gross_mass', self.sized_mass, units='lbm'
        )

        prob_alternate.check_and_preprocess_inputs()
        prob_alternate.build_model()
        prob_alternate.add_driver('SNOPT', max_iter=100)
        prob_alternate.add_design_variables()
        prob_alternate.add_objective()
        prob_alternate.setup()
        prob_alternate.run_aviary_problem()

        fallout_range = prob_fallout.get_val(av.Mission.Summary.RANGE)
        alternate_mass = prob_alternate.get_val(av.Mission.Summary.GROSS_MASS)
        assert_near_equal(fallout_range, self.sized_range, tolerance=0.02)
        assert_near_equal(alternate_mass, self.sized_mass, tolerance=0.02)


if __name__ == '__main__':
    unittest.main()
    # test = TestOffDesign()
    # test.setUp()
    # test.test_off_design_SNOPT()
    # test.test_off_design_IPOPT()
