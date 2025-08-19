import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.missions.height_energy_default import phase_info
from aviary.variable_info.variables import Aircraft, Mission


# @use_tempdirs
class OffDesignTestCase(unittest.TestCase):
    """Test off-design capability for both fallout and alternate missions."""

    def setUp(self):
        # run design case
        prob = self.prob = AviaryProblem(verbosity=1)
        phase_info['post_mission']['target_range'] = (2500.0, 'nmi')

        prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()
        prob.add_driver('SNOPT', max_iter=20)
        prob.add_design_variables()

        # Load optimization problem formulation
        # Detail which variables the optimizer can control
        prob.add_objective()
        prob.setup()
        prob.set_initial_guesses()
        prob.run_aviary_problem()

    def compare_results(self, comparison_prob):
        # compares provided problem with design problem
        var_list = [
            Mission.Design.RANGE,
            Mission.Summary.RANGE,
            Mission.Summary.FUEL_MASS,
            Mission.Summary.TOTAL_FUEL_MASS,
            Mission.Summary.OPERATING_MASS,
            Aircraft.CrewPayload.CARGO_MASS,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            Mission.Design.GROSS_MASS,
            Mission.Summary.GROSS_MASS,
        ]

        for var in var_list:
            assert_near_equal(comparison_prob.get_val(var), self.prob.get_val(var), tolerance=1e-7)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_fallout_mission_match(self):
        # run a fallout mission with no changes, essentially recreating the design mission with
        # different constraints/design variables
        prob_fallout = self.prob.run_off_design_mission(problem_type='fallout')
        self.compare_results(prob_fallout)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_fallout_mission_changed(self):
        # run a fallout mission with modified payload and gross mass (and therefore different fuel)
        prob_fallout = self.prob.run_off_design_mission(
            problem_type='fallout', cargo_mass=5000, mission_gross_mass=150_000
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Design.RANGE),
            self.prob.get_val(Mission.Design.RANGE),
            tolerance=1e-7,
        )
        assert_near_equal(prob_fallout.get_val(Mission.Summary.RANGE), 896.07, tolerance=1e-3)
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            25562.6,
            tolerance=1e-3,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            14196.8,
            tolerance=1e-3,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            97778.2,
            tolerance=1e-3,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            5000,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            38025,
            tolerance=1e-3,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            self.prob.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            150000,
            tolerance=1e-7,
        )

    @require_pyoptsparse(optimizer='SNOPT')
    def test_alternate_mission_match(self):
        # run an alternate mission with no changes, essentially recreating the design mission with
        # different constraints/design variables
        prob_alternate = self.prob.run_off_design_mission(problem_type='alternate')
        self.compare_results(prob_alternate)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_alternate_mission_changed(self):
        # run an alternate mission with modified range and payload
        prob_alternate = self.prob.run_off_design_mission(
            problem_type='alternate', cargo_mass=2500, mission_range=1800
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Design.RANGE),
            self.prob.get_val(Mission.Design.RANGE),
            tolerance=1e-7,
        )
        assert_near_equal(prob_alternate.get_val(Mission.Summary.RANGE), 1800, tolerance=1e-7)
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            28587.6,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            23797.6,
            tolerance=1e-3,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            97253.22,
            tolerance=1e-3,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            2500,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            38025,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            self.prob.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            159075.8,
            tolerance=1e-3,
        )


if __name__ == '__main__':
    unittest.main()
    # test = OffDesignTestCase()
    # test.setUp()
    # test.test_fallout_mission_changed()
