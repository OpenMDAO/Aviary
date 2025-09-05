import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.missions.height_energy_default import phase_info as energy_phase_info
from aviary.models.missions.two_dof_default import phase_info as twodof_phase_info
from aviary.variable_info.variables import Aircraft, Mission, Settings


@use_tempdirs
class TestHeightEnergyOffDesign(unittest.TestCase):
    """Test off-design capability for both fallout and alternate missions."""

    def setUp(self):
        # run design case
        prob = self.prob = AviaryProblem(verbosity=1)
        energy_phase_info['post_mission']['target_range'] = (2500.0, 'nmi')

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', energy_phase_info
        )
        # define passengers of every seat class so we can change their values later
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 169)
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 144)
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 15)
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 10)

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
        prob_var_list = [
            Mission.Design.RANGE,
            Mission.Summary.RANGE,
            Mission.Summary.FUEL_MASS,
            Mission.Summary.TOTAL_FUEL_MASS,
            Mission.Summary.OPERATING_MASS,
            Aircraft.CrewPayload.CARGO_MASS,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            Mission.Design.GROSS_MASS,
            Mission.Summary.GROSS_MASS,
            Aircraft.Design.EMPTY_MASS,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
        ]

        inputs_var_list = [
            Aircraft.CrewPayload.NUM_TOURIST_CLASS,
            Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
            Aircraft.CrewPayload.NUM_FIRST_CLASS,
            Aircraft.CrewPayload.NUM_PASSENGERS,
        ]

        for var in prob_var_list:
            assert_near_equal(comparison_prob.get_val(var), self.prob.get_val(var), tolerance=1e-7)

        for var in inputs_var_list:
            assert_near_equal(
                comparison_prob.aviary_inputs.get_val(var),
                self.prob.aviary_inputs.get_val(var),
                tolerance=1e-12,
            )

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
            problem_type='fallout',
            cargo_mass=5000,
            mission_gross_mass=150_000,
            num_first_class=1,
            num_business=5,
            num_tourist=75,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Design.RANGE),
            self.prob.get_val(Mission.Design.RANGE),
            tolerance=1e-12,
        )
        assert_near_equal(prob_fallout.get_val(Mission.Summary.RANGE), 2438.6, tolerance=1e-3)
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            46161.85,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            29031.74,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            97743.26,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            5000,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            23225,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            18225,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
            self.prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            self.prob.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            150000,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS),
            1,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS),
            5,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS),
            75,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS),
            81,
            tolerance=1e-12,
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
            problem_type='alternate',
            cargo_mass=2500,
            mission_range=1800,
            num_first_class=1,
            num_business=5,
            num_tourist=144,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Design.RANGE),
            self.prob.get_val(Mission.Design.RANGE),
            tolerance=1e-12,
        )
        assert_near_equal(prob_alternate.get_val(Mission.Summary.RANGE), 1800, tolerance=1e-7)
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            33136.79,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            23661.46,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            97743.32,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            2500,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            36250,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            33750,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
            self.prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            self.prob.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            157654.78,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS),
            1,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS),
            5,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS),
            144,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS),
            150,
            tolerance=1e-12,
        )


# @use_tempdirs
class Test2DOFOffDesign(unittest.TestCase):
    """Test off-design capability for both fallout and alternate missions."""

    def setUp(self):
        # run design case
        prob = self.prob = AviaryProblem(verbosity=1)

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv', twodof_phase_info
        )

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
        prob_var_list = [
            Mission.Summary.RANGE,
            Mission.Summary.FUEL_MASS,
            Mission.Summary.TOTAL_FUEL_MASS,
            Mission.Summary.OPERATING_MASS,
            Aircraft.CrewPayload.CARGO_MASS,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            Mission.Design.GROSS_MASS,
            Mission.Summary.GROSS_MASS,
            # currently not a GASP variable
            # Aircraft.Design.EMPTY_MASS,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
        ]

        inputs_var_list = [
            (Mission.Design.RANGE, 'nmi'),
            (Aircraft.CrewPayload.NUM_PASSENGERS, 'unitless'),
        ]

        for var in prob_var_list:
            assert_near_equal(comparison_prob.get_val(var), self.prob.get_val(var), tolerance=1e-7)

        for var in inputs_var_list:
            assert_near_equal(
                comparison_prob.aviary_inputs.get_val(var[0], var[1]),
                self.prob.aviary_inputs.get_val(var[0], var[1]),
                tolerance=1e-12,
            )

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
            problem_type='fallout',
            cargo_mass=5000,
            mission_gross_mass=160_000,
            num_pax=75,
        )
        assert_near_equal(
            prob_fallout.aviary_inputs.get_val(Mission.Design.RANGE, 'nmi'),
            self.prob.aviary_inputs.get_val(Mission.Design.RANGE, 'nmi'),
            tolerance=1e-12,
        )
        assert_near_equal(prob_fallout.get_val(Mission.Summary.RANGE), 4538.54, tolerance=1e-6)
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            40092.42,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            45044.63,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            94955.36,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            5000,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            20000,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            15000,
            tolerance=1e-7,
        )
        # currently not a GASP variable
        # assert_near_equal(
        #     prob_fallout.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     self.prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     tolerance=1e-12,
        # )
        assert_near_equal(
            prob_fallout.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            self.prob.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            160000,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS),
            75,
            tolerance=1e-12,
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
        alternate_phase_info = twodof_phase_info.copy()
        alternate_phase_info['desc1']['time_duration_bounds'] = ((200.0, 900.0), 's')

        prob_alternate = self.prob.run_off_design_mission(
            problem_type='alternate',
            cargo_mass=2100,
            mission_range=1800,
            num_pax=150,
        )
        assert_near_equal(
            prob_alternate.aviary_inputs.get_val(Mission.Design.RANGE, 'nmi'),
            self.prob.aviary_inputs.get_val(Mission.Design.RANGE, 'nmi'),
            tolerance=1e-12,
        )
        assert_near_equal(prob_alternate.get_val(Mission.Summary.RANGE), 1800, tolerance=1e-7)
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            40092.42,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            34825.03,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            94955.36,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            2100,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            38100,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            36000,
            tolerance=1e-7,
        )
        # currently not a GASP variable
        # assert_near_equal(
        #     prob_alternate.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     self.prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     tolerance=1e-12,
        # )
        assert_near_equal(
            prob_alternate.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            self.prob.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            167880.40,
            tolerance=1e-7,
        )
        assert_near_equal(
            prob_alternate.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS),
            150,
            tolerance=1e-12,
        )


@use_tempdirs
class PayloadRangeTest(unittest.TestCase):
    @require_pyoptsparse(optimizer='SNOPT')
    def test_payload_range(self):
        # run design case
        prob = self.prob = AviaryProblem(verbosity=1)
        energy_phase_info['post_mission']['target_range'] = (2500.0, 'nmi')

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
        prob.run_payload_range()
        assert_near_equal(
            prob.payload_range_data.get_val('Payload', 'lbm'),
            [38025.0, 38025.0, 23625.0, 225.0],  # due to bug ferry mission must carry 1 passenger
            tolerance=1e-12,
        )
        assert_near_equal(
            prob.payload_range_data.get_val('Fuel', 'lbm'),
            [0, 28538.30, 43524.41, 43780.55],
            tolerance=1e-7,
        )
        assert_near_equal(
            prob.payload_range_data.get_val('Range', 'NM'),
            [0, 2500, 4064.07, 4509.02],
            tolerance=1e-6,
        )


if __name__ == '__main__':
    # unittest.main()
    # test = Test2DOFOffDesign()
    # test.setUp()
    # test.test_fallout_mission_changed()

    test = PayloadRangeTest()
    test.test_payload_range()
