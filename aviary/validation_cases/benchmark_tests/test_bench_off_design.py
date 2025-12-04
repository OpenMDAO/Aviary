import unittest
from copy import deepcopy

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
        prob = self.prob = AviaryProblem(verbosity=0)
        copy_energy_phase_info = deepcopy(energy_phase_info)
        copy_energy_phase_info['post_mission']['target_range'] = (2500.0, 'nmi')

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', copy_energy_phase_info
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
            assert_near_equal(comparison_prob.get_val(var), self.prob.get_val(var), tolerance=1e-6)

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
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            29031.74,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            97743.26,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            5000,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            23225,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            18225,
            tolerance=1e-6,
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
        assert_near_equal(prob_alternate.get_val(Mission.Summary.RANGE), 1800, tolerance=1e-6)
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            33136.79,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            23661.46,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            97743.32,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            2500,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            36250,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            33750,
            tolerance=1e-6,
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
            tolerance=1e-6,
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


@use_tempdirs
class Test2DOFOffDesign(unittest.TestCase):
    """Test off-design capability for both fallout and alternate missions."""

    # TODO this test needs more manual verification to root out any remaining bugs

    def setUp(self):
        # run design case
        prob = self.prob = AviaryProblem(verbosity=0)

        copy_twodof_phase_info = deepcopy(twodof_phase_info)

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv', copy_twodof_phase_info
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
            assert_near_equal(comparison_prob.get_val(var), self.prob.get_val(var), tolerance=1e-6)

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
        prob = self.prob
        prob_fallout = prob.run_off_design_mission(
            problem_type='fallout',
            cargo_mass=5000,
            mission_gross_mass=155000.0,
            num_pax=75,
        )
        assert_near_equal(
            prob_fallout.aviary_inputs.get_val(Mission.Design.RANGE, 'nmi'),
            prob.aviary_inputs.get_val(Mission.Design.RANGE, 'nmi'),
            tolerance=1e-12,
        )
        assert_near_equal(prob_fallout.get_val(Mission.Summary.RANGE), 3988.58, tolerance=1e-4)
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            40546.40,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            39899.924,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            95100.08,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            5000,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            20000,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_fallout.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            15000,
            tolerance=1e-6,
        )
        # currently not a GASP variable
        # assert_near_equal(
        #     prob_fallout.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     tolerance=1e-12,
        # )
        assert_near_equal(
            prob_fallout.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            prob.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            155000,
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
        prob = self.prob
        alternate_phase_info = deepcopy(twodof_phase_info)
        alternate_phase_info['desc1']['time_duration_bounds'] = ((200.0, 900.0), 's')

        prob_alternate = prob.run_off_design_mission(
            problem_type='alternate',
            cargo_mass=2100,
            mission_range=1800,
            num_pax=150,
        )
        assert_near_equal(
            prob_alternate.aviary_inputs.get_val(Mission.Design.RANGE, 'nmi'),
            prob.aviary_inputs.get_val(Mission.Design.RANGE, 'nmi'),
            tolerance=1e-12,
        )
        assert_near_equal(prob_alternate.get_val(Mission.Summary.RANGE), 1800, tolerance=1e-6)
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.FUEL_MASS, 'lbm'),
            40546.40,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.TOTAL_FUEL_MASS, 'lbm'),
            21499.71,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.OPERATING_MASS, 'lbm'),
            95100.08,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            2100,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            32100,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_alternate.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            30000,
            tolerance=1e-6,
        )
        # currently not a GASP variable
        # assert_near_equal(
        #     prob_alternate.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     tolerance=1e-12,
        # )
        assert_near_equal(
            prob_alternate.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            prob.get_val(Mission.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            148699.79,
            tolerance=1e-6,
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
        prob = self.prob = AviaryProblem(verbosity=0)
        phase_info = deepcopy(energy_phase_info)

        phase_info['post_mission']['target_range'] = (2500.0, 'nmi')
        phase_info['climb']['user_options']['time_duration_bounds'] = ((20.0, 90.0), 'min')
        phase_info['cruise']['user_options']['time_initial_bounds'] = ((20.0, 192.0), 'min')
        phase_info['descent']['user_options']['time_duration_bounds'] = (
            (25.0, 60.0),
            'min',
        )
        prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)
        # prob.aviary_inputs.set_val(Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT, True)

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
        off_design_probs = prob.run_payload_range()
        # test outputted payload-range data
        assert_near_equal(
            prob.payload_range_data.get_val('Payload', 'lbm'),
            [
                38025.0,
                38025.0,
                24365.60919974074,
                225.0,
            ],  # due to bug ferry mission must carry 1 passenger
            tolerance=1e-10,
        )
        assert_near_equal(
            prob.payload_range_data.get_val('Fuel', 'lbm'),
            [0, 28108.99, 42192.70, 42192.70],
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.payload_range_data.get_val('Range', 'NM'),
            [0, 2500, 3973.34, 4415.77],
            tolerance=1e-6,
        )

        # verify TOGW for each off-design problem
        assert_near_equal(
            off_design_probs[0].get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            165896.26754754,
            tolerance=1e-12,
        )
        assert_near_equal(
            off_design_probs[1].get_val(Mission.Summary.GROSS_MASS, 'lbm'),
            140868.42657438,
            tolerance=1e-12,
        )


if __name__ == '__main__':
    # unittest.main()
    test = Test2DOFOffDesign()
    test.setUp()
    test.test_alternate_mission_changed()

    # test = PayloadRangeTest()
    # test.test_payload_range()
