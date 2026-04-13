import unittest
import aviary.api as av
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.missions.energy_state_default import phase_info as energy_phase_info
from aviary.models.missions.two_dof_default import phase_info as twodof_phase_info
from aviary.variable_info.variables import Aircraft, Mission, Settings


@use_tempdirs
class TestEnergyStateOffDesign(unittest.TestCase):
    """Test off-design capability for both OFF_DESIGN_MAX_RANGE and OFF_DESIGN_MIN_FUEL missions."""

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
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_ECONOMY_CLASS, 144)
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 15)
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 10)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()
        prob.add_driver('SNOPT', max_iter=50)
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
            Aircraft.Design.RANGE,
            Mission.RANGE,
            Mission.TOTAL_FUEL,
            Mission.OPERATING_MASS,
            Aircraft.CrewPayload.CARGO_MASS,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            Aircraft.Design.GROSS_MASS,
            Mission.GROSS_MASS,
            Aircraft.Design.EMPTY_MASS,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
        ]

        inputs_var_list = [
            Aircraft.CrewPayload.NUM_ECONOMY_CLASS,
            Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
            Aircraft.CrewPayload.NUM_FIRST_CLASS,
            Aircraft.CrewPayload.NUM_PASSENGERS,
        ]

        for var in prob_var_list:
            with self.subTest(var=var):
                assert_near_equal(
                    comparison_prob.get_val(var), self.prob.get_val(var), tolerance=1e-6
                )

        for var in inputs_var_list:
            with self.subTest(var=var):
                assert_near_equal(
                    comparison_prob.aviary_inputs.get_val(var),
                    self.prob.aviary_inputs.get_val(var),
                    tolerance=1e-12,
                )

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_max_range_mission_match(self):
        # run a off_design_max_range mission with no changes, essentially recreating the design mission with
        # different constraints/design variables
        prob_off_design_max_range = self.prob.run_off_design_mission(
            problem_type='off_design_max_range'
        )
        self.compare_results(prob_off_design_max_range)
        self.assertTrue(prob_off_design_max_range.result.success)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_max_range_mission_changed(self):
        # run a off_design_max_range mission with modified payload and gross mass (and therefore different fuel)
        prob_off_design_max_range = self.prob.run_off_design_mission(
            problem_type='off_design_max_range',
            cargo_mass=5000,
            mission_gross_mass=150_000,
            num_first_class=1,
            num_business=5,
            num_economy=75,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.Design.RANGE),
            self.prob.get_val(Aircraft.Design.RANGE),
            tolerance=1e-12,
        )
        assert_near_equal(prob_fallout.get_val(Mission.RANGE), 2377.4, tolerance=1e-3)
        assert_near_equal(
            prob_fallout.get_val(Mission.TOTAL_FUEL, 'lbm'),
            28976.71270599,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_fallout.get_val(Mission.OPERATING_MASS, 'lbm'),
            97798.28729401,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            5000,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            23225,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            18225,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
            self.prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.Design.GROSS_MASS, 'lbm'),
            self.prob.get_val(Aircraft.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Mission.GROSS_MASS, 'lbm'),
            150000,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS),
            1,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.aviary_inputs.get_val(
                Aircraft.CrewPayload.NUM_BUSINESS_CLASS
            ),
            5,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_ECONOMY_CLASS),
            75,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS),
            81,
            tolerance=1e-12,
        )
        self.assertTrue(prob_off_design_max_range.result.success)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_min_fuel_mission_match(self):
        # run an off_design_min_fuel mission with no changes, essentially recreating the design mission with
        # different constraints/design variables
        prob_off_design_min_fuel = self.prob.run_off_design_mission(
            problem_type='off_design_min_fuel'
        )
        self.compare_results(prob_off_design_min_fuel)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_min_fuel_mission_changed(self):
        # run an off_design_min_fuel mission with modified range and payload
        prob_off_design_min_fuel = self.prob.run_off_design_mission(
            problem_type='off_design_min_fuel',
            cargo_mass=2500,
            mission_range=1800,
            num_first_class=1,
            num_business=5,
            num_economy=144,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.Design.RANGE),
            self.prob.get_val(Aircraft.Design.RANGE),
            tolerance=1e-12,
        )
        assert_near_equal(prob_off_design_min_fuel.get_val(Mission.RANGE), 1800, tolerance=1e-6)
        assert_near_equal(
            prob_alternate.get_val(Mission.TOTAL_FUEL, 'lbm'),
            24245.7724282,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.OPERATING_MASS, 'lbm'),
            97798.34840008,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            2500,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            36250,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            33750,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
            self.prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.Design.GROSS_MASS, 'lbm'),
            self.prob.get_val(Aircraft.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_alternate.get_val(Mission.GROSS_MASS, 'lbm'),
            158294.12082828,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_min_fuel.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS),
            1,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_min_fuel.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS),
            5,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_min_fuel.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_ECONOMY_CLASS),
            144,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_min_fuel.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS),
            150,
            tolerance=1e-12,
        )
        self.assertTrue(prob_off_design_min_fuel.result.success)


@use_tempdirs
class Test2DOFOffDesign(unittest.TestCase):
    """Test off-design capability for both off_design_max_range and off_design_min_fuel missions."""

    # TODO this test needs more manual verification to root out any remaining bugs

    def setUp(self):
        # run design case
        prob = self.prob = AviaryProblem(verbosity=0)

        copy_twodof_phase_info = deepcopy(twodof_phase_info)

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv', copy_twodof_phase_info
        )

        prob.aviary_inputs.set_val(Aircraft.Design.GROSS_MASS, val=150000, units='lbm')

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
            Mission.RANGE,
            Mission.TOTAL_FUEL,
            Mission.OPERATING_MASS,
            Aircraft.CrewPayload.CARGO_MASS,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            Aircraft.Design.GROSS_MASS,
            Mission.GROSS_MASS,
            # currently not a GASP variable
            # Aircraft.Design.EMPTY_MASS,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
        ]

        inputs_var_list = [
            (Aircraft.Design.RANGE, 'nmi'),
            (Aircraft.CrewPayload.NUM_PASSENGERS, 'unitless'),
        ]

        for var in prob_var_list:
            with self.subTest(var=var):
                assert_near_equal(
                    comparison_prob.get_val(var), self.prob.get_val(var), tolerance=1e-5
                )

        for var in inputs_var_list:
            with self.subTest(var=var[0]):
                assert_near_equal(
                    comparison_prob.aviary_inputs.get_val(var[0], var[1]),
                    self.prob.aviary_inputs.get_val(var[0], var[1]),
                    tolerance=1e-12,
                )

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_max_range_mission_match(self):
        # run a off_design_max_range mission with no changes, essentially recreating the design mission with
        # different constraints/design variables
        prob_off_design_max_range = self.prob.run_off_design_mission(
            problem_type='off_design_max_range'
        )
        self.compare_results(prob_off_design_max_range)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_max_range_mission_changed(self):
        # run a off_design_max_range mission with modified payload and gross mass (and therefore different fuel)
        prob = self.prob
        prob_off_design_max_range = prob.run_off_design_mission(
            problem_type='off_design_max_range',
            cargo_mass=5000,
            mission_gross_mass=155000.0,
            num_pax=75,
        )
        assert_near_equal(
            prob_off_design_max_range.aviary_inputs.get_val(Aircraft.Design.RANGE, 'nmi'),
            prob.aviary_inputs.get_val(Aircraft.Design.RANGE, 'nmi'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Mission.RANGE), 3994.25223046, tolerance=1e-4
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Mission.TOTAL_FUEL, 'lbm'),
            39909.74193096,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Mission.OPERATING_MASS, 'lbm'),
            95090.25806904,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            5000,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            20000,
            tolerance=1e-5,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            15000,
            tolerance=1e-6,
        )
        # currently not a GASP variable
        # assert_near_equal(
        #     prob_off_design_max_range.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     tolerance=1e-12,
        # )
        assert_near_equal(
            prob_off_design_max_range.get_val(Aircraft.Design.GROSS_MASS, 'lbm'),
            prob.get_val(Aircraft.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.get_val(Mission.GROSS_MASS, 'lbm'),
            155000,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_max_range.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS),
            75,
            tolerance=1e-12,
        )
        self.assertTrue(prob_off_design_max_range.result.success)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_min_fuel_mission_match(self):
        # run an off_design_min_fuel mission with no changes, essentially recreating the design mission with
        # different constraints/design variables
        prob_off_design_min_fuel = self.prob.run_off_design_mission(
            problem_type='off_design_min_fuel'
        )
        self.compare_results(prob_off_design_min_fuel)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_min_fuel_mission_changed(self):
        # run an off_design_min_fuel mission with modified range and payload
        prob = self.prob
        off_design_min_fuel_phase_info = deepcopy(twodof_phase_info)
        off_design_min_fuel_phase_info['desc1']['time_duration_bounds'] = ((200.0, 900.0), 's')

        prob_off_design_min_fuel = prob.run_off_design_mission(
            problem_type='off_design_min_fuel',
            cargo_mass=2100,
            mission_range=1800,
            num_pax=150,
        )
        assert_near_equal(
            prob_off_design_min_fuel.aviary_inputs.get_val(Aircraft.Design.RANGE, 'nmi'),
            prob.aviary_inputs.get_val(Aircraft.Design.RANGE, 'nmi'),
            tolerance=1e-12,
        )
        assert_near_equal(prob_off_design_min_fuel.get_val(Mission.RANGE), 1800, tolerance=1e-6)
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Mission.TOTAL_FUEL, 'lbm'),
            21484.97566914,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Mission.OPERATING_MASS, 'lbm'),
            95090.25806904,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm'),
            2100,
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 'lbm'),
            32100,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 'lbm'),
            30000,
            tolerance=1e-6,
        )
        # currently not a GASP variable
        # assert_near_equal(
        #     prob_off_design_min_fuel.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     prob.get_val(Aircraft.Design.EMPTY_MASS, 'lbm'),
        #     tolerance=1e-12,
        # )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Aircraft.Design.GROSS_MASS, 'lbm'),
            prob.get_val(Aircraft.Design.GROSS_MASS, 'lbm'),
            tolerance=1e-12,
        )
        assert_near_equal(
            prob_off_design_min_fuel.get_val(Mission.GROSS_MASS, 'lbm'),
            148675.23373818,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob_off_design_min_fuel.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS),
            150,
            tolerance=1e-12,
        )
        self.assertTrue(prob_off_design_min_fuel.result.success)


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

        (aviary_inputs, initialization_guesses) = av.create_vehicle(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv'
        )
        aviary_inputs.set_val(Settings.PAYLOAD_RANGE, True)
        prob.load_inputs(aviary_inputs, phase_info)

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

        # test outputted payload-range data
        assert_near_equal(
            prob.payload_range_data.get_val('Payload', 'lbm'),
            [
                38025.0,
                38025.0,
                24953.7,
                0,
            ],
            tolerance=1e-3,
        )
        assert_near_equal(
            prob.payload_range_data.get_val('Fuel', 'lbm'),
            [0, 28697.02, 42192.69, 42192.69],
            tolerance=1e-3,
        )
        assert_near_equal(
            prob.payload_range_data.get_val('Range', 'NM'),
            [0, 2500, 3910.17, 4362.62],
            tolerance=1e-3,
        )

        # verify TOGW for each payload range problem
        assert_near_equal(
            prob.economic_range_prob.get_val(Mission.GROSS_MASS, 'lbm'),
            166539.46027154,
            tolerance=1e-8,
        )
        assert_near_equal(
            prob.ferry_range_prob.get_val(Mission.GROSS_MASS, 'lbm'),
            140596.07154268,
            tolerance=1e-8,
        )
        self.assertTrue(prob.result.success)
        self.assertTrue(prob.economic_range_prob.result.success)
        self.assertTrue(prob.ferry_range_prob.result.success)


if __name__ == '__main__':
    # unittest.main()
    test = Test2DOFOffDesign()
    # test = TestEnergyStateOffDesign()
    test.setUp()
    test.test_off_design_min_fuel_mission_match()

    # test = PayloadRangeTest()
    # test.test_payload_range()
