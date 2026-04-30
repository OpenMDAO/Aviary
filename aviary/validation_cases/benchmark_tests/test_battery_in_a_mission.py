import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

import aviary.api as av
from aviary.models.missions.two_dof_default import phase_info as twodof_phase_info
from aviary.subsystems.energy.battery_builder import BatteryBuilder


@use_tempdirs
class TestBatteryMission(unittest.TestCase):
    """Test the setup and optimization of a model with a battery subsystem."""

    def setUp(self):
        self.phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'optimize_mass': True,
            },
            'cruise1': {
                'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'num_segments': 5,
                    'order': 3,
                    'mach_optimize': False,
                    'mach_polynomial_order': 1,
                    'mach_initial': (0.72, 'unitless'),
                    'mach_bounds': ((0.7, 0.74), 'unitless'),
                    'altitude_optimize': False,
                    'altitude_polynomial_order': 1,
                    'altitude_initial': (35000.0, 'ft'),
                    'altitude_final': (35000.0, 'ft'),
                    'altitude_bounds': ((23000.0, 38000.0), 'ft'),
                    'throttle_enforcement': 'boundary_constraint',
                    'time_initial': (0.0, 'min'),
                    'time_duration_bounds': ((5.0, 30.0), 'min'),
                },
            },
            'cruise2': {
                'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'num_segments': 5,
                    'order': 3,
                    'mach_optimize': False,
                    'mach_polynomial_order': 1,
                    'mach_initial': (0.72, 'unitless'),
                    'mach_final': (0.72, 'unitless'),
                    'mach_bounds': ((0.7, 0.74), 'unitless'),
                    'altitude_optimize': False,
                    'altitude_polynomial_order': 1,
                    'altitude_initial': (35000.0, 'ft'),
                    'altitude_final': (35000.0, 'ft'),
                    'altitude_bounds': ((23000.0, 38000.0), 'ft'),
                    'throttle_enforcement': 'boundary_constraint',
                    'time_initial': (0.0, 'min'),
                    'time_duration_bounds': ((5.0, 30.0), 'min'),
                },
            },
            'post_mission': {
                'include_landing': False,
            },
        }

    def test_subsystems_in_a_mission(self):
        phase_info = deepcopy(self.phase_info)

        prob = av.AviaryProblem(verbosity=0)

        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_FwFm_with_electric.csv',
            phase_info,
        )
        prob.load_external_subsystems([BatteryBuilder()])

        prob.aviary_inputs.set_val(av.Aircraft.Battery.EFFICIENCY, 0.95, 'unitless')

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver('SLSQP')

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.set_val(av.Aircraft.Battery.PACK_ENERGY_DENSITY, 550, units='kJ/kg')
        prob.set_val(av.Aircraft.Battery.PACK_MASS, 1000, units='lbm')
        prob.set_val(av.Aircraft.Battery.ADDITIONAL_MASS, 115, units='lbm')

        prob.run_aviary_problem()
        self.assertTrue(prob.result.success)

        # Check fuel/electric energy split
        cumulative_energy_var = (
            f'traj.cruise2.timeseries.{av.Dynamic.Vehicle.CUMULATIVE_ELECTRIC_ENERGY_USED}'
        )
        expected_scalar_values = {
            cumulative_energy_var: (38.46817379, 'kW*h'),
            av.Mission.FUEL: (1249.64666191, 'lbm'),
        }

        for var_name, (expected, units) in expected_scalar_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                if actual.size > 1:
                    actual = actual[-1]
                assert_near_equal(actual, expected, 1.0e-7)

        # Check battery state-of-charge over mission
        soc_cruise1_var = f'traj.cruise1.timeseries.{av.Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE}'
        soc_cruise2_var = f'traj.cruise2.timeseries.{av.Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE}'

        expected_soc_values = {
            soc_cruise1_var: (
                [
                    0.9999957806265609,
                    0.9877977965127587,
                    0.9709679386644526,
                    0.9656415564989798,
                    0.9656415564989798,
                    0.9407334200010511,
                    0.906368905403873,
                    0.8954935308182879,
                    0.8954935308182879,
                    0.8658878715348804,
                    0.8250432325811342,
                    0.8121172826623315,
                    0.8121172826623315,
                    0.7872227709968601,
                    0.752877052113554,
                    0.7420076253955674,
                    0.7420076253955674,
                    0.729820851948888,
                    0.7130064610124855,
                    0.7076849737044318,
                ],
                None,
            ),
            soc_cruise2_var: (
                [
                    0.7076849737044318,
                    0.6954996910206249,
                    0.6786873568305393,
                    0.6733665204268243,
                    0.6733665204268243,
                    0.6484843167073028,
                    0.6141555768684874,
                    0.6032915231366659,
                    0.6032915231366659,
                    0.5737166802223987,
                    0.532914551421322,
                    0.5200020533696793,
                    0.5200020533696793,
                    0.49513344752347366,
                    0.460823466208866,
                    0.44996534857628184,
                    0.44996534857628184,
                    0.4377912370406455,
                    0.4209942619176692,
                    0.41567827333310314,
                ],
                None,
            ),
        }

        for var_name, (expected, units) in expected_soc_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name).ravel()
                assert_near_equal(actual, expected, 1e-6)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_subsystems_in_a_mission_2dof(self):
        """
        Does not actually use electric power but tests that externally added states are successfully
        added to all mission phases.
        """
        phase_info = deepcopy(twodof_phase_info)

        prob = av.AviaryProblem(verbosity=0)

        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_GwGm.csv',
            phase_info,
        )
        prob.load_external_subsystems([BatteryBuilder()])

        prob.aviary_inputs.set_val(av.Aircraft.Battery.EFFICIENCY, 0.95, 'unitless')

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver('SNOPT')

        prob.add_design_variables()

        prob.add_objective('fuel_burned', ref=1e3)

        prob.setup()

        prob.set_val(av.Aircraft.Battery.PACK_ENERGY_DENSITY, 550, units='kJ/kg')
        prob.set_val(av.Aircraft.Battery.PACK_MASS, 1000, units='lbm')
        prob.set_val(av.Aircraft.Battery.ADDITIONAL_MASS, 115, units='lbm')

        prob.run_aviary_problem()
        self.assertTrue(prob.result.success)


if __name__ == '__main__':
    # unittest.main()
    test = TestBatteryMission()
    test.setUp()
    test.test_subsystems_in_a_mission()
