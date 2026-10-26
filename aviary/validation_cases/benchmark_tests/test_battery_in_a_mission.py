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
            cumulative_energy_var: (38.61381379, 'kW*h'),
            av.Mission.FUEL_MASS: (1254.14061154, 'lbm'),
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
                    1.0,
                    0.9877557,
                    0.97086194,
                    0.96551534,
                    0.96551534,
                    0.94051264,
                    0.90601767,
                    0.89510102,
                    0.89510102,
                    0.86538299,
                    0.82438334,
                    0.81140834,
                    0.81140834,
                    0.78641936,
                    0.75194333,
                    0.74103266,
                    0.74103266,
                    0.72879965,
                    0.71192148,
                    0.7065798,
                ],
                None,
            ),
            soc_cruise2_var: (
                [
                    0.7065798,
                    0.6943483,
                    0.67747219,
                    0.67213117,
                    0.67213117,
                    0.64715459,
                    0.61269566,
                    0.60179041,
                    0.60179041,
                    0.57210341,
                    0.53114658,
                    0.51818513,
                    0.51818513,
                    0.49322225,
                    0.45878221,
                    0.44788293,
                    0.44788293,
                    0.4356627,
                    0.41880214,
                    0.41346604,
                ],
                None,
            ),
        }

        for var_name, (expected, units) in expected_soc_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name).ravel()
                assert_near_equal(actual, expected, 1e-6)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_subsystems_in_a_mission_2dof(self):
        """
        Does not actually use electric power but tests that externally added states are successfully
        added to all mission phases.
        """
        phase_info = deepcopy(twodof_phase_info)

        prob = av.AviaryProblem(verbosity=1)

        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_GwGm.csv',
            phase_info,
        )
        prob.load_external_subsystems([BatteryBuilder()])

        prob.aviary_inputs.set_val(av.Aircraft.Battery.EFFICIENCY, 0.95, 'unitless')

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver('IPOPT')

        prob.add_design_variables()

        prob.add_objective('fuel_burned', ref=1e3)

        prob.setup()

        prob.set_val(av.Aircraft.Battery.PACK_ENERGY_DENSITY, 550, units='kJ/kg')
        prob.set_val(av.Aircraft.Battery.PACK_MASS, 1000, units='lbm')
        prob.set_val(av.Aircraft.Battery.ADDITIONAL_MASS, 115, units='lbm')

        prob.run_aviary_problem()
        self.assertTrue(prob.result.success)


if __name__ == '__main__':
    unittest.main()
    # test = TestBatteryMission()
    # test.setUp()
    # test.test_subsystems_in_a_mission()
