import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import aviary.api as av
from aviary.subsystems.energy.battery_builder import BatteryBuilder


@use_tempdirs
class TestBatteryMission(unittest.TestCase):
    """Test the setup and run optimization model with a battery subsystem."""

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
        phase_info = self.phase_info.copy()

        prob = av.AviaryProblem(verbosity=0)

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm_with_electric.csv',
            phase_info,
        )
        prob.load_external_subsystems([BatteryBuilder()])

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

        electric_energy_used_cruise2 = prob.get_val(
            f'traj.cruise2.timeseries.{av.Dynamic.Vehicle.CUMULATIVE_ELECTRIC_ENERGY_USED}',
            units='kW*h',
        )
        soc_cruise1 = prob.get_val(
            f'traj.cruise1.timeseries.{av.Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE}',
        )
        soc_cruise2 = prob.get_val(
            f'traj.cruise2.timeseries.{av.Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE}',
        )
        fuel_burned = prob.get_val(av.Mission.Summary.FUEL_BURNED, units='lbm')

        # Check outputs
        # indirectly check mission trajectory by checking total fuel/electric split
        assert_near_equal(electric_energy_used_cruise2[-1], 38.60747069, 1.0e-7)
        assert_near_equal(fuel_burned, 676.93670291, 1.0e-7)
        # check battery state-of-charge over mission

        assert_near_equal(
            soc_cruise1.ravel(),
            [
                0.99999578,
                0.98770725,
                0.97075246,
                0.96538654,
                0.96538654,
                0.94029352,
                0.90567396,
                0.89471788,
                0.89471788,
                0.86489252,
                0.82374482,
                0.81072297,
                0.81072297,
                0.78564377,
                0.75104327,
                0.74009322,
                0.74009322,
                0.72781605,
                0.71087695,
                0.70551599,
            ],
            1e-7,
        )

        assert_near_equal(
            soc_cruise2.ravel(),
            [
                0.70551599,
                0.69324034,
                0.67630332,
                0.67094303,
                0.67094303,
                0.64587631,
                0.61129303,
                0.60034842,
                0.60034842,
                0.57055432,
                0.52944973,
                0.51644152,
                0.51644152,
                0.49138859,
                0.45682433,
                0.44588575,
                0.44588575,
                0.43362144,
                0.41670008,
                0.41134474,
            ],
            1e-7,
        )


if __name__ == '__main__':
    unittest.main()
    # test = TestSubsystemsMission()
    # test.setUp()
    # test.test_subsystems_in_a_mission()
