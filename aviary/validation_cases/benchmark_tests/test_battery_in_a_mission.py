import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

import aviary.api as av
from aviary.models.missions.two_dof_default import phase_info as twodof_phase_info
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
        phase_info = deepcopy(self.phase_info)

        prob = av.AviaryProblem(verbosity=0)

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm_with_electric.csv',
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
        fuel_burned = prob.get_val(av.Mission.FUEL, units='lbm')

        # Check outputs
        # indirectly check mission trajectory by checking total fuel/electric split
        assert_near_equal(electric_energy_used_cruise2[-1], 38.61409623, 1.0e-7)
        assert_near_equal(fuel_burned, 1254.14075517, 1.0e-7)
        # check battery state-of-charge over mission

        assert_near_equal(
            soc_cruise1.ravel(),
            [
                0.99999578,
                0.98775148,
                0.97085772,
                0.96551112,
                0.96551112,
                0.94050841,
                0.90601344,
                0.89509679,
                0.89509679,
                0.86537875,
                0.82437910,
                0.81140410,
                0.81140410,
                0.78641512,
                0.75193908,
                0.74102841,
                0.74102841,
                0.72879540,
                0.71191722,
                0.70657555,
            ],
            1e-6,
        )

        assert_near_equal(
            soc_cruise2.ravel(),
            [
                0.70657555,
                0.69434404,
                0.67746793,
                0.67212691,
                0.67212691,
                0.64715033,
                0.61269139,
                0.60178614,
                0.60178614,
                0.57209914,
                0.53114231,
                0.51818085,
                0.51818085,
                0.49321797,
                0.45877792,
                0.44787865,
                0.44787865,
                0.43565841,
                0.41879785,
                0.41346175,
            ],
            1e-6,
        )

    @require_pyoptsparse(optimizer='SNOPT')
    def test_subsystems_in_a_mission_2dof(self):
        phase_info = deepcopy(twodof_phase_info)

        prob = av.AviaryProblem(verbosity=0)

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
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
    test.test_subsystems_in_a_mission_2dof()
