import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import aviary.api as av
from aviary.subsystems.energy.battery_builder import BatteryBuilder


@use_tempdirs
class TestSubsystemsMission(unittest.TestCase):
    """Test the setup and run optimization model with a bettery subsystem."""

    def setUp(self):
        self.phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [BatteryBuilder()],
                'optimize_mass': True,
            },
            'cruise1': {
                'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
                'external_subsystems': [BatteryBuilder()],
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
                'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
                'external_subsystems': [BatteryBuilder()],
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
                'external_subsystems': [],
            },
        }

    def test_subsystems_in_a_mission(self):
        phase_info = self.phase_info.copy()

        prob = av.AviaryProblem(verbosity=0)

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm_with_electric.csv',
            phase_info,
        )

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
                0.9999957806265609,
                0.9877535827147117,
                0.9708627248497483,
                0.9655170378370462,
                0.9655170378370462,
                0.9405186291064791,
                0.9060295822713065,
                0.8951147997010394,
                0.8951147997010394,
                0.8654018683927851,
                0.8244092512817177,
                0.8114364753183335,
                0.8114364753183335,
                0.7864517853197655,
                0.7519816637312142,
                0.7410728700181042,
                0.7410728700181042,
                0.7288419600341913,
                0.7119666756734181,
                0.7066259172513626,
            ],
            1e-7,
        )

        assert_near_equal(
            soc_cruise2.ravel(),
            [
                0.7066259172513626,
                0.6943965083029002,
                0.677523294860927,
                0.672183191828372,
                0.672183191828372,
                0.6472108945884663,
                0.6127578690420414,
                0.601854485338024,
                0.601854485338024,
                0.5721725827519708,
                0.531222768718283,
                0.5182635373126063,
                0.5182635373126063,
                0.4933049316352059,
                0.4588707938555728,
                0.44797338714853385,
                0.44797338714853385,
                0.435755243772871,
                0.41889757299299735,
                0.4135623887905029,
            ],
            1e-7,
        )


if __name__ == '__main__':
    unittest.main()
    # test = TestSubsystemsMission()
    # test.setUp()
    # test.test_subsystems_in_a_mission()
