import unittest

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

import aviary.api as av
from aviary.subsystems.energy.battery_builder import BatteryBuilder


@use_tempdirs
class TestSubsystemsMission(unittest.TestCase):
    """
    Test the setup and run optimization model with a bettery subsystem.
    """

    def setUp(self):
        self.phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [BatteryBuilder()],
                'optimize_mass': True,
            },
            'cruise': {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                'external_subsystems': [BatteryBuilder()],
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.72, "unitless"),
                    "final_mach": (0.72, "unitless"),
                    "mach_bounds": ((0.7, 0.74), "unitless"),
                    "initial_altitude": (35000.0, "ft"),
                    "final_altitude": (35000.0, "ft"),
                    "altitude_bounds": ((23000.0, 38000.0), "ft"),
                    "throttle_enforcement": "boundary_constraint",
                    "fix_initial": True,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((0.0, 0.0), "min"),
                    "duration_bounds": ((10.0, 30.0), "min"),
                },
            },
            'post_mission': {
                'include_landing': False,
                'external_subsystems': [],
            },
        }

    def test_subsystems_in_a_mission(self):
        phase_info = self.phase_info.copy()

        prob = av.AviaryProblem()

        prob.load_inputs(
            "models/test_aircraft/aircraft_for_bench_FwFm_with_electric.csv", phase_info
        )

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver('SLSQP')

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.set_initial_guesses()

        prob.set_val(av.Aircraft.Battery.PACK_ENERGY_DENSITY, 550, units='kJ/kg')
        prob.set_val(av.Aircraft.Battery.PACK_MASS, 1000, units='lbm')
        prob.set_val(av.Aircraft.Battery.ADDITIONAL_MASS, 115, units='lbm')
        prob.set_val(av.Aircraft.Battery.EFFICIENCY, 0.95, units='unitless')

        prob.run_aviary_problem()

        electric_energy_used = prob.get_val(
            'traj.cruise.timeseries.'
            f'{av.Dynamic.Mission.CUMULATIVE_ELECTRIC_ENERGY_USED}',
            units='kW*h',
        )
        fuel_burned = prob.get_val(av.Mission.Summary.FUEL_BURNED, units='lbm')
        soc = prob.get_val(
            'traj.cruise.rhs_all.battery.battery_state_of_charge', units='unitless'
        )

        # Check outputs
        # indirectly check mission trajectory by checking total fuel/electric split
        assert_near_equal(electric_energy_used[-1], 38.60538132, 1.e-7)
        assert_near_equal(fuel_burned, 676.87235486, 1.e-7)
        # check battery state-of-charge over mission
        assert_near_equal(
            soc,
            [0.99999578, 0.97551324, 0.94173584, 0.93104625, 0.93104625,
             0.8810605, 0.81210498, 0.79028433, 0.79028433, 0.73088701,
             0.64895148, 0.62302415, 0.62302415, 0.57309323, 0.50421334,
             0.48241661, 0.48241661, 0.45797918, 0.42426402, 0.41359413],
            1e-7,
        )


if __name__ == "__main__":
    unittest.main()
