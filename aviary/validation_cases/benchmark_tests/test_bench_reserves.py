import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.interface.run_aviary import run_aviary
from aviary.models.missions.energy_state_default import phase_info as energy_phase_info
from aviary.models.missions.two_dof_default import phase_info as twodof_phase_info
from aviary.variable_info.enums import PhaseType
from aviary.variable_info.variables import Aircraft, Mission, Settings


# @use_tempdirs
class ReserveTest(unittest.TestCase):
    def test_reserves_2dof(self):
        phase_info_local = deepcopy(twodof_phase_info)

        phase_info_local.update(
            {
                'reserve_cruise': {
                    'subsystem_options': {'aerodynamics': {'method': 'cruise'}},
                    'user_options': {
                        'reserve': True,
                        'phase_type': PhaseType.SIMPLE_CRUISE,
                        'target_distance': (300, 'km'),
                        'alt_cruise': (20_000, 'ft'),
                        'mach_cruise': 0.5,
                        'mass_ref': (100_000, 'lbm'),
                        'time_duration_bounds': ((0, 300), 'min'),
                    },
                    'initial_guesses': {
                        'mass': ([168500.0, 135000], 'lbm'),
                        'time': ([1504.0, 18000.0], 's'),
                    },
                },
            }
        )

        prob = AviaryProblem(verbosity=0)
        prob.load_inputs(
            'large_single_aisle_1_GASP.csv',
            phase_info_local,
        )

        prob.aviary_inputs.set_val(Mission.RESERVE_FUEL_MARGIN, 5)
        prob.aviary_inputs.set_val(Mission.RESERVE_FUEL_MASS_ADDITIONAL, 125, units='lbm')

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver()

        prob.add_design_variables()

        prob.add_objective()

        prob.setup()

        prob.run_aviary_problem()

        expected_values = {
            'energy.reserve_fuel_margin_mass': (1887.21889734, 'lbm'),
            Mission.RESERVE_FUEL_MASS: (37744.37794685, 'lbm'),
            Mission.TOTAL_RESERVE_FUEL_MASS: (39756.5968442, 'lbm'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob.get_val(var_name, units=units), expected, 1e-10)

    def test_reserves_energy(self):
        phase_info_local = deepcopy(energy_phase_info)
        phase_info_local.update(
            {
                'reserve_cruise': {
                    'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                    'user_options': {
                        'reserve': True,
                        # Distance traveled in this phase
                        'target_distance': (300, 'km'),
                        'num_segments': 5,
                        'order': 3,
                        'mach_optimize': False,
                        'mach_polynomial_order': 1,
                        'mach_initial': (0.5, 'unitless'),
                        'mach_final': (0.5, 'unitless'),
                        'altitude_optimize': False,
                        'altitude_polynomial_order': 1,
                        'altitude_initial': (20_000, 'ft'),
                        'altitude_final': (20_000, 'ft'),
                        'throttle_enforcement': 'boundary_constraint',
                        'time_initial_bounds': ((149.5, 448.5), 'min'),
                        'time_duration_bounds': ((0, 300), 'min'),
                    },
                    'initial_guesses': {
                        'time': ([30, 120], 'min'),
                    },
                }
            }
        )

        prob = AviaryProblem(verbosity=0)
        prob.load_inputs(
            'advanced_single_aisle_FLOPS.csv',
            phase_info_local,
        )

        prob.aviary_inputs.set_val(Mission.RESERVE_FUEL_MARGIN, 5)
        prob.aviary_inputs.set_val(Mission.RESERVE_FUEL_MASS_ADDITIONAL, 125, units='lbm')

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver()

        prob.add_design_variables()

        prob.add_objective()

        prob.setup()

        prob.run_aviary_problem()

        expected_values = {
            'energy.reserve_fuel_margin_mass': (790.46785778, 'lbm'),
            Mission.RESERVE_FUEL_MASS: (15809.3571555, 'lbm'),
            Mission.TOTAL_RESERVE_FUEL_MASS: (16724.82501328, 'lbm'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob.get_val(var_name, units=units), expected, 1e-10)


if __name__ == '__main__':
    unittest.main()
    # test = ReserveTest()
    # test.test_reserves_energy()
    # test.test_reserves_2dof()
