from copy import deepcopy
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

import aviary.api as av
from aviary.models.missions.energy_state_default import phase_info
from aviary.subsystems.energy.fuel_summation import FuelSummationGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission


class TestFuelSummation(unittest.TestCase):
    def test_fuel_group(self):
        local_phase_info = deepcopy(phase_info)

        local_phase_info['descent']['user_options']['reserve'] = True
        prob = om.Problem()
        prob.model.add_subsystem(
            'fuel_group', FuelSummationGroup(mission_info=local_phase_info), promotes=['*']
        )

        setup_model_options(
            prob,
            AviaryValues(
                {
                    Mission.RESERVE_FUEL_MARGIN: (0.2, 'unitless'),
                    Mission.RESERVE_FUEL_MASS_ADDITIONAL: (300, 'lbm'),
                    Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT: (False, 'unitless'),
                }
            ),
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(Mission.GROSS_MASS, 140_000, units='lbm')
        prob.set_val('fuel_burned.mass_final', 120_000, units='lbm')
        prob.set_val('reserve_fuel_burned.mass_initial', 120_000, 'lbm')
        prob.set_val('reserve_fuel_burned.mass_final', 110_000, 'lbm')
        prob.set_val('reserve_fuel_frac.final_mass', 110_000, 'lbm')
        prob.set_val(Mission.TOTAL_FUEL_MASS, 30_000, 'lbm')
        prob.set_val(Aircraft.Fuel.MAX_CAPACITY_MASS, 20_000, units='lbm')
        prob.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, 750, units='lbm')
        prob.set_val(Mission.Taxi.FUEL_MASS_TAXI_IN, 20, units='lbm')

        prob.run_model()

        expected_values = {
            Mission.Constraints.MASS_RESIDUAL: (-360, 'lbm'),
            Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY: (-10750, 'lbm'),
            Mission.BLOCK_FUEL_MASS: (20020, 'lbm'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob.get_val(var_name, units=units), expected, 1e-10)

        constraints = prob.list_driver_vars(out_stream=None)['constraints'][0][0]
        with self.subTest('constraint_test'):
            assert_near_equal(constraints, Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-9)

    def test_no_constraint(self):
        """Test the component again, this time not adding the excess fuel capacity constraint."""
        local_phase_info = deepcopy(phase_info)

        local_phase_info['descent']['user_options']['reserve'] = True
        prob = om.Problem()
        prob.model.add_subsystem(
            'fuel_group', FuelSummationGroup(mission_info=local_phase_info), promotes=['*']
        )

        setup_model_options(
            prob,
            AviaryValues(
                {
                    Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT: (True, 'unitless'),
                }
            ),
        )

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        with self.subTest('constraint_test'):
            try:
                constraints = prob.list_driver_vars(out_stream=None)['constraints'][0][0]
                # assert_near_equal(constraints, Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY)
            except IndexError:
                pass
            else:
                raise UserWarning(
                    f'Unexpected constraints are active in the problem: {constraints}'
                )


if __name__ == '__main__':
    unittest.main()
