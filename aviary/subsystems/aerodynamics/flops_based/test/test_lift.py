import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.aerodynamics.flops_based.lift import LiftEqualsWeight, SimpleLift
from aviary.utils.aviary_values import AviaryValues
from aviary.validation_cases.validation_tests import (
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Dynamic

data_sets = get_flops_case_names(
    only=['LargeSingleAisle1FLOPS', 'LargeSingleAisle2FLOPS', 'advanced_single_aisle']
)


class SimpleLiftTest(unittest.TestCase):
    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        mission_data: AviaryValues = mission_test_data[case_name]

        # area = 4 digits precision
        inputs_keys = (Aircraft.Wing.AREA,)

        # dynamic pressure = 4 digits precision
        # lift coefficient = 5 digits precision
        mission_keys = (Dynamic.Atmosphere.DYNAMIC_PRESSURE, 'cl')

        # lift = 6 digits precision
        outputs_keys = (Dynamic.Vehicle.LIFT,)

        # use lowest precision from all available data
        tol = 1e-4

        prob = om.Problem()
        model = prob.model

        q, _ = mission_data.get_item(Dynamic.Atmosphere.DYNAMIC_PRESSURE)
        nn = len(q)
        model.add_subsystem('simple_lift', SimpleLift(num_nodes=nn), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        for key in inputs_keys:
            val, units = flops_inputs.get_item(key)
            prob.set_val(key, val, units)

        for key in mission_keys:
            val, units = mission_data.get_item(key)
            prob.set_val(key, val, units)

        prob.run_model()

        for key in outputs_keys:
            try:
                desired, units = mission_data.get_item(key)
                actual = prob.get_val(key, units)

                assert_near_equal(actual, desired, tol)

            except ValueError as error:
                msg = f'"{key}": {error!s}'

                raise ValueError(msg) from None

        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=2.5e-10, rtol=1e-12)

        assert_near_equal(prob.get_val(Dynamic.Vehicle.LIFT), mission_simple_data[case_name], 1e-6)


class LiftEqualsWeightTest(unittest.TestCase):
    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        mission_data: AviaryValues = mission_test_data[case_name]

        # area = 4 digits precision
        inputs_keys = (Aircraft.Wing.AREA,)

        # dynamic pressure = 4 digits precision
        # mass = 6 digits precision
        mission_keys = (Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Vehicle.MASS)

        # lift coefficient = 5 digits precision
        # lift = 6 digits precision
        outputs_keys = ('cl', Dynamic.Vehicle.LIFT)

        # use lowest precision from all available data
        tol = 1e-4

        prob = om.Problem()
        model = prob.model

        q, _ = mission_data.get_item(Dynamic.Atmosphere.DYNAMIC_PRESSURE)
        nn = len(q)

        model.add_subsystem('lift_equals_weight', LiftEqualsWeight(num_nodes=nn), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        for key in inputs_keys:
            val, units = flops_inputs.get_item(key)
            prob.set_val(key, val, units)

        for key in mission_keys:
            val, units = mission_data.get_item(key)
            prob.set_val(key, val, units)

        prob.run_model()

        for key in outputs_keys:
            try:
                desired, units = mission_data.get_item(key)
                actual = prob.get_val(key, units)

                assert_near_equal(actual, desired, tol)

            except ValueError as error:
                msg = f'"{key}": {error!s}'

                raise ValueError(msg) from None

        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=2.5e-10, rtol=1e-12)

        assert_near_equal(prob.get_val(Dynamic.Vehicle.LIFT), mission_equal_data[case_name], 1e-6)


# dynamic test data taken from the baseline FLOPS output for each case
#    - first: # DETAILED FLIGHT SEGMENT SUMMARY
#        - first: SEGMENT... CRUISE
#            - first three points
mission_test_data = {}
mission_simple_data = {}
mission_equal_data = {}

mission_test_data['LargeSingleAisle1FLOPS'] = _mission_data = AviaryValues()
_mission_data.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, [206.0, 205.6, 205.4], 'lbf/ft**2')
_mission_data.set_val('cl', [0.62630, 0.62623, 0.62619])
_mission_data.set_val(Dynamic.Vehicle.LIFT, [176751.0, 176400.0, 176185.0], 'lbf')
_mission_data.set_val(Dynamic.Vehicle.MASS, [176751.0, 176400.0, 176185.0], 'lbm')
mission_simple_data['LargeSingleAisle1FLOPS'] = [786242.68, 784628.29, 783814.96]
mission_equal_data['LargeSingleAisle1FLOPS'] = [786227.62, 784666.29, 783709.93]

mission_test_data['LargeSingleAisle2FLOPS'] = _mission_data = AviaryValues()
_mission_data.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, [215.4, 215.4, 215.4], 'lbf/ft**2')
_mission_data.set_val('cl', [0.58761, 0.58578, 0.57954])
_mission_data.set_val(Dynamic.Vehicle.LIFT, [169730.0, 169200.0, 167400.0], 'lbf')
_mission_data.set_val(Dynamic.Vehicle.MASS, [169730.0, 169200.0, 167400.0], 'lbm')
mission_simple_data['LargeSingleAisle2FLOPS'] = [755005.42, 752654.10, 744636.48]
mission_equal_data['LargeSingleAisle2FLOPS'] = [754996.65, 752639.10, 744632.30]

mission_test_data['advanced_single_aisle'] = _mission_data = AviaryValues()
_mission_data.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, [208.4, 288.5, 364.0], 'lbf/ft**2')
_mission_data.set_val('cl', [0.50651, 0.36573, 0.28970])
_mission_data.set_val(Dynamic.Vehicle.LIFT, [128777.0, 128721.0, 128667.0], 'lbf')
_mission_data.set_val(Dynamic.Vehicle.MASS, [128777.0, 128721.0, 128667.0], 'lbm')
mission_simple_data['advanced_single_aisle'] = [572838.22, 572601.72, 572263.60]
mission_equal_data['advanced_single_aisle'] = [572828.63, 572579.53, 572339.33]


if __name__ == '__main__':
    unittest.main()
