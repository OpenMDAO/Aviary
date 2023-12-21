import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.aerodynamics.flops_based.lift import (LiftEqualsWeight,
                                                             SimpleLift)
from aviary.utils.aviary_values import AviaryValues
from aviary.validation_cases.validation_tests import (get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Dynamic

data_sets = get_flops_case_names(
    only=['LargeSingleAisle1FLOPS', 'LargeSingleAisle2FLOPS', 'N3CC'])


class SimpleLiftTest(unittest.TestCase):

    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        dynamics_data: AviaryValues = dynamics_test_data[case_name]

        # area = 4 digits precision
        inputs_keys = (Aircraft.Wing.AREA,)

        # dynamic pressure = 4 digits precision
        # lift coefficient = 5 digits precision
        dynamics_keys = (Dynamic.Mission.DYNAMIC_PRESSURE, 'cl')

        # lift = 6 digits precision
        outputs_keys = (Dynamic.Mission.LIFT,)

        # use lowest precision from all available data
        tol = 1e-4

        prob = om.Problem()
        model = prob.model

        q, _ = dynamics_data.get_item(Dynamic.Mission.DYNAMIC_PRESSURE)
        nn = len(q)
        model.add_subsystem('simple_lift', SimpleLift(num_nodes=nn), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        for key in inputs_keys:
            val, units = flops_inputs.get_item(key)
            prob.set_val(key, val, units)

        for key in dynamics_keys:
            val, units = dynamics_data.get_item(key)
            prob.set_val(key, val, units)

        prob.run_model()

        for key in outputs_keys:
            try:
                desired, units = dynamics_data.get_item(key)
                actual = prob.get_val(key, units)

                assert_near_equal(actual, desired, tol)

            except ValueError as error:
                msg = f'"{key}": {error!s}'

                raise ValueError(msg) from None

        data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(data, atol=2.5e-10, rtol=1e-12)


class LiftEqualsWeightTest(unittest.TestCase):

    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        dynamics_data: AviaryValues = dynamics_test_data[case_name]

        # area = 4 digits precision
        inputs_keys = (Aircraft.Wing.AREA,)

        # dynamic pressure = 4 digits precision
        # mass = 6 digits precision
        dynamics_keys = (Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.MASS)

        # lift coefficient = 5 digits precision
        # lift = 6 digits precision
        outputs_keys = ('cl', Dynamic.Mission.LIFT)

        # use lowest precision from all available data
        tol = 1e-4

        prob = om.Problem()
        model = prob.model

        q, _ = dynamics_data.get_item(Dynamic.Mission.DYNAMIC_PRESSURE)
        nn = len(q)

        model.add_subsystem(
            'lift_equals_weight', LiftEqualsWeight(num_nodes=nn), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        for key in inputs_keys:
            val, units = flops_inputs.get_item(key)
            prob.set_val(key, val, units)

        for key in dynamics_keys:
            val, units = dynamics_data.get_item(key)
            prob.set_val(key, val, units)

        prob.run_model()

        for key in outputs_keys:
            try:
                desired, units = dynamics_data.get_item(key)
                actual = prob.get_val(key, units)

                assert_near_equal(actual, desired, tol)

            except ValueError as error:
                msg = f'"{key}": {error!s}'

                raise ValueError(msg) from None

        data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(data, atol=2.5e-10, rtol=1e-12)


# dynamic test data taken from the baseline FLOPS output for each case
#    - first: # DETAILED FLIGHT SEGMENT SUMMARY
#        - first: SEGMENT... CRUISE
#            - first three points
dynamics_test_data = {}

dynamics_test_data['LargeSingleAisle1FLOPS'] = _dynamics_data = AviaryValues()
_dynamics_data.set_val(Dynamic.Mission.DYNAMIC_PRESSURE, [
                       206.0, 205.6, 205.4], 'lbf/ft**2')
_dynamics_data.set_val('cl', [0.62630, 0.62623, 0.62619])
_dynamics_data.set_val(Dynamic.Mission.LIFT, [176751., 176400., 176185.], 'lbf')
_dynamics_data.set_val(Dynamic.Mission.MASS, [176751., 176400., 176185.], 'lbm')

dynamics_test_data['LargeSingleAisle2FLOPS'] = _dynamics_data = AviaryValues()
_dynamics_data.set_val(Dynamic.Mission.DYNAMIC_PRESSURE, [
                       215.4, 215.4, 215.4], 'lbf/ft**2')
_dynamics_data.set_val('cl', [0.58761, 0.58578, 0.57954])
_dynamics_data.set_val(Dynamic.Mission.LIFT, [169730., 169200., 167400.], 'lbf')
_dynamics_data.set_val(Dynamic.Mission.MASS, [169730., 169200., 167400.], 'lbm')

dynamics_test_data['N3CC'] = _dynamics_data = AviaryValues()
_dynamics_data.set_val(Dynamic.Mission.DYNAMIC_PRESSURE, [
                       208.4, 288.5, 364.0], 'lbf/ft**2')
_dynamics_data.set_val('cl', [0.50651, 0.36573, 0.28970])
_dynamics_data.set_val(Dynamic.Mission.LIFT, [128777., 128721., 128667.], 'lbf')
_dynamics_data.set_val(Dynamic.Mission.MASS, [128777., 128721., 128667.], 'lbm')


if __name__ == "__main__":
    unittest.main()
