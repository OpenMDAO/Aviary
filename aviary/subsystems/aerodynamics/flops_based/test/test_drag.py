import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)
from parameterized import parameterized

from aviary.subsystems.aerodynamics.flops_based.computed_aero_group import ComputedDrag
from aviary.subsystems.aerodynamics.flops_based.drag import SimpleDrag, TotalDrag
from aviary.utils.aviary_values import AviaryValues
from aviary.validation_cases.validation_tests import (get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Dynamic

data_sets = get_flops_case_names(
    only=['LargeSingleAisle1FLOPS', 'LargeSingleAisle2FLOPS', 'N3CC'])


class SimpleDragTest(unittest.TestCase):

    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        dynamics_data: AviaryValues = dynamics_test_data[case_name]

        # area = 4 digits precision
        # FCDSUB - 2 digits precision
        # FCDSUP - 2 digits precision
        inputs_keys = (
            Aircraft.Wing.AREA, Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
            Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR)

        # dynamic pressure = 4 digits precision
        # drag coefficient = 5 digits precision
        dynamics_keys = (Dynamic.Mission.DYNAMIC_PRESSURE,
                         'drag_coefficient', Dynamic.Mission.MACH)

        # drag = 4 digits precision
        outputs_keys = (Dynamic.Mission.DRAG,)

        # using lowest precision from all available data should "always" work
        # - will a higher precision comparison work? find a practical tolerance that fits
        #   the data set
        tol = 2e-4

        prob = om.Problem()
        model = prob.model

        q, _ = dynamics_data.get_item(Dynamic.Mission.DYNAMIC_PRESSURE)
        nn = len(q)
        model.add_subsystem('simple_drag', SimpleDrag(num_nodes=nn), promotes=['*'])

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


class TotalDragTest(unittest.TestCase):

    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        dynamics_data: AviaryValues = dynamics_test_data[case_name]

        # area = 4 digits precision
        # FCDSUB - 2 digits precision
        # FCDSUP - 2 digits precision
        inputs_keys = (
            Aircraft.Wing.AREA, Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
            Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
            Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
            Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR)

        # dynamic pressure = 4 digits precision
        # drag coefficient = 5 digits precision
        dynamics_keys = (Dynamic.Mission.DYNAMIC_PRESSURE,
                         Dynamic.Mission.MACH, 'CD0', 'CDI')

        # drag = 4 digits precision
        outputs_keys = ('drag_coefficient', Dynamic.Mission.DRAG)

        # using lowest precision from all available data should "always" work
        # - will a higher precision comparison work? find a practical tolerance that fits
        #   the data set
        tol = 2e-4

        prob = om.Problem()
        model = prob.model

        q, _ = dynamics_data.get_item(Dynamic.Mission.DYNAMIC_PRESSURE)
        nn = len(q)
        model.add_subsystem('total_drag', TotalDrag(num_nodes=nn), promotes=['*'])

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


class ComputedDragTest(unittest.TestCase):

    def test_derivs(self):
        nn = 2

        cdp = np.array([2.5, 1.3])
        cdc = np.array([0.3, 0.7])
        cdi = np.array([1.5, 4.3])
        cdf = np.array([0.534, 0.763])
        M = np.array([0.8, 1.2])

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            'computed_drag', ComputedDrag(num_nodes=nn), promotes_inputs=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val('skin_friction_drag_coeff', cdf)
        prob.set_val('pressure_drag_coeff', cdp)
        prob.set_val('compress_drag_coeff', cdc)
        prob.set_val('induced_drag_coeff', cdi)
        prob.set_val(Dynamic.Mission.MACH, M)

        prob.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 0.7)
        prob.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 0.3)
        prob.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.4)
        prob.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.1)
        prob.set_val(Aircraft.Wing.AREA, 13.7)

        derivs = prob.check_partials(out_stream=None, method="cs")

        # TODO: need to test outputs too
        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)


# region - dynamic test data taken from the baseline FLOPS output for each case
#    - first "# DETAILED FLIGHT SEGMENT SUMMARY"
#        - second "SEGMENT..."
#            - first three points
# NOTE: FLOPS bakes FCDO, FCDI, FCDSUB, and FCDSUP into the final reported drag
# coefficient - no intermediate calculations are reported in the "# DETAILED FLIGHT
# SEGMENT SUMMARY", so we have to back calculate what we can
def _add_drag_coefficients(
    case_name, dynamics_data: AviaryValues, M: np.ndarray, CD_scaled: np.ndarray,
    CD0_scaled: np.ndarray, CDI_scaled: np.ndarray
):
    '''
    Insert drag coefficients into the dynamics data, undoing FLOPS scaling.
    '''
    flops_inputs = get_flops_inputs(case_name)
    FCDSUB = flops_inputs.get_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR)
    FCDSUP = flops_inputs.get_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR)

    idx_sup = np.where(M >= 1.0)
    CD = CD_scaled / FCDSUB
    CD[idx_sup] = CD_scaled[idx_sup] / FCDSUP
    dynamics_data.set_val('drag_coefficient', CD)

    FCD0 = flops_inputs.get_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR)
    CD0 = CD0_scaled / FCD0
    dynamics_data.set_val('CD0', CD0)

    FCDI = flops_inputs.get_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR)
    CDI = CDI_scaled / FCDI
    dynamics_data.set_val('CDI', CDI)


dynamics_test_data = {}

key = 'LargeSingleAisle1FLOPS'
dynamics_test_data[key] = _dynamics_data = AviaryValues()
_dynamics_data.set_val(Dynamic.Mission.DYNAMIC_PRESSURE, np.array(
    [206.0, 205.6, 205.4]), 'lbf/ft**2')
_dynamics_data.set_val(Dynamic.Mission.MASS, np.array(
    [176751., 176400., 176185.]), 'lbm')
_dynamics_data.set_val(Dynamic.Mission.DRAG, np.array([9350., 9333., 9323.]), 'lbf')

M = np.array([0.7750, 0.7750, 0.7750])
CD_scaled = np.array([0.03313, 0.03313, 0.03313])
CD0_scaled = np.array([0.02012, 0.02013, 0.02013])
CDI_scaled = np.array([0.01301, 0.01301, 0.01300])

_dynamics_data.set_val(Dynamic.Mission.MACH, M)
_add_drag_coefficients(key, _dynamics_data, M, CD_scaled, CD0_scaled, CDI_scaled)

key = 'LargeSingleAisle2FLOPS'
dynamics_test_data[key] = _dynamics_data = AviaryValues()
_dynamics_data.set_val(Dynamic.Mission.DYNAMIC_PRESSURE, [
                       215.4, 215.4, 215.4], 'lbf/ft**2')
_dynamics_data.set_val(Dynamic.Mission.MASS, [169730., 169200., 167400.], 'lbm')
_dynamics_data.set_val(Dynamic.Mission.DRAG, [9542., 9512., 9411.], 'lbf')

M = np.array([0.7850, 0.7850, 0.7850])
CD_scaled = np.array([0.03304, 0.03293, 0.03258])
CD0_scaled = np.array([0.02016, 0.02016, 0.02016])
CDI_scaled = np.array([0.01288, 0.01277, 0.01242])

_dynamics_data.set_val(Dynamic.Mission.MACH, M)
_add_drag_coefficients(key, _dynamics_data, M, CD_scaled, CD0_scaled, CDI_scaled)

key = 'N3CC'
dynamics_test_data[key] = _dynamics_data = AviaryValues()
_dynamics_data.set_val(Dynamic.Mission.DYNAMIC_PRESSURE, [
                       208.4, 288.5, 364.0], 'lbf/ft**2')
_dynamics_data.set_val(Dynamic.Mission.MASS, [128777., 128721., 128667.], 'lbm')
_dynamics_data.set_val(Dynamic.Mission.DRAG, [5837., 6551., 7566.], 'lbf')

M = np.array([0.4522, 0.5321, 0.5985])
CD_scaled = np.array([0.02296, 0.01861, 0.01704])
CD0_scaled = np.array([0.01611, 0.01569, 0.01556])
CDI_scaled = np.array([0.00806, 0.00390, 0.00237])

_dynamics_data.set_val(Dynamic.Mission.MACH, M)
_add_drag_coefficients(key, _dynamics_data, M, CD_scaled, CD0_scaled, CDI_scaled)
# endregion - dynamic test data taken from the baseline FLOPS output for each case


if __name__ == "__main__":
    unittest.main()
