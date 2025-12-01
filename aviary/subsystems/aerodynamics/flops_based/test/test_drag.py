import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.aerodynamics.flops_based.computed_aero_group import ComputedDrag
from aviary.subsystems.aerodynamics.flops_based.drag import ScaledCD, SimpleDrag, TotalDrag
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


class SimpleDragTest(unittest.TestCase):
    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        mission_data: AviaryValues = mission_test_data[case_name]

        # area = 4 digits precision
        # FCDSUB - 2 digits precision
        # FCDSUP - 2 digits precision
        inputs_keys = (
            Aircraft.Wing.AREA,
            Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
            Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
        )

        # dynamic pressure = 4 digits precision
        # drag coefficient = 5 digits precision
        mission_keys = (
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            'CD_prescaled',
            'CD',
            Dynamic.Atmosphere.MACH,
        )

        # drag = 4 digits precision
        outputs_keys = (Dynamic.Vehicle.DRAG,)

        # using lowest precision from all available data should "always" work
        # - will a higher precision comparison work? find a practical tolerance that fits
        #   the data set
        tol = 2e-4

        prob = om.Problem()
        model = prob.model

        q, _ = mission_data.get_item(Dynamic.Atmosphere.DYNAMIC_PRESSURE)
        nn = len(q)
        model.add_subsystem('simple_drag', SimpleDrag(num_nodes=nn), promotes=['*'])
        model.add_subsystem('simple_cd', ScaledCD(num_nodes=nn), promotes=['*'])

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

        assert_near_equal(prob.get_val('CD'), mission_simple_CD[case_name], 1e-6)
        assert_near_equal(prob.get_val(Dynamic.Vehicle.DRAG), mission_simple_drag[case_name], 1e-6)


class TotalDragTest(unittest.TestCase):
    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        mission_data: AviaryValues = mission_test_data[case_name]

        # area = 4 digits precision
        # FCDSUB - 2 digits precision
        # FCDSUP - 2 digits precision
        inputs_keys = (
            Aircraft.Wing.AREA,
            Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
            Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
            Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
            Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
        )

        # dynamic pressure = 4 digits precision
        # drag coefficient = 5 digits precision
        mission_keys = (
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            Dynamic.Atmosphere.MACH,
            'CD0',
            'CDI',
        )

        # drag = 4 digits precision
        outputs_keys = ('CD_prescaled', 'CD', Dynamic.Vehicle.DRAG)

        # using lowest precision from all available data should "always" work
        # - will a higher precision comparison work? find a practical tolerance that fits
        #   the data set
        tol = 2e-4

        prob = om.Problem()
        model = prob.model

        q, _ = mission_data.get_item(Dynamic.Atmosphere.DYNAMIC_PRESSURE)
        nn = len(q)
        model.add_subsystem('total_drag', TotalDrag(num_nodes=nn), promotes=['*'])

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

        assert_near_equal(prob.get_val('CD'), mission_total_CD[case_name], 1e-6)
        assert_near_equal(prob.get_val(Dynamic.Vehicle.DRAG), mission_total_drag[case_name], 1e-6)


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
            'computed_drag',
            ComputedDrag(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['CD', Dynamic.Vehicle.DRAG],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('skin_friction_drag_coeff', 0.01 * cdf)
        prob.set_val('pressure_drag_coeff', 0.01 * cdp)
        prob.set_val('compress_drag_coeff', 0.01 * cdc)
        prob.set_val('induced_drag_coeff', 0.01 * cdi)
        prob.set_val(Dynamic.Atmosphere.MACH, M)

        prob.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 0.7)
        prob.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 0.3)
        prob.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.4)
        prob.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.1)
        prob.set_val(Aircraft.Wing.AREA, 1370, units='ft**2')
        prob.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, [206.0, 205.6], 'lbf/ft**2')

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)

        assert_near_equal(prob.get_val('CD'), [0.0249732, 0.0297451], 1e-6)
        assert_near_equal(prob.get_val(Dynamic.Vehicle.DRAG), [31350.8, 37268.8], 1e-6)


# region - mission test data taken from the baseline FLOPS output for each case
#    - first "# DETAILED FLIGHT SEGMENT SUMMARY"
#        - second "SEGMENT..."
#            - first three points
# NOTE: FLOPS bakes FCDO, FCDI, FCDSUB, and FCDSUP into the final reported drag
# coefficient - no intermediate calculations are reported in the "# DETAILED FLIGHT
# SEGMENT SUMMARY", so we have to back calculate what we can
def _add_drag_coefficients(
    case_name,
    mission_data: AviaryValues,
    M: np.ndarray,
    CD_scaled: np.ndarray,
    CD0_scaled: np.ndarray,
    CDI_scaled: np.ndarray,
):
    """Insert drag coefficients into the mission data, undoing FLOPS scaling."""
    flops_inputs = get_flops_inputs(case_name)
    FCDSUB = flops_inputs.get_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR)
    FCDSUP = flops_inputs.get_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR)

    idx_sup = np.where(M >= 1.0)
    mission_data.set_val('CD', CD_scaled)
    CD = CD_scaled / FCDSUB
    CD[idx_sup] = CD_scaled[idx_sup] / FCDSUP
    mission_data.set_val('CD_prescaled', CD)
    mission_data.set_val('CD', CD_scaled)

    FCD0 = flops_inputs.get_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR)
    CD0 = CD0_scaled / FCD0
    mission_data.set_val('CD0', CD0)

    FCDI = flops_inputs.get_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR)
    CDI = CDI_scaled / FCDI
    mission_data.set_val('CDI', CDI)


mission_test_data = {}
mission_simple_CD = {}
mission_simple_drag = {}
mission_total_CD = {}
mission_total_drag = {}

key = 'LargeSingleAisle1FLOPS'
mission_test_data[key] = _mission_data = AviaryValues()
_mission_data.set_val(
    Dynamic.Atmosphere.DYNAMIC_PRESSURE, np.array([206.0, 205.6, 205.4]), 'lbf/ft**2'
)
_mission_data.set_val(Dynamic.Vehicle.MASS, np.array([176751.0, 176400.0, 176185.0]), 'lbm')
_mission_data.set_val(Dynamic.Vehicle.DRAG, np.array([9350.0, 9333.0, 9323.0]), 'lbf')

M = np.array([0.7750, 0.7750, 0.7750])
CD_scaled = np.array([0.03313, 0.03313, 0.03313])
CD0_scaled = np.array([0.02012, 0.02013, 0.02013])
CDI_scaled = np.array([0.01301, 0.01301, 0.01300])

_mission_data.set_val(Dynamic.Atmosphere.MACH, M)
_add_drag_coefficients(key, _mission_data, M, CD_scaled, CD0_scaled, CDI_scaled)

mission_simple_CD[key] = np.array([0.03313, 0.03313, 0.03313])
mission_simple_drag[key] = np.array([41590.643508, 41509.884977, 41469.505712])
mission_total_CD[key] = np.array([0.03313, 0.03314, 0.03313])
mission_total_drag[key] = np.array([41590.64350841, 41522.41437213, 41469.50571178])

key = 'LargeSingleAisle2FLOPS'
mission_test_data[key] = _mission_data = AviaryValues()
_mission_data.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, [215.4, 215.4, 215.4], 'lbf/ft**2')
_mission_data.set_val(Dynamic.Vehicle.MASS, [169730.0, 169200.0, 167400.0], 'lbm')
_mission_data.set_val(Dynamic.Vehicle.DRAG, [9542.0, 9512.0, 9411.0], 'lbf')

M = np.array([0.7850, 0.7850, 0.7850])
CD_scaled = np.array([0.03304, 0.03293, 0.03258])
CD0_scaled = np.array([0.02016, 0.02016, 0.02016])
CDI_scaled = np.array([0.01288, 0.01277, 0.01242])

_mission_data.set_val(Dynamic.Atmosphere.MACH, M)
_add_drag_coefficients(key, _mission_data, M, CD_scaled, CD0_scaled, CDI_scaled)

mission_simple_CD[key] = np.array([0.03304, 0.03293, 0.03258])
mission_simple_drag[key] = np.array([42452.271402, 42310.935148, 41861.228883])
mission_total_CD[key] = np.array([0.03304, 0.03293, 0.03258])
mission_total_drag[key] = np.array([42452.27140246, 42310.93514779, 41861.22888293])

key = 'AdvancedSingleAisle'
mission_test_data[key] = _mission_data = AviaryValues()
_mission_data.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, [208.4, 288.5, 364.0], 'lbf/ft**2')
_mission_data.set_val(Dynamic.Vehicle.MASS, [128777.0, 128721.0, 128667.0], 'lbm')
_mission_data.set_val(Dynamic.Vehicle.DRAG, [5837.0, 6551.0, 7566.0], 'lbf')

M = np.array([0.4522, 0.5321, 0.5985])
CD_scaled = np.array([0.02296, 0.01861, 0.01704])
CD0_scaled = np.array([0.01611, 0.01569, 0.01556])
CDI_scaled = np.array([0.00806, 0.00390, 0.00237])

_mission_data.set_val(Dynamic.Atmosphere.MACH, M)
_add_drag_coefficients(key, _mission_data, M, CD_scaled, CD0_scaled, CDI_scaled)
# endregion - mission test data taken from the baseline FLOPS output for each case

mission_simple_CD[key] = np.array([0.02296, 0.01861, 0.01704])
mission_simple_drag[key] = np.array([25966.645302, 29136.570888, 33660.241019])
mission_total_CD[key] = np.array([0.0229615, 0.0186105, 0.0170335])
mission_total_drag[key] = np.array([25968.341729, 29137.353709, 33647.401139])


if __name__ == '__main__':
    unittest.main()
