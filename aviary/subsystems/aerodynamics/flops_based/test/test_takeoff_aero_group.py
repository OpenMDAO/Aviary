import unittest

import numpy as np
import openmdao.api as om
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.utils.aviary_values import AviaryValues, get_items
from aviary.models.N3CC.N3CC_data import (
    N3CC, takeoff_subsystem_options, takeoff_subsystem_options_spoilers)
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.variables import Dynamic as _Dynamic
from aviary.variable_info.variables import Mission
from aviary.variable_info.enums import LegacyCode

Dynamic = _Dynamic.Mission


class TestTakeoffAeroGroup(unittest.TestCase):

    def test_takeoff_aero_group(self):
        prob: om.Problem = make_problem(takeoff_subsystem_options)

        prob.run_model()

        tol = 1e-12

        for (key, (desired, units)) in get_items(_regression_data):
            try:
                actual = prob.get_val(key, units)

                assert_near_equal(actual, desired, tol)

            except ValueError as error:
                msg = f'"{key}": {error!s}'

                raise ValueError(msg) from None

        partials = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partials, atol=1e-10, rtol=1e-12)

    def test_takeoff_aero_group_spoiler(self):
        prob: om.Problem = make_problem(takeoff_subsystem_options_spoilers)

        prob.run_model()

        tol = 1e-12

        for (key, (desired, units)) in get_items(_regression_data_spoiler):
            try:
                actual = prob.get_val(key, units)

                assert_near_equal(actual, desired, tol)

            except ValueError as error:
                msg = f'"{key}": {error!s}'

                raise ValueError(msg) from None

        partials = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partials, atol=1e-10, rtol=1e-12)


def make_problem(subsystem_options={}):
    '''
    Return a problem that covers all three computational ranges:
    - in ground effect (on the ground);
    - in transition (in the air, but near enough to the ground for limited effect);
    - in free flight (no ground effect).
    '''
    nn = 3

    aviary_keys = (
        Mission.Takeoff.DRAG_COEFFICIENT_MIN,
        Aircraft.Wing.ASPECT_RATIO, Aircraft.Wing.HEIGHT,
        Aircraft.Wing.SPAN, Aircraft.Wing.AREA)

    aviary_inputs = AviaryValues(N3CC['inputs'])

    dynamic_inputs = AviaryValues({
        'angle_of_attack': (np.array([0., 2., 6.]), 'deg'),
        Dynamic.ALTITUDE: (np.array([0., 32., 55.]), 'm'),
        Dynamic.FLIGHT_PATH_ANGLE: (np.array([0., 0.5, 1.]), 'deg')})

    prob = om.Problem()

    prob.model.add_subsystem(
        "USatm",
        USatm1976Comp(num_nodes=nn),
        promotes_inputs=[("h", Dynamic.ALTITUDE)],
        promotes_outputs=[
            "rho", ("sos", Dynamic.SPEED_OF_SOUND), ("temp", Dynamic.TEMPERATURE),
            ("pres", Dynamic.STATIC_PRESSURE), "viscosity"])

    aero_builder = CoreAerodynamicsBuilder(code_origin=LegacyCode.FLOPS)

    prob.model.add_subsystem(
        name='core_aerodynamics',
        subsys=aero_builder.build_mission(num_nodes=nn,
                                          aviary_inputs=aviary_inputs,
                                          **subsystem_options['core_aerodynamics']),
        promotes_inputs=aero_builder.mission_inputs(
            **subsystem_options['core_aerodynamics']),
        promotes_outputs=aero_builder.mission_outputs(**subsystem_options['core_aerodynamics']))

    prob.model.set_input_defaults(Dynamic.ALTITUDE, np.zeros(nn), 'm')

    prob.setup(force_alloc_complex=True)

    for key in aviary_keys:
        try:
            val, units = aviary_inputs.get_item(key, ())

        except Exception as error:
            msg = f'"{key}": {error!s}'

            raise ValueError(msg) from None

        prob.set_val(key, val, units)

    for (key, (val, units)) in dynamic_inputs:
        prob.set_val(key, val, units)

    return prob


_units_lift = 'N'
_units_drag = 'N'


# NOTE:
# - results from `generate_regression_data()`
#    - last generated 2023 June 8
# - generate new regression data if, and only if, takeoff aero group is updated with a
#   more trusted implementation
_regression_data = AviaryValues({
    Dynamic.LIFT: (
        [3028.138891962988, 4072.059743068957, 6240.85493286], _units_lift),
    Dynamic.DRAG: (
        [434.6285684000267, 481.5245555324278, 586.0976806512001], _units_drag)})

# NOTE:
# - results from `generate_regression_data_spoiler()`
#    - last generated 2023 June 8
# - generate new regression data if, and only if, takeoff aero group is updated with a
#   more trusted implementation
_regression_data_spoiler = AviaryValues({
    Dynamic.LIFT: (
        [-1367.5937129210124, -323.67286181504335, 1845.1223279759993], _units_lift),
    Dynamic.DRAG: (
        [895.9091503940268, 942.8051375264279, 1047.3782626452], _units_drag)})


def generate_regression_data():
    '''
    Generate regression data that covers all three computational ranges:
    - in ground effect (on the ground);
    - in transition (in the air, but near enough to the ground for limited effect);
    - in free flight (no ground effect).

    Note, spoiler disabled.

    Notes
    -----
    Use this function to generate new regression data if, and only if, ground effect is
    updated with a more trusted implementation.
    '''
    _generate_regression_data(takeoff_subsystem_options)


def generate_regression_data_spoiler():
    '''
    Generate regression data that covers all three computational ranges:
    - in ground effect (on the ground);
    - in transition (in the air, but near enough to the ground for limited effect);
    - in free flight (no ground effect).

    Note, spoiler enabled.

    Notes
    -----
    Use this function to generate new regression data if, and only if, ground effect is
    updated with a more trusted implementation.
    '''
    _generate_regression_data(takeoff_subsystem_options_spoilers)


def _generate_regression_data(subsystem_options={}):
    '''
    Generate regression data that covers all three computational ranges:
    - in ground effect (on the ground);
    - in transition (in the air, but near enough to the ground for limited effect);
    - in free flight (no ground effect).

    Note, results are aero builder specific.

    Notes
    -----
    Use this function to generate new regression data if, and only if, ground effect is
    updated with a more trusted implementation.
    '''
    prob: om.Problem = make_problem(subsystem_options)

    prob.run_model()

    lift = prob.get_val(Dynamic.LIFT, _units_lift)
    drag = prob.get_val(Dynamic.DRAG, _units_drag)

    prob.check_partials(compact_print=True, method="cs")

    print('lift', *lift, sep=', ')
    print('drag', *drag, sep=', ')


if __name__ == "__main__":
    unittest.main()
    # test = TestTakeoffAeroGroup()
    # test.test_takeoff_aero_group()
