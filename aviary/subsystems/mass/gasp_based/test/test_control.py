import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary import constants
from aviary.subsystems.mass.gasp_based.control import ControlMassGroup
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


# this is the large single aisle 1 V3 test case
@use_tempdirs
class ControlMassTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('control_mass', ControlMassGroup(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1392.1, units='ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.951, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )  # bug fixed value and original value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            Aircraft.Controls.MASS: 3945,
        }
        tol = 5e-4

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-11, rtol=1e-12)


@use_tempdirs
class BWBControlMassTestCase(unittest.TestCase):
    """GAST BWB model"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('control_mass', ControlMassGroup(), promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, 0.5, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85714286, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.97744787, units='unitless'
        )
        prob.model.set_input_defaults('min_dive_vel', 420, units='kn')
        prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, 16.5, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, 0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, 1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, 1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, 1, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Controls.CONTROL_MASS_INCREMENT, 0, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            Aircraft.Wing.SURFACE_CONTROL_MASS: 2045.5556421,
            Aircraft.Controls.MASS: 2174.28611375,
        }
        tol = 1e-7

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-11, rtol=1e-12)


# this is the large single aisle 1 V3 test case
@use_tempdirs
class ControlGroupTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'control_group',
            ControlMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1392.1, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.951, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )  # bug fixed value and original value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            Aircraft.Controls.COCKPIT_CONTROL_MASS: 137.25749725,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS: 0.0,
            Aircraft.Wing.SURFACE_CONTROL_MASS: 3807.92115815,
            Aircraft.Controls.MASS: 3945.0,
        }
        tol = 5e-4

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=3e-11, rtol=1e-12)


class ControlGroupTestCase2(unittest.TestCase):
    def setUp(self):
        import aviary.subsystems.mass.gasp_based.control as control

        # Set GRAV_ENGLISH_LBM = 1.1 to find errors that aren't
        # caught when GRAV_ENGLISH_LBM = 1 and is misplaced.
        constants.GRAV_ENGLISH_LBM = 1.1
        control.GRAV_ENGLISH_LBM = 1.1

        options = get_option_defaults()

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'control_group',
            ControlMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1392.1, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.951, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )  # bug fixed value and original value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.control as control

        constants.GRAV_ENGLISH_LBM = 1.0
        control.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            Aircraft.Controls.COCKPIT_CONTROL_MASS: 129.75209879,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS: 0.0,
            Aircraft.Wing.SURFACE_CONTROL_MASS: 3668.57521123,
            Aircraft.Controls.MASS: 3798.32731002,
        }
        tol = 5e-4

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=3e-11, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
