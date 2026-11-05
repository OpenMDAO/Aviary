import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.landing import (
    LandingGearMass,
    LandingMass,
    TotalLandingGearMass,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import extract_options, setup_model_options
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class LandingMassTestCase(unittest.TestCase):
    """Tests for the LandingMass component against GASP output, including a BWB model."""

    def _make_prob(self, gross_mass, landing_to_takeoff_mass_ratio):
        prob = om.Problem()
        prob.model.add_subsystem('landing_mass', LandingMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=gross_mass, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO,
            val=landing_to_takeoff_mass_ratio,
            units='unitless',
        )

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def _check(self, prob, expected_values, tol, check_partials=False):
        prob.run_model()

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        if check_partials:
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_case1(self):
        """This is the large single aisle 1 V3 test case."""
        prob = self._make_prob(
            gross_mass=175400,  # bug fixed value and original value
            landing_to_takeoff_mass_ratio=1.0,
        )
        expected_values = {
            Aircraft.Design.TOUCHDOWN_MASS_MAX: 175400.0,
        }
        self._check(prob, expected_values, tol=1e-8, check_partials=True)

    def test_case2(self):
        prob = self._make_prob(
            gross_mass=175400,  # bug fixed value and original value
            landing_to_takeoff_mass_ratio=1.0,
        )
        expected_values = {
            Aircraft.Design.TOUCHDOWN_MASS_MAX: 175400.0,
        }
        # same single code path as test_case1, just re-derived from the same inputs;
        # no need for its own partials check
        self._check(prob, expected_values, tol=5e-4)

    def test_bwb(self):
        """GASP BWB model."""
        prob = self._make_prob(
            gross_mass=150000,
            landing_to_takeoff_mass_ratio=1.0,
        )
        expected_values = {
            Aircraft.Design.TOUCHDOWN_MASS_MAX: 150000.0,
        }
        # same single code path as test_case1, just different numbers; no need for
        # its own partials check
        self._check(prob, expected_values, tol=1e-7)


@use_tempdirs
class TotalLandingGearMassTestCase(unittest.TestCase):
    """Tests for the TotalLandingGearMass component against GASP output, including a BWB model."""

    def _make_prob(
        self,
        num_engines,
        touchdown_mass_max,
        vertical_mount_location,
        mass_coefficient,
        total_mass_scaler,
        clearance_ratio,
        avg_diameter,
    ):
        prob = om.Problem()
        prob.model.add_subsystem('total_landing_gear', TotalLandingGearMass(), promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.Design.TOUCHDOWN_MASS_MAX, val=touchdown_mass_max, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=vertical_mount_location, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=mass_coefficient, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=total_mass_scaler, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=clearance_ratio, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Nacelle.AVG_DIAMETER, val=avg_diameter, units='ft')

        setup_model_options(
            prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: (num_engines, 'unitless')})
        )

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def _check(self, prob, expected_values, tol, check_partials=False):
        prob.run_model()

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        if check_partials:
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_case1(self):
        """This is the large single aisle 1 V3 test case."""
        prob = self._make_prob(
            num_engines=[2],
            touchdown_mass_max=175400.0,
            vertical_mount_location=0.0,
            mass_coefficient=0.04,  # bug fixed value and original value
            total_mass_scaler=1.0,
            clearance_ratio=0.2,  # bug fixed value and original value
            avg_diameter=7.35,  # bug fixed value and original value
        )
        expected_values = {
            Aircraft.LandingGear.TOTAL_MASS: 7510.885838,
        }
        # this is the "normal" code path; sibling test_case2 and test_bwb exercise the
        # same equations/Jacobian with different numbers, so only this one checks partials
        self._check(prob, expected_values, tol=1e-8, check_partials=True)

    def test_case2(self):
        prob = self._make_prob(
            num_engines=[2],
            touchdown_mass_max=175400.0,
            vertical_mount_location=0.1,
            mass_coefficient=0.04,  # bug fixed value and original value
            total_mass_scaler=1.0,
            clearance_ratio=0.0,
            avg_diameter=0.0,
        )
        expected_values = {
            Aircraft.LandingGear.TOTAL_MASS: 7016,
        }
        # same smooth vertical_mount_location blend as test_case1, just a different
        # point on the curve; no need for its own partials check
        self._check(prob, expected_values, tol=5e-4)

    def test_multiengine(self):
        """Multiple engine types exercise the KSfunction max-clearance code path."""
        prob = om.Problem()
        prob.model.add_subsystem('total_landing_gear', TotalLandingGearMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Design.TOUCHDOWN_MASS_MAX, val=152000, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=[0.0, 0.15], units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Nacelle.AVG_DIAMETER, val=[7.5, 8.22], units='ft')

        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 4]))
        prob.model_options['*'] = extract_options(options)

        prob.setup(check=False, force_alloc_complex=True)

        expected_values = {
            Aircraft.LandingGear.TOTAL_MASS: 6605.095476,
        }
        self._check(prob, expected_values, tol=5e-4, check_partials=True)

    def test_bwb(self):
        """GASP BWB model."""
        prob = self._make_prob(
            num_engines=[2],
            touchdown_mass_max=150000.0,
            vertical_mount_location=0.5,
            mass_coefficient=0.0520,
            total_mass_scaler=1.0,
            clearance_ratio=0.2,
            avg_diameter=7.35163168,
        )
        expected_values = {
            Aircraft.LandingGear.TOTAL_MASS: 7800.0,
        }
        # same smooth vertical_mount_location blend as test_case1, just a different
        # point on the curve; no need for its own partials check
        self._check(prob, expected_values, tol=1e-7)

    def test_alt_gravity(self):
        prob = om.Problem()

        prob.model.add_subsystem('total_landing_gear', TotalLandingGearMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Design.TOUCHDOWN_MASS_MAX, val=175400.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless')
        prob.model.set_input_defaults(Aircraft.Nacelle.AVG_DIAMETER, val=7.35, units='ft')

        setup_model_options(
            prob,
            AviaryValues(
                {Aircraft.Engine.NUM_ENGINES: ([2], 'unitless'), Mission.GRAVITY: (35, 'ft/s**2')}
            ),
        )
        prob.setup(check=False, force_alloc_complex=True)

        expected_values = {
            Aircraft.LandingGear.TOTAL_MASS: 8170.59139663,
        }
        # this is the "normal" code path; sibling test_case2 and test_bwb exercise the
        # same equations/Jacobian with different numbers, so only this one checks partials
        self._check(prob, expected_values, tol=1e-8, check_partials=True)


@use_tempdirs
class LandingGearMassTestCase(unittest.TestCase):
    """Tests for the LandingGearMass component against GASP output, including a BWB model."""

    def _make_prob(self, total_mass, main_gear_mass_fraction):
        prob = om.Problem()
        prob.model.add_subsystem('landing_gear', LandingGearMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=total_mass, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION,
            val=main_gear_mass_fraction,
            units='unitless',
        )

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def _check(self, prob, expected_values, tol, check_partials=False):
        prob.run_model()

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        if check_partials:
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_case1(self):
        """This is the large single aisle 1 V3 test case."""
        prob = self._make_prob(
            total_mass=7510.885838,  # bug fixed value and original value
            main_gear_mass_fraction=0.85,  # bug fixed value and original value
        )
        expected_values = {
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6384.2529623,
            Aircraft.LandingGear.NOSE_GEAR_MASS: 1126.6328757,
        }
        # this is the only code path in this component; sibling test_case2,
        # test_multiengine, and test_bwb exercise the same equations/Jacobian with
        # different numbers, so only this one checks partials
        self._check(prob, expected_values, tol=1e-8, check_partials=True)

    def test_case2(self):
        prob = self._make_prob(
            total_mass=7016,
            main_gear_mass_fraction=0.85,  # bug fixed value and original value
        )
        expected_values = {
            Aircraft.LandingGear.MAIN_GEAR_MASS: 5963.6,
            Aircraft.LandingGear.NOSE_GEAR_MASS: 1052.4,
        }
        self._check(prob, expected_values, tol=5e-4)

    def test_multiengine(self):
        prob = self._make_prob(
            total_mass=6605.095476,
            main_gear_mass_fraction=0.85,
        )
        expected_values = {
            Aircraft.LandingGear.MAIN_GEAR_MASS: 5614.3311546,
            Aircraft.LandingGear.NOSE_GEAR_MASS: 990.7643214,
        }
        self._check(prob, expected_values, tol=5e-4)

    def test_bwb(self):
        """GASP BWB model."""
        prob = self._make_prob(
            total_mass=7800.0,
            main_gear_mass_fraction=0.85,
        )
        expected_values = {
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6630.0,
            Aircraft.LandingGear.NOSE_GEAR_MASS: 1170.0,
        }
        self._check(prob, expected_values, tol=1e-7)


if __name__ == '__main__':
    unittest.main()
