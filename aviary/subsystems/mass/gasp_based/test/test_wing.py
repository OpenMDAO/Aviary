import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.wing import (
    BWBWingMassGroup,
    BWBWingMassSolve,
    StrutAndFoldMass,
    StrutAndFoldMass,
    WingMassGroup,
    WingMassSolve,
    WingMassTotal,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class WingMassSolveTestCase(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wingfuel', WingMassSolve(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.HIGH_LIFT_MASS, val=3645, units='lbm')
        self.prob.model.set_input_defaults('c_strut_braced', val=1.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MATERIAL_FACTOR, val=1.2213063198183813, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER, val=0.98, units='unitless'
        )
        self.prob.model.set_input_defaults('c_gear_loc', val=1.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults('half_sweep', val=0.3947081519145335, units='rad')

        newton = self.prob.model.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-9
        newton.options['rtol'] = 1e-9
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['err_on_non_converge'] = True
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1
        newton.options['err_on_non_converge'] = False

        self.prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        with self.subTest(check='value'):
            assert_near_equal(self.prob['isolated_wing_mass'], 15830, tol)

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_alt_gravity(self):
        setup_model_options(
            self.prob,
            AviaryValues(
                {Aircraft.Engine.NUM_ENGINES: ([2], 'unitless'), Mission.GRAVITY: (35, 'ft/s**2')}
            ),
        )

        self.prob.run_model()

        tol = 5e-4
        with self.subTest(check='value'):
            assert_near_equal(self.prob['isolated_wing_mass'], 15830, tol)

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TotalWingMassTestCase(unittest.TestCase):
    """this is the large single aisle 1 V3 test case, with fold/strut variations."""

    def _make_prob(self, has_fold=False, has_strut=False):
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=has_fold, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=has_strut, units='unitless')

        prob = om.Problem()
        prob.model.add_subsystem('strut_fold', StrutAndFoldMass(), promotes=['*'])
        prob.model.add_subsystem('total', WingMassTotal(), promotes=['*'])

        # Input values below are the _MetaData defaults except where noted (marked "not actual GASP value").
        prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')

        if has_fold:
            prob.model.set_input_defaults(
                Aircraft.Wing.AREA, val=100, units='ft**2'
            )  # not actual GASP value
            prob.model.set_input_defaults(
                Aircraft.Wing.FOLDING_AREA, val=50, units='ft**2'
            )  # not actual GASP value
            prob.model.set_input_defaults(
                Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
            )  # not actual GASP value

        if has_strut:
            prob.model.set_input_defaults(
                Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units='unitless'
            )  # not actual GASP value

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_no_fold_no_strut(self):
        prob = self._make_prob(has_fold=False, has_strut=False)
        prob.run_model()

        tol = 5e-4
        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.Wing.MASS], 15830.0, tol)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_fold_no_strut(self):
        prob = self._make_prob(has_fold=True, has_strut=False)
        prob.run_model()

        tol = 5e-4
        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.Wing.MASS], 17413, tol)  # not actual GASP value

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_strut_no_fold(self):
        prob = self._make_prob(has_fold=False, has_strut=True)
        prob.run_model()

        tol = 5e-4
        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.Wing.MASS], 23745, tol)  # not actual GASP value

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_fold_and_strut(self):
        prob = self._make_prob(has_fold=True, has_strut=True)
        prob.run_model()

        tol = 5e-4
        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.Wing.MASS], 25328, tol)  # not actual GASP value

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class WingMassGroupTestCase(unittest.TestCase):
    """this is the large single aisle 1 V3 test case, with fold/strut variations."""

    def _make_prob(self, has_fold=False, has_strut=False):
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=has_fold, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=has_strut, units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, [2], units='unitless')

        prob = om.Problem()
        prob.model.add_subsystem('group', WingMassGroup(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.HIGH_LIFT_MASS, val=3645, units='lbm')
        prob.model.set_input_defaults('c_strut_braced', val=1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.MATERIAL_FACTOR, val=1.2213063198183813, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER, val=0.98, units='unitless'
        )
        prob.model.set_input_defaults('c_gear_loc', val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        prob.model.set_input_defaults('half_sweep', val=0.3947081519145335, units='rad')

        if has_fold:
            prob.model.set_input_defaults(
                Aircraft.Wing.AREA, val=100, units='ft**2'
            )  # not actual GASP value
            prob.model.set_input_defaults(
                Aircraft.Wing.FOLDING_AREA, val=50, units='ft**2'
            )  # not actual GASP value
            prob.model.set_input_defaults(
                Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
            )  # not actual GASP value

        if has_strut:
            prob.model.set_input_defaults(
                Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units='unitless'
            )  # not actual GASP value

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_no_fold_no_strut(self):
        prob = self._make_prob(has_fold=False, has_strut=False)
        prob.run_model()

        tol = 5e-4
        expected_values = {
            Aircraft.Wing.MASS: (15830, 'lbm'),
            'isolated_wing_mass': (15830, 'lbm'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_fold_no_strut(self):
        prob = self._make_prob(has_fold=True, has_strut=False)
        prob.run_model()

        tol = 5e-4
        expected_values = {
            Aircraft.Wing.MASS: (17417, 'lbm'),  # not actual GASP value
            'isolated_wing_mass': (15830, 'lbm'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_strut_no_fold(self):
        prob = self._make_prob(has_fold=False, has_strut=True)
        prob.run_model()

        tol = 5e-4
        expected_values = {
            Aircraft.Wing.MASS: (23750, 'lbm'),  # not actual GASP value
            'isolated_wing_mass': (15830, 'lbm'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_fold_and_strut(self):
        prob = self._make_prob(has_fold=True, has_strut=True)
        prob.run_model()

        tol = 5e-4
        expected_values = {
            Aircraft.Wing.MASS: (25333, 'lbm'),  # not actual GASP value
            'isolated_wing_mass': (15830, 'lbm'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class BWBWingMassSolveTestCase(unittest.TestCase):
    """this is BWB test case."""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('wingfuel', BWBWingMassSolve(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.HIGH_LIFT_MASS, 1068.88854499, units='lbm')
        prob.model.set_input_defaults('c_strut_braced', 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.77335889, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, 75.78, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MATERIAL_FACTOR, 1.19461189, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER, 1.05, units='unitless'
        )
        prob.model.set_input_defaults('c_gear_loc', 0.95, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults('half_sweep', 0.479839474, units='rad')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LIFT_COEFFICIENT_RATIO_BODY_TO_WING, 0.35, units='unitless'
        )

        newton = self.prob.model.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-9
        newton.options['rtol'] = 1e-9
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['err_on_non_converge'] = True
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1
        newton.options['err_on_non_converge'] = False

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        with self.subTest(check='value'):
            assert_near_equal(
                self.prob['isolated_wing_mass'], 6946.57966315, tol
            )  # 7645.-107.9-682.6=6854.5

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class BWBWingMassGroupTest(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, val=[2], units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            BWBWingMassGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.HIGH_LIFT_MASS, 1068.88854499, units='lbm')
        prob.model.set_input_defaults('c_strut_braced', 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.77335889, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, 75.78, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MATERIAL_FACTOR, 1.19461189, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER, 1.05, units='unitless'
        )
        prob.model.set_input_defaults('c_gear_loc', 0.95, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults('half_sweep', 0.479839474, units='rad')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.FOLDING_AREA, 224.82529025, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.FOLD_MASS_COEFFICIENT, 0.15, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LIFT_COEFFICIENT_RATIO_BODY_TO_WING, 0.35, units='unitless'
        )

        setup_model_options(self.prob, options)
        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        expected_values = {
            Aircraft.Wing.MASS: (7055.90333649, 'lbm'),
            Aircraft.Strut.MASS: (0, 'lbm'),
            Aircraft.Wing.FOLD_MASS: (109.32367334, 'lbm'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
