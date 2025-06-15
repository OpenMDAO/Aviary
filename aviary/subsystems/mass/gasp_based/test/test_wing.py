import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.wing import WingMassGroup, WingMassSolve, WingMassTotal
from aviary.subsystems.mass.gasp_based.wing import BWBWingMassSolve, BWBWingMassGroup
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission
from aviary.utils.aviary_values import AviaryValues


class WingMassSolveTestCase(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wingfuel', WingMassSolve(), promotes=['*'])

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
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
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units='unitless'
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
        assert_near_equal(self.prob['isolated_wing_mass'], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class WingMassSolveTestCase2(unittest.TestCase):
    def setUp(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem('wingfuel', WingMassSolve(), promotes=['*'])
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.HIGH_LIFT_MASS, val=3645, units='lbm')
        prob.model.set_input_defaults('c_strut_braced', val=1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MATERIAL_FACTOR, val=1.2213, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.POSITION_FACTOR, val=0.98, units='unitless')
        prob.model.set_input_defaults('c_gear_loc', val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        prob.model.set_input_defaults('half_sweep', val=0.3947, units='rad')

        newton = prob.model.nonlinear_solver = om.NewtonSolver()
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

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=5e-11, rtol=1e-12)


class TotalWingMassTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'total',
            WingMassTotal(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 15830.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TotalWingMassTestCase2(unittest.TestCase):
    """Has fold and no strut."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'total',
            WingMassTotal(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=100, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDING_AREA, val=50, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 17413, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TotalWingMassTestCase3(unittest.TestCase):
    """Has strut and no fold."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'total',
            WingMassTotal(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units='unitless'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 23745, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TotalWingMassTestCase4(unittest.TestCase):
    """Has fold and strut."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem('total', WingMassTotal(), promotes=['*'])

        self.prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=100, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDING_AREA, val=50, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units='unitless'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 25328, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TotalWingMassTestCase5(unittest.TestCase):
    """
    Test mass-weight conversion
    No fold, no strut.
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'total',
            WingMassTotal(),
            promotes=['*'],
        )
        prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class TotalWingMassTestCase6(unittest.TestCase):
    """
    Test mass-weight conversion
    Has fold and no strut.
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'total',
            WingMassTotal(),
            promotes=['*'],
        )
        self.prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=100, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDING_AREA, val=50, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class TotalWingMassTestCase7(unittest.TestCase):
    """
    Test mass-weight conversion
    Has strut and no fold.
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        prob = om.Problem()
        prob.model.add_subsystem(
            'total',
            WingMassTotal(),
            promotes=['*'],
        )
        prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units='unitless')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class TotalWingMassTestCase8(unittest.TestCase):
    """
    Test mass-weight conversion
    Has fold and strut.
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        prob = om.Problem()
        prob.model.add_subsystem('total', WingMassTotal(), promotes=['*'])
        prob.model.set_input_defaults('isolated_wing_mass', val=15830.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=100, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.FOLDING_AREA, val=50, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units='unitless')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class WingMassGroupTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
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
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units='unitless'
        )
        self.prob.model.set_input_defaults('c_gear_loc', val=1.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults('half_sweep', val=0.3947081519145335, units='rad')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 15830, tol)
        assert_near_equal(self.prob['isolated_wing_mass'], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class WingMassGroupTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
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
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units='unitless'
        )
        self.prob.model.set_input_defaults('c_gear_loc', val=1.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults('half_sweep', val=0.3947081519145335, units='rad')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=100, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDING_AREA, val=50, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 17417, tol)  # not actual GASP value
        assert_near_equal(self.prob['isolated_wing_mass'], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class WingMassGroupTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
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
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units='unitless'
        )
        self.prob.model.set_input_defaults('c_gear_loc', val=1.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults('half_sweep', val=0.3947081519145335, units='rad')
        self.prob.model.set_input_defaults(
            Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units='unitless'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 23750, tol)  # not actual GASP value
        assert_near_equal(self.prob['isolated_wing_mass'], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class WingMassGroupTestCase4(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', WingMassGroup(), promotes=['*'])

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
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
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units='unitless'
        )
        self.prob.model.set_input_defaults('c_gear_loc', val=1.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults('half_sweep', val=0.3947081519145335, units='rad')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=100, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDING_AREA, val=50, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units='unitless'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 25333, tol)  # not actual GASP value
        assert_near_equal(self.prob['isolated_wing_mass'], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class BWBWingMassSolveTestCase(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('wingfuel', BWBWingMassSolve(), promotes=['*'])

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.HIGH_LIFT_MASS, 1068.88854499, units='lbm')
        prob.model.set_input_defaults('c_strut_braced', 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.77335889, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, 75.78, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MATERIAL_FACTOR, 1.19461189, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.POSITION_FACTOR, 1.05, units='unitless')
        prob.model.set_input_defaults('c_gear_loc', 0.95, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults('half_sweep', 0.479839474, units='rad')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LIFT_COEFFICENT_RATIO_BODY_TO_WING, 0.35, units='unitless'
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
        assert_near_equal(
            self.prob['isolated_wing_mass'], 6946.57966315, tol
        )  # 7645.-107.9-682.6=6854.5

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class BWBWingMassGroupTest(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, val=[2], units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            BWBWingMassGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.HIGH_LIFT_MASS, 1068.88854499, units='lbm')
        prob.model.set_input_defaults('c_strut_braced', 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.77335889, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, 75.78, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MATERIAL_FACTOR, 1.19461189, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.POSITION_FACTOR, 1.05, units='unitless')
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
            Aircraft.Fuselage.LIFT_COEFFICENT_RATIO_BODY_TO_WING, 0.35, units='unitless'
        )

        setup_model_options(self.prob, options)
        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 7055.90333649, tol)
        assert_near_equal(self.prob[Aircraft.Strut.MASS], 0, tol)
        assert_near_equal(self.prob[Aircraft.Wing.FOLD_MASS], 109.32367334, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
