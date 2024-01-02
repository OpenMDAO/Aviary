import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.wing import (WingMassGroup,
                                                    WingMassSolve,
                                                    WingMassTotal)
from aviary.variable_info.options import get_option_defaults
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.variables import Aircraft, Mission


# this is the large single aisle 1 V3 test case
class WingMassSolveTestCase(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem("wingfuel", WingMassSolve(
            aviary_options=get_option_defaults()), promotes=["*"])

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS, val=3645, units="lbm")
        self.prob.model.set_input_defaults("c_strut_braced", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MATERIAL_FACTOR, val=1.2213063198183813, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units="unitless")
        self.prob.model.set_input_defaults("c_gear_loc", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            "half_sweep", val=0.3947081519145335, units="rad"
        )

        newton = self.prob.model.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-9
        newton.options["rtol"] = 1e-9
        newton.options["iprint"] = 2
        newton.options["maxiter"] = 10
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 10
        newton.options["err_on_non_converge"] = True
        newton.options["reraise_child_analysiserror"] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["bound_enforcement"] = "scalar"
        newton.linesearch.options["iprint"] = -1
        newton.options["err_on_non_converge"] = False

        self.prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob["isolated_wing_mass"], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


# this is the large single aisle 1 V3 test case
class TotalWingMassTestCase1(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "total",
            WingMassTotal(aviary_options=get_option_defaults()),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "isolated_wing_mass", val=15830.0, units="lbm")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 15830.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TotalWingMassTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "total",
            WingMassTotal(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "isolated_wing_mass", val=15830.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=100, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDING_AREA, val=50, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MASS], 17413, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TotalWingMassTestCase3(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "total",
            WingMassTotal(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "isolated_wing_mass", val=15830.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MASS], 23745, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class TotalWingMassTestCase4(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "total", WingMassTotal(aviary_options=options), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            "isolated_wing_mass", val=15830.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=100, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDING_AREA, val=50, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MASS], 25328, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


# this is the large single aisle 1 V3 test case
class WingMassGroupTestCase1(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingMassGroup(aviary_options=get_option_defaults()),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS, val=3645, units="lbm")
        self.prob.model.set_input_defaults("c_strut_braced", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MATERIAL_FACTOR, val=1.2213063198183813, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units="unitless")
        self.prob.model.set_input_defaults("c_gear_loc", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            "half_sweep", val=0.3947081519145335, units="rad"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 15830, tol)
        assert_near_equal(self.prob["isolated_wing_mass"], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    @skipIfMissingXDSM('mass_and_sizing_basic_specs/wing_mass.json')
    def test_io_equip_and_useful_group_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "mass_and_sizing_basic_specs/wing_mass.json")


class WingMassGroupTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingMassGroup(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS, val=3645, units="lbm")
        self.prob.model.set_input_defaults("c_strut_braced", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MATERIAL_FACTOR, val=1.2213063198183813, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units="unitless")
        self.prob.model.set_input_defaults("c_gear_loc", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            "half_sweep", val=0.3947081519145335, units="rad"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=100, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDING_AREA, val=50, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MASS], 17417, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["isolated_wing_mass"], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class WingMassGroupTestCase3(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingMassGroup(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS, val=3645, units="lbm")
        self.prob.model.set_input_defaults("c_strut_braced", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MATERIAL_FACTOR, val=1.2213063198183813, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units="unitless")
        self.prob.model.set_input_defaults("c_gear_loc", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            "half_sweep", val=0.3947081519145335, units="rad"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MASS], 23750, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["isolated_wing_mass"], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class WingMassGroupTestCase4(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group", WingMassGroup(aviary_options=options), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS, val=3645, units="lbm")
        self.prob.model.set_input_defaults("c_strut_braced", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MATERIAL_FACTOR, val=1.2213063198183813, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.POSITION_FACTOR, val=0.98, units="unitless")
        self.prob.model.set_input_defaults("c_gear_loc", val=1.0, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            "half_sweep", val=0.3947081519145335, units="rad"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=100, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDING_AREA, val=50, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.MASS_COEFFICIENT, val=0.5, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MASS], 25333, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["isolated_wing_mass"], 15830, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    @skipIfMissingXDSM('mass_and_sizing_both_specs/wing_mass.json')
    def test_io_equip_and_useful_group_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "mass_and_sizing_both_specs/wing_mass.json")


if __name__ == "__main__":
    unittest.main()
