import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class WingMassSolve(om.ImplicitComponent):
    """
    Computation of isolated wing mass, namely wing mass including high lift devices
    (but excluding struts and fold effects) using a nonlinear solver.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.HIGH_LIFT_MASS, units='lbm')
        self.add_input(
            'c_strut_braced',
            val=1.00000001,
            units='unitless',
            desc='SKSTR: reduction in bending moment factor for strut braced wing',
        )
        add_aviary_input(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Wing.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.MATERIAL_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Engine.POSITION_FACTOR, shape=num_engine_type)
        self.add_input(
            'c_gear_loc',
            val=1.000000001,
            units='unitless',
            desc='SKGEAR: landing gear location factor',
        )
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless')
        self.add_input(
            'half_sweep', val=0.3947081519, units='rad', desc='SWC2: wing half-chord sweep angle'
        )

        self.add_output(
            'isolated_wing_mass',
            val=17670,
            units='lbm',
            desc='WW: wing mass including high lift devices (but excluding struts and fold effects)',
        )

        self.declare_partials('isolated_wing_mass', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        high_lift_wt = inputs[Aircraft.Wing.HIGH_LIFT_MASS] * GRAV_ENGLISH_LBM
        c_strut_braced = inputs['c_strut_braced']
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        c_wing_mass = inputs[Aircraft.Wing.MASS_COEFFICIENT]
        c_material = inputs[Aircraft.Wing.MATERIAL_FACTOR]
        c_eng_pos = inputs[Aircraft.Engine.POSITION_FACTOR]
        c_gear_loc = inputs['c_gear_loc']
        wingspan = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        half_sweep = inputs['half_sweep']

        isolated_wing_wt = outputs['isolated_wing_mass'] * GRAV_ENGLISH_LBM

        foo = (c_strut_braced * ULF * (gross_wt_initial - 0.8 * isolated_wing_wt)) ** 0.757
        wing_wt_guess = (
            c_wing_mass
            * c_material
            * c_eng_pos
            * c_gear_loc
            * foo
            * wingspan**1.049
            * (1.0 + taper_ratio) ** 0.4
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535) + high_lift_wt

        residuals['isolated_wing_mass'] = (isolated_wing_wt - wing_wt_guess) / GRAV_ENGLISH_LBM

    def linearize(self, inputs, outputs, J):
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        c_strut_braced = inputs['c_strut_braced']
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        c_wing_mass = inputs[Aircraft.Wing.MASS_COEFFICIENT]
        c_material = inputs[Aircraft.Wing.MATERIAL_FACTOR]
        c_eng_pos = inputs[Aircraft.Engine.POSITION_FACTOR]
        c_gear_loc = inputs['c_gear_loc']
        wingspan = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        half_sweep = inputs['half_sweep']

        isolated_wing_wt = outputs['isolated_wing_mass'] * GRAV_ENGLISH_LBM

        foo = (c_strut_braced * ULF * (gross_wt_initial - 0.8 * isolated_wing_wt)) ** 0.757

        J['isolated_wing_mass', Mission.Design.GROSS_MASS] = (
            -(
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (c_strut_braced * ULF * (gross_wt_initial - 0.8 * isolated_wing_wt)) ** (-0.243)
            * c_strut_braced
            * ULF
        )
        J['isolated_wing_mass', Aircraft.Wing.HIGH_LIFT_MASS] = -1
        J['isolated_wing_mass', 'c_strut_braced'] = (
            -(
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (c_strut_braced * ULF * (gross_wt_initial - 0.8 * isolated_wing_wt)) ** (-0.243)
            * ULF
            * (gross_wt_initial - 0.8 * isolated_wing_wt)
            / GRAV_ENGLISH_LBM
        )
        J['isolated_wing_mass', Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = (
            -(
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (c_strut_braced * ULF * (gross_wt_initial - 0.8 * isolated_wing_wt)) ** (-0.243)
            * c_strut_braced
            * (gross_wt_initial - 0.8 * isolated_wing_wt)
            / GRAV_ENGLISH_LBM
        )
        J['isolated_wing_mass', Aircraft.Wing.MASS_COEFFICIENT] = -(
            c_material
            * c_eng_pos
            * c_gear_loc
            * foo
            * wingspan**1.049
            * (1.0 + taper_ratio) ** 0.4
            / GRAV_ENGLISH_LBM
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        J['isolated_wing_mass', Aircraft.Wing.MATERIAL_FACTOR] = -(
            c_wing_mass
            * c_eng_pos
            * c_gear_loc
            * foo
            * wingspan**1.049
            * (1.0 + taper_ratio) ** 0.4
            / GRAV_ENGLISH_LBM
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        J['isolated_wing_mass', Aircraft.Engine.POSITION_FACTOR] = -(
            c_wing_mass
            * c_material
            * c_gear_loc
            * foo
            * wingspan**1.049
            * (1.0 + taper_ratio) ** 0.4
            / GRAV_ENGLISH_LBM
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        J['isolated_wing_mass', 'c_gear_loc'] = -(
            c_wing_mass
            * c_material
            * c_eng_pos
            * foo
            * wingspan**1.049
            * (1.0 + taper_ratio) ** 0.4
            / GRAV_ENGLISH_LBM
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        J['isolated_wing_mass', Aircraft.Wing.SPAN] = (
            -1.049
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan**0.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        )
        J['isolated_wing_mass', Aircraft.Wing.TAPER_RATIO] = (
            -0.4
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan**1.049
                * (1.0 + taper_ratio) ** (-0.6)
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        )
        J['isolated_wing_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
            0.4
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**1.4 * np.cos(half_sweep) ** 1.535)
        )
        J['isolated_wing_mass', 'half_sweep'] = (
            1.535
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 2.535)
            * (-np.sin(half_sweep))
        )
        J['isolated_wing_mass', 'isolated_wing_mass'] = (
            1
            - (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (c_strut_braced * ULF * (gross_wt_initial - 0.8 * isolated_wing_wt)) ** (-0.243)
            * (-0.8)
            * c_strut_braced
            * ULF
        )


class WingMassTotal(om.ExplicitComponent):
    """Computation of wing mass, strut mass, and wing fold mass."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):
        self.add_input(
            'isolated_wing_mass',
            val=1500,
            units='lbm',
            desc='WW: wing mass including high lift devices (but excluding struts and fold effects)',
        )

        if self.options[Aircraft.Wing.HAS_STRUT]:
            add_aviary_input(self, Aircraft.Strut.MASS_COEFFICIENT, units='unitless')

        if self.options[Aircraft.Wing.HAS_FOLD]:
            add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
            add_aviary_input(self, Aircraft.Wing.FOLDING_AREA, units='ft**2')
            add_aviary_input(self, Aircraft.Wing.FOLD_MASS_COEFFICIENT, units='unitless')

        add_aviary_output(self, Aircraft.Wing.MASS, units='lbm')
        add_aviary_output(self, Aircraft.Strut.MASS, units='lbm')
        add_aviary_output(self, Aircraft.Wing.FOLD_MASS, units='lbm')

        self.declare_partials(Aircraft.Wing.MASS, '*')
        if self.options[Aircraft.Wing.HAS_STRUT]:
            self.declare_partials(
                Aircraft.Strut.MASS, [Aircraft.Strut.MASS_COEFFICIENT, 'isolated_wing_mass']
            )
        if self.options[Aircraft.Wing.HAS_FOLD]:
            self.declare_partials(
                Aircraft.Wing.FOLD_MASS,
                [
                    Aircraft.Wing.AREA,
                    Aircraft.Wing.FOLDING_AREA,
                    Aircraft.Wing.FOLD_MASS_COEFFICIENT,
                    'isolated_wing_mass',
                ],
            )

    def compute(self, inputs, outputs):
        isolated_wing_wt = inputs['isolated_wing_mass'] * GRAV_ENGLISH_LBM

        if self.options[Aircraft.Wing.HAS_STRUT]:
            c_strut_mass = inputs[Aircraft.Strut.MASS_COEFFICIENT]

            strut_wt = c_strut_mass * isolated_wing_wt
            outputs[Aircraft.Strut.MASS] = strut_wt / GRAV_ENGLISH_LBM

        else:
            outputs[Aircraft.Strut.MASS] = strut_wt = 0

        if self.options[Aircraft.Wing.HAS_FOLD]:
            wing_area = inputs[Aircraft.Wing.AREA]
            folding_area = inputs[Aircraft.Wing.FOLDING_AREA]
            c_wing_fold = inputs[Aircraft.Wing.FOLD_MASS_COEFFICIENT]

            wt_per_area = isolated_wing_wt / wing_area
            temp_fold_wt = folding_area * wt_per_area
            fold_wt = c_wing_fold * temp_fold_wt
            outputs[Aircraft.Wing.FOLD_MASS] = fold_wt / GRAV_ENGLISH_LBM

        else:
            outputs[Aircraft.Wing.FOLD_MASS] = fold_wt = 0

        total_wing_wt = isolated_wing_wt + strut_wt + fold_wt
        outputs[Aircraft.Wing.MASS] = total_wing_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        isolated_wing_wt = inputs['isolated_wing_mass'] * GRAV_ENGLISH_LBM

        if self.options[Aircraft.Wing.HAS_STRUT]:
            c_strut_mass = inputs[Aircraft.Strut.MASS_COEFFICIENT]

            J[Aircraft.Wing.MASS, Aircraft.Strut.MASS_COEFFICIENT] = J[
                Aircraft.Strut.MASS, Aircraft.Strut.MASS_COEFFICIENT
            ] = isolated_wing_wt / GRAV_ENGLISH_LBM
            J[Aircraft.Wing.MASS, 'isolated_wing_mass'] = 1 + c_strut_mass
            J[Aircraft.Strut.MASS, 'isolated_wing_mass'] = c_strut_mass

        if self.options[Aircraft.Wing.HAS_FOLD]:
            wing_area = inputs[Aircraft.Wing.AREA]
            folding_area = inputs[Aircraft.Wing.FOLDING_AREA]
            c_wing_fold = inputs[Aircraft.Wing.FOLD_MASS_COEFFICIENT]

            wt_per_area = isolated_wing_wt / wing_area
            temp_fold_wt = folding_area * wt_per_area

            J[Aircraft.Wing.MASS, Aircraft.Wing.AREA] = J[
                Aircraft.Wing.FOLD_MASS, Aircraft.Wing.AREA
            ] = -c_wing_fold * folding_area * isolated_wing_wt / wing_area**2 / GRAV_ENGLISH_LBM
            J[Aircraft.Wing.MASS, Aircraft.Wing.FOLDING_AREA] = J[
                Aircraft.Wing.FOLD_MASS, Aircraft.Wing.FOLDING_AREA
            ] = c_wing_fold * wt_per_area / GRAV_ENGLISH_LBM
            J[Aircraft.Wing.MASS, Aircraft.Wing.FOLD_MASS_COEFFICIENT] = J[
                Aircraft.Wing.FOLD_MASS, Aircraft.Wing.FOLD_MASS_COEFFICIENT
            ] = temp_fold_wt / GRAV_ENGLISH_LBM
            J[Aircraft.Wing.MASS, 'isolated_wing_mass'] = 1 + c_wing_fold * folding_area / wing_area
            J[Aircraft.Wing.FOLD_MASS, 'isolated_wing_mass'] = (
                c_wing_fold * folding_area / wing_area
            )

        if self.options[Aircraft.Wing.HAS_FOLD] and self.options[Aircraft.Wing.HAS_STRUT]:
            J[Aircraft.Wing.MASS, 'isolated_wing_mass'] = (
                1 + c_wing_fold * folding_area / wing_area + c_strut_mass
            )

        if (
            self.options[Aircraft.Wing.HAS_STRUT] is False
            and self.options[Aircraft.Wing.HAS_FOLD] is False
        ):
            J[Aircraft.Wing.MASS, 'isolated_wing_mass'] = 1


class BWBWingMassSolve(om.ImplicitComponent):
    """
    Computation of isolated wing mass, namely wing mass including high lift devices
    (but excluding struts and fold effects) using a nonlinear solver.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.HIGH_LIFT_MASS, units='lbm')
        self.add_input(
            'c_strut_braced',
            val=1.00000001,
            units='unitless',
            desc='SKSTR: reduction in bending moment factor for strut braced wing',
        )
        add_aviary_input(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Wing.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.MATERIAL_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Engine.POSITION_FACTOR, shape=num_engine_type)
        self.add_input(
            'c_gear_loc',
            val=1.000000001,
            units='unitless',
            desc='SKGEAR: landing gear location factor',
        )
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless')
        self.add_input(
            'half_sweep', val=0.3947081519, units='rad', desc='SWC2: wing half-chord sweep angle'
        )
        add_aviary_input(
            self, Aircraft.Fuselage.LIFT_COEFFICENT_RATIO_BODY_TO_WING, units='unitless'
        )

        self.add_output(
            'isolated_wing_mass',
            val=12973.96759,
            units='lbm',
            desc='WW: wing mass including high lift devices (but excluding struts and fold effects)',
        )

        self.declare_partials('isolated_wing_mass', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        high_lift_wt = inputs[Aircraft.Wing.HIGH_LIFT_MASS] * GRAV_ENGLISH_LBM
        c_strut_braced = inputs['c_strut_braced']
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        c_wing_mass = inputs[Aircraft.Wing.MASS_COEFFICIENT]
        c_material = inputs[Aircraft.Wing.MATERIAL_FACTOR]
        c_eng_pos = inputs[Aircraft.Engine.POSITION_FACTOR]
        c_gear_loc = inputs['c_gear_loc']
        wingspan = inputs[Aircraft.Wing.SPAN]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        half_sweep = inputs['half_sweep']
        CLBqCLW = inputs[Aircraft.Fuselage.LIFT_COEFFICENT_RATIO_BODY_TO_WING]

        isolated_wing_wt = outputs['isolated_wing_mass'] * GRAV_ENGLISH_LBM

        foo_numer = c_strut_braced * ULF * (gross_wt_initial - 0.8 * isolated_wing_wt)
        foo_denom = 1.0 + CLBqCLW
        foo = (foo_numer / foo_denom) ** 0.757
        wingspan_mod = wingspan - cabin_width
        wing_wt_guess = (
            c_wing_mass
            * c_material
            * c_eng_pos
            * c_gear_loc
            * foo
            * wingspan_mod**1.049
            * (1.0 + taper_ratio) ** 0.4
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535) + high_lift_wt

        residuals['isolated_wing_mass'] = (isolated_wing_wt - wing_wt_guess) / GRAV_ENGLISH_LBM

    def linearize(self, inputs, outputs, J):
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        high_lift_wt = inputs[Aircraft.Wing.HIGH_LIFT_MASS] * GRAV_ENGLISH_LBM
        c_strut_braced = inputs['c_strut_braced']
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        c_wing_mass = inputs[Aircraft.Wing.MASS_COEFFICIENT]
        c_material = inputs[Aircraft.Wing.MATERIAL_FACTOR]
        c_eng_pos = inputs[Aircraft.Engine.POSITION_FACTOR]
        c_gear_loc = inputs['c_gear_loc']
        wingspan = inputs[Aircraft.Wing.SPAN]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        half_sweep = inputs['half_sweep']
        CLBqCLW = inputs[Aircraft.Fuselage.LIFT_COEFFICENT_RATIO_BODY_TO_WING]

        isolated_wing_wt = outputs['isolated_wing_mass'] * GRAV_ENGLISH_LBM

        foo_numer = c_strut_braced * ULF * (gross_wt_initial - 0.8 * isolated_wing_wt)
        foo_denom = 1.0 + CLBqCLW
        foo = (foo_numer / foo_denom) ** 0.757
        wingspan_mod = wingspan - cabin_width  # modification for BWB
        wing_wt_guess = (
            c_wing_mass
            * c_material
            * c_eng_pos
            * c_gear_loc
            * foo
            * wingspan_mod**1.049
            * (1.0 + taper_ratio) ** 0.4
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535) + high_lift_wt

        J['isolated_wing_mass', Mission.Design.GROSS_MASS] = (
            -(
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan_mod**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (foo_numer / foo_denom) ** (-0.243)
            * c_strut_braced
            / foo_denom
            * ULF
        )
        J['isolated_wing_mass', Aircraft.Wing.HIGH_LIFT_MASS] = -1
        J['isolated_wing_mass', 'c_strut_braced'] = (
            -(
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan_mod**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (foo_numer / foo_denom) ** (-0.243)
            * ULF
            * (gross_wt_initial - 0.8 * isolated_wing_wt)
            / foo_denom
            / GRAV_ENGLISH_LBM
        )
        J['isolated_wing_mass', Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = (
            -(
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan_mod**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (foo_numer / foo_denom) ** (-0.243)
            * c_strut_braced
            * (gross_wt_initial - 0.8 * isolated_wing_wt)
            / foo_denom
            / GRAV_ENGLISH_LBM
        )
        J['isolated_wing_mass', Aircraft.Fuselage.LIFT_COEFFICENT_RATIO_BODY_TO_WING] = (
            (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan_mod**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (foo_numer / foo_denom) ** (-0.243)
            * foo_numer
            / foo_denom**2
            / GRAV_ENGLISH_LBM
        )
        J['isolated_wing_mass', Aircraft.Wing.MASS_COEFFICIENT] = -(
            c_material
            * c_eng_pos
            * c_gear_loc
            * foo
            * wingspan_mod**1.049
            * (1.0 + taper_ratio) ** 0.4
            / GRAV_ENGLISH_LBM
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        J['isolated_wing_mass', Aircraft.Wing.MATERIAL_FACTOR] = -(
            c_wing_mass
            * c_eng_pos
            * c_gear_loc
            * foo
            * wingspan_mod**1.049
            * (1.0 + taper_ratio) ** 0.4
            / GRAV_ENGLISH_LBM
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        J['isolated_wing_mass', Aircraft.Engine.POSITION_FACTOR] = -(
            c_wing_mass
            * c_material
            * c_gear_loc
            * foo
            * wingspan_mod**1.049
            * (1.0 + taper_ratio) ** 0.4
            / GRAV_ENGLISH_LBM
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        J['isolated_wing_mass', 'c_gear_loc'] = -(
            c_wing_mass
            * c_material
            * c_eng_pos
            * foo
            * wingspan_mod**1.049
            * (1.0 + taper_ratio) ** 0.4
            / GRAV_ENGLISH_LBM
        ) / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        J['isolated_wing_mass', Aircraft.Wing.SPAN] = (
            -1.049
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan_mod**0.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        )
        J['isolated_wing_mass', Aircraft.Fuselage.AVG_DIAMETER] = (
            1.049
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan_mod**0.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        )
        J['isolated_wing_mass', Aircraft.Wing.TAPER_RATIO] = (
            -0.4
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan_mod**1.049
                * (1.0 + taper_ratio) ** (-0.6)
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
        )
        J['isolated_wing_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
            0.4
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan_mod**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**1.4 * np.cos(half_sweep) ** 1.535)
        )
        J['isolated_wing_mass', 'half_sweep'] = (
            1.535
            * (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * foo
                * wingspan_mod**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / GRAV_ENGLISH_LBM
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 2.535)
            * (-np.sin(half_sweep))
        )
        J['isolated_wing_mass', 'isolated_wing_mass'] = (
            1
            - (
                c_wing_mass
                * c_material
                * c_eng_pos
                * c_gear_loc
                * wingspan_mod**1.049
                * (1.0 + taper_ratio) ** 0.4
            )
            / (100000.0 * tc_ratio_root**0.4 * np.cos(half_sweep) ** 1.535)
            * 0.757
            * (foo_numer / foo_denom) ** (-0.243)
            * (-0.8)
            * c_strut_braced
            / foo_denom
            * ULF
        )


class WingMassGroup(om.Group):
    """Group to compute wing mass for GASP-based mass."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):
        # variables that are calculated at a higher level
        higher_level_inputs_isolated = [
            'c_strut_braced',
            'c_gear_loc',
            'half_sweep',
        ]
        if self.options[Aircraft.Wing.HAS_FOLD] or self.options[Aircraft.Wing.HAS_STRUT]:
            higher_level_inputs_total = ['aircraft:*']
        else:
            higher_level_inputs_total = []

        # variables that are passed within the group but not used at a higher level
        connected_outputs_isolated = [
            'isolated_wing_mass',
        ]
        connected_inputs_total = [
            'isolated_wing_mass',
        ]

        isolated_mass = self.add_subsystem(
            'isolated_mass',
            WingMassSolve(),
            promotes_inputs=higher_level_inputs_isolated + ['aircraft:*', 'mission:*'],
            promotes_outputs=connected_outputs_isolated,
        )

        self.add_subsystem(
            'total_mass',
            WingMassTotal(),
            promotes_inputs=connected_inputs_total + higher_level_inputs_total,
            promotes_outputs=['aircraft:*'],
        )

        newton = isolated_mass.nonlinear_solver = om.NewtonSolver()

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

        isolated_mass.linear_solver = om.DirectSolver(assemble_jac=True)


class BWBWingMassGroup(om.Group):
    """
    Group to compute wing mass for GASP-based mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):
        # variables that are calculated at a higher level
        higher_level_inputs_isolated = [
            'c_strut_braced',
            'c_gear_loc',
            'half_sweep',
        ]
        if self.options[Aircraft.Wing.HAS_FOLD] or self.options[Aircraft.Wing.HAS_STRUT]:
            higher_level_inputs_total = ['aircraft:*']
        else:
            higher_level_inputs_total = []

        # variables that are passed within the group but not used at a higher level
        connected_outputs_isolated = [
            'isolated_wing_mass',
        ]
        connected_inputs_total = [
            'isolated_wing_mass',
        ]

        isolated_mass = self.add_subsystem(
            'isolated_mass',
            BWBWingMassSolve(),
            promotes_inputs=higher_level_inputs_isolated + ['aircraft:*', 'mission:*'],
            promotes_outputs=connected_outputs_isolated,
        )

        self.add_subsystem(
            'total_mass',
            WingMassTotal(),
            promotes_inputs=connected_inputs_total + higher_level_inputs_total,
            promotes_outputs=['aircraft:*'],
        )

        newton = isolated_mass.nonlinear_solver = om.NewtonSolver()

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

        isolated_mass.linear_solver = om.DirectSolver(assemble_jac=True)
