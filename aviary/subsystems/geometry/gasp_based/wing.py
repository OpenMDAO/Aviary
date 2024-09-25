import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.geometry.gasp_based.non_dimensional_conversion import \
    DimensionalNonDimensionalInterchange
from aviary.subsystems.geometry.gasp_based.strut import StrutGeom
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.conflict_checks import check_fold_location_definition
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class WingSize(om.ExplicitComponent):
    """
    Computation of wing area and wing span for GASP-based aerodynamics.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        add_aviary_input(self, Mission.Design.GROSS_MASS, val=152000)
        add_aviary_input(self, Aircraft.Wing.LOADING, val=128)
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=10.13)

        add_aviary_output(self, Aircraft.Wing.AREA, val=0)
        add_aviary_output(self, Aircraft.Wing.SPAN, val=0)

        self.declare_partials(
            Aircraft.Wing.AREA, [Mission.Design.GROSS_MASS, Aircraft.Wing.LOADING]
        )
        self.declare_partials(
            Aircraft.Wing.SPAN,
            [
                Aircraft.Wing.ASPECT_RATIO,
                Mission.Design.GROSS_MASS,
                Aircraft.Wing.LOADING,
            ],
        )

    def compute(self, inputs, outputs):

        gross_mass_initial = inputs[Mission.Design.GROSS_MASS]
        wing_loading = inputs[Aircraft.Wing.LOADING]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]

        wing_area = gross_mass_initial * GRAV_ENGLISH_LBM / wing_loading
        wingspan = (AR * wing_area) ** 0.5

        outputs[Aircraft.Wing.AREA] = wing_area
        outputs[Aircraft.Wing.SPAN] = wingspan

    def compute_partials(self, inputs, J):

        gross_mass_initial = inputs[Mission.Design.GROSS_MASS]
        wing_loading = inputs[Aircraft.Wing.LOADING]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]

        wing_area = gross_mass_initial * GRAV_ENGLISH_LBM / wing_loading

        J[Aircraft.Wing.AREA,
            Mission.Design.GROSS_MASS] = dWA_dGMT = GRAV_ENGLISH_LBM / wing_loading
        J[Aircraft.Wing.AREA, Aircraft.Wing.LOADING] = dWA_dWL = (
            -gross_mass_initial * GRAV_ENGLISH_LBM / wing_loading**2
        )

        J[Aircraft.Wing.SPAN, Aircraft.Wing.ASPECT_RATIO] = (
            0.5 * wing_area**0.5 * AR ** (-0.5)
        )
        J[Aircraft.Wing.SPAN, Mission.Design.GROSS_MASS] = (
            0.5 * AR**0.5 * wing_area ** (-0.5) * dWA_dGMT
        )
        J[Aircraft.Wing.SPAN, Aircraft.Wing.LOADING] = (
            0.5 * AR**0.5 * wing_area ** (-0.5) * dWA_dWL
        )


class WingParameters(om.ExplicitComponent):
    """
    Computation of various wing parameters for GASP-based geometry.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        add_aviary_input(self, Aircraft.Wing.AREA, val=2)
        add_aviary_input(self, Aircraft.Wing.SPAN, val=2)
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=10.13)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, val=0.33)
        add_aviary_input(self, Aircraft.Wing.SWEEP, val=25)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11)
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, val=10)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP, val=0.1)

        if not self.options["aviary_options"].get_val(Aircraft.Wing.HAS_FOLD, units='unitless'):

            add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6)
            add_aviary_output(self, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=0)

            self.declare_partials(
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
                [
                    Aircraft.Fuel.WING_FUEL_FRACTION,
                    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                    Aircraft.Fuselage.AVG_DIAMETER,
                    Aircraft.Wing.SPAN,
                    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                    Aircraft.Wing.AREA,
                    Aircraft.Wing.TAPER_RATIO,
                    Aircraft.Wing.ASPECT_RATIO,
                ],
            )

        add_aviary_output(self, Aircraft.Wing.CENTER_CHORD, val=0)
        add_aviary_output(self, Aircraft.Wing.AVERAGE_CHORD, val=0)
        add_aviary_output(self, Aircraft.Wing.ROOT_CHORD, val=0)
        add_aviary_output(self, Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, val=0)
        add_aviary_output(self, Aircraft.Wing.LEADING_EDGE_SWEEP,
                          val=0.4763948, units="rad")

        self.declare_partials(
            Aircraft.Wing.CENTER_CHORD,
            [Aircraft.Wing.AREA, Aircraft.Wing.SPAN, Aircraft.Wing.TAPER_RATIO],
        )
        self.declare_partials(
            Aircraft.Wing.AVERAGE_CHORD,
            [Aircraft.Wing.AREA, Aircraft.Wing.SPAN, Aircraft.Wing.TAPER_RATIO],
        )
        self.declare_partials(
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            [
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                Aircraft.Wing.TAPER_RATIO,
            ],
        )
        self.declare_partials(
            Aircraft.Wing.ROOT_CHORD,
            [
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
            ],
        )
        self.declare_partials(
            Aircraft.Wing.LEADING_EDGE_SWEEP,
            [
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
            ],
        )

    def compute(self, inputs, outputs):

        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        sweep_c4 = inputs[Aircraft.Wing.SWEEP]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]

        center_chord = 2.0 * wing_area / wingspan / (1.0 + taper_ratio)
        avg_chord = (2.0 * center_chord / 3.0) * (
            (1.0 + taper_ratio) - (taper_ratio / (1.0 + taper_ratio))
        )
        tan_sweep_LE = (1.0 - taper_ratio) / (1.0 + taper_ratio) / AR + np.tan(
            sweep_c4 * np.pi / 180.0
        )
        outputs[Aircraft.Wing.LEADING_EDGE_SWEEP] = np.arctan(tan_sweep_LE)
        tan_sweep_TE = 3.0 * (taper_ratio - 1.0) / (1.0 + taper_ratio) / AR + np.tan(
            sweep_c4 * (np.pi / 180)
        )
        sweep_TE = np.arctan(tan_sweep_TE)
        FHP = (
            2.0
            * (
                tc_ratio_root
                * center_chord
                * (cabin_width - (tc_ratio_root * center_chord))
            )
            ** 0.5
            + 0.4
        )
        HP = FHP * tan_sweep_LE / 2.0
        root_chord = center_chord - HP + FHP * tan_sweep_TE / 2.0
        tc_ratio_avg = (
            (tc_ratio_root - cabin_width / wingspan * (tc_ratio_root - tc_ratio_tip))
            * (1.0 - cabin_width / wingspan * (1.0 - taper_ratio))
            + taper_ratio * tc_ratio_tip
        ) / (1.0 + taper_ratio - cabin_width / wingspan * (1.0 - taper_ratio))

        outputs[Aircraft.Wing.CENTER_CHORD] = center_chord
        outputs[Aircraft.Wing.AVERAGE_CHORD] = avg_chord
        outputs[Aircraft.Wing.ROOT_CHORD] = root_chord
        outputs[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED] = tc_ratio_avg

        if not self.options["aviary_options"].get_val(Aircraft.Wing.HAS_FOLD, units='unitless'):
            fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]

            geometric_fuel_vol = (
                fuel_vol_frac
                * 0.888889
                * tc_ratio_avg
                * (wing_area**1.5)
                * (2.0 * taper_ratio + 1.0)
            ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
            outputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = geometric_fuel_vol

    def compute_partials(self, inputs, J):

        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        sweep_c4 = inputs[Aircraft.Wing.SWEEP]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]

        center_chord = 2.0 * wing_area / wingspan / (1.0 + taper_ratio)
        FHP = (
            2.0
            * (
                tc_ratio_root
                * center_chord
                * (cabin_width - (tc_ratio_root * center_chord))
            )
            ** 0.5
            + 0.4
        )

        tan_sweep_LE = (1.0 - taper_ratio) / (1.0 + taper_ratio) / AR + np.tan(
            sweep_c4 * (np.pi / 180)
        )
        tan_sweep_TE = 3.0 * (taper_ratio - 1.0) / (1.0 + taper_ratio) / AR + np.tan(
            sweep_c4 * (np.pi / 180)
        )

        dCenterChord_dWingArea = 2 / (wingspan * (1 + taper_ratio))
        dCenterChord_dWingspan = -2 * wing_area / ((1 + taper_ratio) * wingspan**2)
        dCenterChord_dTaperRatio = -2 * wing_area / (wingspan * (1 + taper_ratio) ** 2)
        dFHP_dTcRatioRoot = (
            tc_ratio_root * center_chord * (cabin_width - tc_ratio_root * center_chord)
        ) ** (-0.5) * (
            center_chord * cabin_width - 2 * tc_ratio_root * center_chord**2
        )
        dFHP_dCabinWidth = (
            (
                tc_ratio_root
                * center_chord
                * (cabin_width - tc_ratio_root * center_chord)
            )
            ** (-0.5)
            * tc_ratio_root
            * center_chord
        )
        dFHP_dWingArea = (
            (
                tc_ratio_root
                * center_chord
                * (cabin_width - tc_ratio_root * center_chord)
            )
            ** (-0.5)
            * (tc_ratio_root * cabin_width - 2 * tc_ratio_root**2 * center_chord)
            * dCenterChord_dWingArea
        )
        dFHP_dWingspan = (
            (
                tc_ratio_root
                * center_chord
                * (cabin_width - tc_ratio_root * center_chord)
            )
            ** (-0.5)
            * (tc_ratio_root * cabin_width - 2 * tc_ratio_root**2 * center_chord)
            * dCenterChord_dWingspan
        )
        dFHP_dTaperRatio = (
            (
                tc_ratio_root
                * center_chord
                * (cabin_width - tc_ratio_root * center_chord)
            )
            ** (-0.5)
            * (tc_ratio_root * cabin_width - 2 * tc_ratio_root**2 * center_chord)
            * dCenterChord_dTaperRatio
        )

        dTanSweepLE_dTaperRatio = (-(1 + taper_ratio) * AR - (1 - taper_ratio) * AR) / (
            (1 + taper_ratio) ** 2 * AR**2
        )
        dTanSweepLE_dAR = -(1 - taper_ratio) / ((1 + taper_ratio) * AR**2)
        dTanSweepLE_dSweepC4 = dTanSweepTE_dSweepC4 = (
            (np.pi / 180) * 1 / np.cos(sweep_c4 * (np.pi / 180)) ** 2
        )
        dTanSweepTE_dTaperRatio = (
            3
            * ((1 + taper_ratio) * AR - (taper_ratio - 1) * AR)
            / ((1 + taper_ratio) ** 2 * AR**2)
        )
        dTanSweepTE_dAR = -3 * (taper_ratio - 1) / ((1 + taper_ratio) * AR**2)

        dRootChord_dWingArea = dCenterChord_dWingArea + dFHP_dWingArea * (
            -tan_sweep_LE / 2 + tan_sweep_TE / 2
        )
        dRootChord_dWingspan = dCenterChord_dWingspan + dFHP_dWingspan * (
            -tan_sweep_LE / 2 + tan_sweep_TE / 2
        )
        dRootChord_dTaperRatio = (
            dCenterChord_dTaperRatio
            + dFHP_dTaperRatio * (-tan_sweep_LE / 2 + tan_sweep_TE / 2)
            + FHP * (-dTanSweepLE_dTaperRatio / 2 + dTanSweepTE_dTaperRatio / 2)
        )
        dRootChord_dTcRatioRoot = dFHP_dTcRatioRoot * (
            -tan_sweep_LE / 2 + tan_sweep_TE / 2
        )
        dRootChord_dCabinWidth = dFHP_dCabinWidth * (
            -tan_sweep_LE / 2 + tan_sweep_TE / 2
        )
        dRootChord_dAR = FHP * (-dTanSweepLE_dAR / 2 + dTanSweepTE_dAR / 2)
        dRootChord_dSweepC4 = FHP * 0.5 * (-dTanSweepLE_dSweepC4 + dTanSweepTE_dSweepC4)

        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.AREA] = dRootChord_dWingArea
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.SPAN] = dRootChord_dWingspan
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.TAPER_RATIO] = dRootChord_dTaperRatio
        J[
            Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT
        ] = dRootChord_dTcRatioRoot
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Fuselage.AVG_DIAMETER] = dRootChord_dCabinWidth
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.ASPECT_RATIO] = dRootChord_dAR
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.SWEEP] = dRootChord_dSweepC4

        J[Aircraft.Wing.CENTER_CHORD, Aircraft.Wing.AREA] = dCenterChord_dWingArea
        J[Aircraft.Wing.CENTER_CHORD, Aircraft.Wing.SPAN] = dCenterChord_dWingspan
        J[
            Aircraft.Wing.CENTER_CHORD, Aircraft.Wing.TAPER_RATIO
        ] = dCenterChord_dTaperRatio

        J[Aircraft.Wing.AVERAGE_CHORD, Aircraft.Wing.AREA] = (
            (2.0 / 3.0)
            * ((1.0 + taper_ratio) - (taper_ratio / (1.0 + taper_ratio)))
            * dCenterChord_dWingArea
        )
        J[Aircraft.Wing.AVERAGE_CHORD, Aircraft.Wing.SPAN] = (
            (2.0 / 3.0)
            * ((1.0 + taper_ratio) - (taper_ratio / (1.0 + taper_ratio)))
            * dCenterChord_dWingspan
        )
        J[
            Aircraft.Wing.AVERAGE_CHORD, Aircraft.Wing.TAPER_RATIO
        ] = dCenterChord_dTaperRatio * (2.0 / 3.0) * (
            (1.0 + taper_ratio) - (taper_ratio / (1.0 + taper_ratio))
        ) + (
            2.0 * center_chord / 3.0
        ) * (
            1 - ((1 + taper_ratio) - taper_ratio) / (1.0 + taper_ratio) ** 2
        )

        tc_ratio_avg = (
            (tc_ratio_root - cabin_width / wingspan * (tc_ratio_root - tc_ratio_tip))
            * (1.0 - cabin_width / wingspan * (1.0 - taper_ratio))
            + taper_ratio * tc_ratio_tip
        ) / (1.0 + taper_ratio - cabin_width / wingspan * (1.0 - taper_ratio))
        a = tc_ratio_root - cabin_width / wingspan * (tc_ratio_root - tc_ratio_tip)
        b = 1.0 - cabin_width / wingspan * (1.0 - taper_ratio)
        c = taper_ratio * tc_ratio_tip
        d = 1.0 + taper_ratio - cabin_width / wingspan * (1.0 - taper_ratio)

        dAB_dCabW = (
            a * (taper_ratio - 1) / wingspan
            + b * (tc_ratio_tip - tc_ratio_root) / wingspan
        )
        dD_dCabW = (taper_ratio - 1) / wingspan
        dAB_dWingspan = (
            a * cabin_width * (1 - taper_ratio) / wingspan**2
            + b * cabin_width * (tc_ratio_root - tc_ratio_tip) / wingspan**2
        )
        dD_dWingspan = cabin_width * (1 - taper_ratio) / wingspan**2
        dABC_dTR = a * cabin_width / wingspan + tc_ratio_tip
        dD_dTR = 1 + cabin_width / wingspan

        J[
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
        ] = dTCA_dTCR = (
            (1 - cabin_width / wingspan) * b / d
        )
        J[
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, Aircraft.Fuselage.AVG_DIAMETER
        ] = dTCA_dCabW = (d * dAB_dCabW - (a * b + c) * dD_dCabW) / d**2
        J[
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, Aircraft.Wing.SPAN
        ] = dTCA_dWingspan = (d * dAB_dWingspan - (a * b + c) * dD_dWingspan) / d**2
        J[
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
        ] = dTCA_dTCT = (cabin_width / wingspan * b + taper_ratio) / d
        J[
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, Aircraft.Wing.TAPER_RATIO
        ] = dTCA_dTR = (d * dABC_dTR - (a * b + c) * dD_dTR) / d**2

        trp1 = taper_ratio + 1
        swprad = np.pi * sweep_c4 / 180.0
        tswprad = np.tan(swprad)
        denom = AR**2 * trp1**2 + (AR * trp1 * tswprad - taper_ratio + 1) ** 2
        J[Aircraft.Wing.LEADING_EDGE_SWEEP, Aircraft.Wing.TAPER_RATIO] = -2 * AR / denom
        J[Aircraft.Wing.LEADING_EDGE_SWEEP, Aircraft.Wing.ASPECT_RATIO] = (
            (taper_ratio - 1) * trp1 / denom
        )
        J[Aircraft.Wing.LEADING_EDGE_SWEEP, Aircraft.Wing.SWEEP] = (
            np.pi * AR**2 * trp1**2 / denom / 180 / np.cos(swprad) ** 2
        )

        if not self.options["aviary_options"].get_val(Aircraft.Wing.HAS_FOLD, units='unitless'):
            fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
            geometric_fuel_vol = (
                fuel_vol_frac
                * 0.888889
                * tc_ratio_avg
                * (wing_area**1.5)
                * (2.0 * taper_ratio + 1.0)
            ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
            num = (
                fuel_vol_frac
                * 0.888889
                * tc_ratio_avg
                * (wing_area**1.5)
                * (2.0 * taper_ratio + 1.0)
            )
            den = (AR**0.5) * ((taper_ratio + 1.0) ** 2.0)
            dNum_dTR = (
                fuel_vol_frac
                * 0.888889
                * dTCA_dTR
                * (wing_area**1.5)
                * (2.0 * taper_ratio + 1.0)
                + fuel_vol_frac * 0.888889 * tc_ratio_avg * (wing_area**1.5) * 2
            )
            dDen_dTR = 2 * (AR**0.5) * (taper_ratio + 1.0)

            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Fuel.WING_FUEL_FRACTION] = (
                0.888889 * tc_ratio_avg * (wing_area**1.5) * (2.0 * taper_ratio + 1.0)
            ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.AREA] = (
                1.5
                * (
                    fuel_vol_frac
                    * 0.888889
                    * tc_ratio_avg
                    * (wing_area**0.5)
                    * (2.0 * taper_ratio + 1.0)
                )
                / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.TAPER_RATIO] = (
                den * dNum_dTR - num * dDen_dTR
            ) / den**2
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.ASPECT_RATIO] = (
                -0.5
                * (
                    fuel_vol_frac
                    * 0.888889
                    * tc_ratio_avg
                    * (wing_area**1.5)
                    * (2.0 * taper_ratio + 1.0)
                )
                / ((AR**1.5) * (taper_ratio + 1.0) ** 2.0)
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
                fuel_vol_frac
                * 0.888889
                * dTCA_dTCR
                * (wing_area**1.5)
                * (2.0 * taper_ratio + 1.0)
            ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Fuselage.AVG_DIAMETER] = (
                fuel_vol_frac
                * 0.888889
                * dTCA_dCabW
                * (wing_area**1.5)
                * (2.0 * taper_ratio + 1.0)
            ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.SPAN] = (
                fuel_vol_frac
                * 0.888889
                * dTCA_dWingspan
                * (wing_area**1.5)
                * (2.0 * taper_ratio + 1.0)
            ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = (
                fuel_vol_frac
                * 0.888889
                * dTCA_dTCT
                * (wing_area**1.5)
                * (2.0 * taper_ratio + 1.0)
            ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))


class WingFold(om.ExplicitComponent):
    """
    Computation of taper ratio between wing root and fold location, wing area of
    part of wings that does not fold, mean value of thickess to chord ratio between
    root and fold, aspect ratio of non-folding part of wing, wing tank fuel volume.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        if not self.options["aviary_options"].get_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, units='unitless'):
            self.add_input(
                "strut_y",
                val=25,
                units="ft",
                desc="YSTRUT: attachment location of strut",
            )

            self.declare_partials("nonfolded_taper_ratio", "strut_y")
            self.declare_partials(Aircraft.Wing.FOLDING_AREA, "strut_y")
            self.declare_partials("nonfolded_wing_area", "strut_y")
            self.declare_partials("tc_ratio_mean_folded", "strut_y")
            self.declare_partials("nonfolded_AR", "strut_y")
            self.declare_partials(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, "strut_y")

        else:
            self.declare_partials("nonfolded_taper_ratio", Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials(Aircraft.Wing.FOLDING_AREA, Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials("nonfolded_wing_area", Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials("tc_ratio_mean_folded", Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials("nonfolded_AR", Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
                                  Aircraft.Wing.FOLDED_SPAN)

            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN, val=25, units='ft')

        add_aviary_input(self, Aircraft.Wing.AREA, val=200)
        add_aviary_input(self, Aircraft.Wing.SPAN, val=118)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, val=0.33)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP, val=0.1)
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6)

        self.add_output(
            "nonfolded_taper_ratio",
            val=0.1,
            units="unitless",
            desc="SLM_NF: taper ratio between wing root and fold location",
        )

        add_aviary_output(self, Aircraft.Wing.FOLDING_AREA, val=50)

        self.add_output(
            "nonfolded_wing_area",
            val=150,
            units="ft**2",
            desc="SW_NF: wing area of part of wings that does not fold",
        )
        self.add_output(
            "tc_ratio_mean_folded",
            val=0.12,
            units="unitless",
            desc="TCM: mean value of thickess to chord ratio between root and fold",
        )
        self.add_output(
            "nonfolded_AR",
            val=10,
            units="unitless",
            desc="AR_NF: aspect ratio of non-folding part of wing",
        )

        add_aviary_output(self, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=0)

        self.declare_partials(
            "nonfolded_taper_ratio",
            [Aircraft.Wing.AREA, Aircraft.Wing.SPAN, Aircraft.Wing.TAPER_RATIO],
        )
        self.declare_partials(
            Aircraft.Wing.FOLDING_AREA,
            [Aircraft.Wing.SPAN, Aircraft.Wing.AREA, Aircraft.Wing.TAPER_RATIO],
        )
        self.declare_partials(
            "nonfolded_wing_area",
            [Aircraft.Wing.AREA, Aircraft.Wing.SPAN, Aircraft.Wing.TAPER_RATIO],
        )
        self.declare_partials(
            "tc_ratio_mean_folded",
            [
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                Aircraft.Wing.SPAN,
            ],
        )
        self.declare_partials(
            "nonfolded_AR",
            [Aircraft.Wing.AREA, Aircraft.Wing.SPAN, Aircraft.Wing.TAPER_RATIO],
        )
        self.declare_partials(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
            [
                Aircraft.Fuel.WING_FUEL_FRACTION,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.AREA,
                Aircraft.Wing.TAPER_RATIO,
            ],
        )

    def compute(self, inputs, outputs):

        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]

        if not self.options["aviary_options"].get_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, units='unitless'):

            strut_y = inputs["strut_y"]
            location = strut_y

        else:
            fold_y = inputs[Aircraft.Wing.FOLDED_SPAN]
            location = fold_y / 2.0

        root_chord_wing = 2 * wing_area / (wingspan * (1 + taper_ratio))
        tip_chord = taper_ratio * root_chord_wing
        fold_chord = root_chord_wing + location * (tip_chord - root_chord_wing) / (
            wingspan / 2.0
        )
        nonfolded_taper_ratio = fold_chord / root_chord_wing
        folding_area = (wingspan / 2.0 - location) * (fold_chord + tip_chord)

        nonfolded_wing_area = wing_area - folding_area

        if (wingspan / 2.0) < location:
            raise ValueError(
                "Error: The wingspan provided is less than the wingspan of the wing fold."
            )

        tc_ratio_fold = tc_ratio_root + location * (tc_ratio_tip - tc_ratio_root) / (
            wingspan / 2.0
        )
        tc_ratio_mean_folded = 0.5 * (tc_ratio_root + tc_ratio_fold)
        nonfolded_AR = 4.0 * location**2 / nonfolded_wing_area
        geometric_fuel_vol = (
            fuel_vol_frac
            * 0.888889
            * tc_ratio_mean_folded
            * (nonfolded_wing_area**1.5)
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) / ((nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0))

        outputs["nonfolded_taper_ratio"] = nonfolded_taper_ratio
        outputs[Aircraft.Wing.FOLDING_AREA] = folding_area
        outputs["nonfolded_wing_area"] = nonfolded_wing_area
        outputs["tc_ratio_mean_folded"] = tc_ratio_mean_folded
        outputs["nonfolded_AR"] = nonfolded_AR
        outputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = geometric_fuel_vol

    def compute_partials(self, inputs, J):

        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]

        if not self.options["aviary_options"].get_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, units='unitless'):

            strut_y = inputs["strut_y"]
            location = strut_y
            dLoc_dWingspan = 0
            dLoc_dy = 1
            wrt = "strut_y"

        else:
            fold_y = inputs[Aircraft.Wing.FOLDED_SPAN]
            wrt = Aircraft.Wing.FOLDED_SPAN

            location = fold_y / 2.0
            dLoc_dWingspan = 0
            dLoc_dy = 1 / 2

        root_chord_wing = 2 * wing_area / (wingspan * (1 + taper_ratio))
        tip_chord = taper_ratio * root_chord_wing
        fold_chord = root_chord_wing + location * (tip_chord - root_chord_wing) / (
            wingspan / 2.0
        )
        nonfolded_taper_ratio = fold_chord / root_chord_wing
        folding_area = (wingspan / 2.0 - location) * (fold_chord + tip_chord)
        nonfolded_wing_area = wing_area - folding_area
        tc_ratio_fold = tc_ratio_root + location * (tc_ratio_tip - tc_ratio_root) / (
            wingspan / 2.0
        )
        tc_ratio_mean_folded = 0.5 * (tc_ratio_root + tc_ratio_fold)
        nonfolded_AR = 4.0 * location**2 / nonfolded_wing_area

        dRootChordWing_dWingArea = 2 / (wingspan * (1 + taper_ratio))
        dRootChordWing_dWingspan = -2 * wing_area / (wingspan**2 * (1 + taper_ratio))
        dRootChordWing_dTaperRatio = (
            -2 * wing_area / (wingspan * (1 + taper_ratio) ** 2)
        )

        dTipChord_dTaperRatio = (
            taper_ratio * dRootChordWing_dTaperRatio + root_chord_wing
        )
        dTipChord_dWingArea = taper_ratio * dRootChordWing_dWingArea
        dTipChord_dWingspan = taper_ratio * dRootChordWing_dWingspan

        dFoldChord_dWingArea = dRootChordWing_dWingArea + location * (
            dTipChord_dWingArea - dRootChordWing_dWingArea
        ) / (wingspan / 2)
        dFoldChord_dWingspan = (
            dRootChordWing_dWingspan
            + location
            * (
                (wingspan / 2) * (dTipChord_dWingspan - dRootChordWing_dWingspan)
                - (tip_chord - root_chord_wing) * (1 / 2)
            )
            / (wingspan / 2) ** 2
            + dLoc_dWingspan * (tip_chord - root_chord_wing) / (wingspan / 2.0)
        )
        dFoldChord_dTaperRatio = dRootChordWing_dTaperRatio + location * (
            dTipChord_dTaperRatio - dRootChordWing_dTaperRatio
        ) / (wingspan / 2)
        dFoldChord_dy = dLoc_dy * (tip_chord - root_chord_wing) / (wingspan / 2.0)

        dTcRatioFold_dTCR = 1 + location * (-1) / (wingspan / 2)
        dTcRatioFold_dTCT = location / (wingspan / 2)
        dTcRatioFold_dWingspan = (
            (tc_ratio_tip - tc_ratio_root)
            * ((wingspan / 2) * dLoc_dWingspan - location * 0.5)
            / (wingspan / 2) ** 2
        )
        dTcRatioFold_dy = dLoc_dy * (tc_ratio_tip - tc_ratio_root) / (wingspan / 2.0)

        J["nonfolded_taper_ratio", Aircraft.Wing.AREA] = dNFTR_dWingArea = (
            root_chord_wing * dFoldChord_dWingArea
            - fold_chord * dRootChordWing_dWingArea
        ) / root_chord_wing**2
        J["nonfolded_taper_ratio", Aircraft.Wing.SPAN] = dNFTR_dWingspan = (
            root_chord_wing * dFoldChord_dWingspan
            - fold_chord * dRootChordWing_dWingspan
        ) / root_chord_wing**2
        J["nonfolded_taper_ratio", Aircraft.Wing.TAPER_RATIO] = dNFTR_dTaperRatio = (
            root_chord_wing * dFoldChord_dTaperRatio
            - fold_chord * dRootChordWing_dTaperRatio
        ) / root_chord_wing**2
        J["nonfolded_taper_ratio", wrt] = dNFTR_dy = dFoldChord_dy / root_chord_wing

        J[Aircraft.Wing.FOLDING_AREA, Aircraft.Wing.AREA] = dFoldingArea_dWingArea = (
            wingspan / 2.0 - location
        ) * (dFoldChord_dWingArea + dTipChord_dWingArea)
        J[Aircraft.Wing.FOLDING_AREA, Aircraft.Wing.SPAN] = dFoldingArea_dWingspan = (
            wingspan / 2.0 - location
        ) * (dFoldChord_dWingspan + dTipChord_dWingspan) + (fold_chord + tip_chord) * (
            0.5 - dLoc_dWingspan
        )
        J[
            Aircraft.Wing.FOLDING_AREA, Aircraft.Wing.TAPER_RATIO
        ] = dFoldingArea_dTaperRatio = (wingspan / 2.0 - location) * (
            dFoldChord_dTaperRatio + dTipChord_dTaperRatio
        )
        J[Aircraft.Wing.FOLDING_AREA, wrt] = dFoldingArea_dy = -dLoc_dy * (
            fold_chord + tip_chord
        ) + (wingspan / 2 - location) * dLoc_dy * (tip_chord - root_chord_wing) / (
            wingspan / 2.0
        )

        J["nonfolded_wing_area", Aircraft.Wing.AREA] = dNFWA_dWingArea = (
            1 - dFoldingArea_dWingArea
        )
        J[
            "nonfolded_wing_area", Aircraft.Wing.SPAN
        ] = dNFWA_dWingspan = -dFoldingArea_dWingspan
        J[
            "nonfolded_wing_area", Aircraft.Wing.TAPER_RATIO
        ] = dNFWA_dTaperRatio = -dFoldingArea_dTaperRatio
        J["nonfolded_wing_area", wrt] = dNFWA_dy = -dFoldingArea_dy

        J[
            "tc_ratio_mean_folded", Aircraft.Wing.THICKNESS_TO_CHORD_ROOT
        ] = dTCRMeanFolded_dTCR = 0.5 * (1 + dTcRatioFold_dTCR)
        J[
            "tc_ratio_mean_folded", Aircraft.Wing.THICKNESS_TO_CHORD_TIP
        ] = dTCRMeanFolded_dTCT = (0.5 * dTcRatioFold_dTCT)
        J["tc_ratio_mean_folded", Aircraft.Wing.SPAN] = dTCRMeanFolded_dWingspan = (
            0.5 * dTcRatioFold_dWingspan
        )
        J["tc_ratio_mean_folded", wrt] = dTCRMeanFolded_dy = 0.5 * dTcRatioFold_dy

        J["nonfolded_AR", Aircraft.Wing.AREA] = dNFAR_dWingArea = (
            -4 * location**2 / nonfolded_wing_area**2 * dNFWA_dWingArea
        )
        J["nonfolded_AR", Aircraft.Wing.SPAN] = dNFAR_dWingspan = (
            4
            * (
                nonfolded_wing_area * 2 * location * dLoc_dWingspan
                - location**2 * dNFWA_dWingspan
            )
            / nonfolded_wing_area**2
        )
        J["nonfolded_AR", Aircraft.Wing.TAPER_RATIO] = dNFAR_dTaperRatio = (
            -4 * location**2 / nonfolded_wing_area**2 * dNFWA_dTaperRatio
        )
        J["nonfolded_AR", wrt] = dNFAR_dy = (
            4
            * (nonfolded_wing_area * 2 * location * dLoc_dy - location**2 * dNFWA_dy)
            / nonfolded_wing_area**2
        )

        geometric_fuel_vol = (
            fuel_vol_frac
            * 0.888889
            * tc_ratio_mean_folded
            * (nonfolded_wing_area**1.5)
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) / ((nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0))
        a = (nonfolded_wing_area**1.5) * (2.0 * nonfolded_taper_ratio + 1.0)
        num = fuel_vol_frac * 0.888889 * tc_ratio_mean_folded * a
        den = (nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0)

        dA_dWingspan = (
            1.5
            * nonfolded_wing_area**0.5
            * dNFWA_dWingspan
            * (2 * nonfolded_taper_ratio + 1)
            + nonfolded_wing_area**1.5 * 2 * dNFTR_dWingspan
        )
        dA_dy = (
            1.5
            * nonfolded_wing_area**0.5
            * dNFWA_dy
            * (2 * nonfolded_taper_ratio + 1)
            + nonfolded_wing_area**1.5 * 2 * dNFTR_dy
        )

        dNum_dWingspan = (
            fuel_vol_frac * 0.888889 * dTCRMeanFolded_dWingspan * a
            + fuel_vol_frac * 0.888889 * tc_ratio_mean_folded * dA_dWingspan
        )
        dDen_dWingspan = (
            0.5
            * (nonfolded_AR ** (-0.5))
            * dNFAR_dWingspan
            * ((nonfolded_taper_ratio + 1.0) ** 2.0)
            + ((nonfolded_AR**0.5) * 2 * ((nonfolded_taper_ratio + 1.0)))
            * dNFTR_dWingspan
        )
        dNum_dy = (
            fuel_vol_frac * 0.888889 * dTCRMeanFolded_dy * a
            + fuel_vol_frac * 0.888889 * tc_ratio_mean_folded * dA_dy
        )
        dDen_dy = (
            0.5
            * (nonfolded_AR ** (-0.5))
            * dNFAR_dy
            * ((nonfolded_taper_ratio + 1.0) ** 2.0)
            + ((nonfolded_AR**0.5) * 2 * ((nonfolded_taper_ratio + 1.0))) * dNFTR_dy
        )
        dNum_dWingArea = (
            fuel_vol_frac
            * 0.888889
            * tc_ratio_mean_folded
            * 1.5
            * (nonfolded_wing_area**0.5)
            * dNFWA_dWingArea
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) + (
            fuel_vol_frac
            * 0.888889
            * tc_ratio_mean_folded
            * (nonfolded_wing_area**1.5)
            * 2
        ) * dNFTR_dWingArea
        dDen_dWingArea = (
            0.5
            * (nonfolded_AR ** (-0.5))
            * dNFAR_dWingArea
            * ((nonfolded_taper_ratio + 1.0) ** 2.0)
            + ((nonfolded_AR**0.5) * 2 * ((nonfolded_taper_ratio + 1.0)))
            * dNFTR_dWingArea
        )
        dNum_dTaperRatio = (
            fuel_vol_frac
            * 0.888889
            * tc_ratio_mean_folded
            * 1.5
            * (nonfolded_wing_area**0.5)
            * dNFWA_dTaperRatio
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) + (
            fuel_vol_frac
            * 0.888889
            * tc_ratio_mean_folded
            * (nonfolded_wing_area**1.5)
            * 2
        ) * dNFTR_dTaperRatio
        dDen_dTaperRatio = (
            0.5
            * (nonfolded_AR ** (-0.5))
            * dNFAR_dTaperRatio
            * ((nonfolded_taper_ratio + 1.0) ** 2.0)
            + ((nonfolded_AR**0.5) * 2 * ((nonfolded_taper_ratio + 1.0)))
            * dNFTR_dTaperRatio
        )

        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Fuel.WING_FUEL_FRACTION] = (
            0.888889
            * tc_ratio_mean_folded
            * (nonfolded_wing_area**1.5)
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) / ((nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0))
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
            fuel_vol_frac
            * 0.888889
            * dTCRMeanFolded_dTCR
            * (nonfolded_wing_area**1.5)
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) / ((nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0))
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = (
            fuel_vol_frac
            * 0.888889
            * dTCRMeanFolded_dTCT
            * (nonfolded_wing_area**1.5)
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) / ((nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0))
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.SPAN] = (
            den * dNum_dWingspan - num * dDen_dWingspan
        ) / den**2
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.AREA] = (
            den * dNum_dWingArea - num * dDen_dWingArea
        ) / den**2
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.TAPER_RATIO] = (
            den * dNum_dTaperRatio - num * dDen_dTaperRatio
        ) / den**2
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, wrt] = (
            den * dNum_dy - num * dDen_dy) / den**2


class WingGroup(om.Group):
    """
    Group of WingSize, WingParameters, and WingFold for wing parameter computations.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        aviary_options = self.options['aviary_options']

        size = self.add_subsystem(
            "size",
            WingSize(aviary_options=aviary_options,),
            promotes_inputs=["aircraft:*", "mission:*"],
            promotes_outputs=["aircraft:*"],
        )

        if self.options["aviary_options"].get_val(Aircraft.Wing.HAS_FOLD, units='unitless') or self.options["aviary_options"].get_val(Aircraft.Wing.HAS_STRUT, units='unitless'):
            self.add_subsystem(
                "dimensionless_calcs",
                DimensionalNonDimensionalInterchange(aviary_options=aviary_options),
                promotes_inputs=["aircraft:*"],
                promotes_outputs=["aircraft:*"]
            )

        parameters = self.add_subsystem(
            "parameters",
            WingParameters(aviary_options=aviary_options,),
            promotes_inputs=["aircraft:*"],
            promotes_outputs=["aircraft:*"],
        )

        if self.options["aviary_options"].get_val(Aircraft.Wing.HAS_STRUT, units='unitless'):
            strut = self.add_subsystem(
                "strut",
                StrutGeom(
                    aviary_options=aviary_options,
                ),
                promotes_inputs=["aircraft:*"],
                promotes_outputs=["aircraft:*"],
            )

        if self.options["aviary_options"].get_val(Aircraft.Wing.HAS_FOLD, units='unitless'):
            fold = self.add_subsystem(
                "fold",
                WingFold(
                    aviary_options=aviary_options,
                ),
                promotes_inputs=["aircraft:*"],
                promotes_outputs=["aircraft:*"],
            )

            if not self.options["aviary_options"].get_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, units='unitless'):
                check_fold_location_definition(None, aviary_options)
                self.promotes("strut", outputs=["strut_y"])
                self.promotes("fold", inputs=["strut_y"])

        self.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units="unitless")
        self.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless")
        self.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11, units="unitless"
        )
        self.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, val=0.1, units="unitless"
        )
