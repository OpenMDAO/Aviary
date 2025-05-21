import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.geometry.gasp_based.non_dimensional_conversion import (
    DimensionalNonDimensionalInterchange,
)
from aviary.subsystems.geometry.gasp_based.strut import StrutGeom
from aviary.utils.conflict_checks import check_fold_location_definition
from aviary.utils.functions import sigmoidX, dSigmoidXdx
from aviary.variable_info.enums import AircraftTypes, Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission, Settings


class WingSize(om.ExplicitComponent):
    """Computation of wing area and wing span for GASP-based aerodynamics."""

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.LOADING, units='lbf/ft**2')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')

        add_aviary_output(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_output(self, Aircraft.Wing.SPAN, units='ft')

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

        J[Aircraft.Wing.AREA, Mission.Design.GROSS_MASS] = dWA_dGMT = (
            GRAV_ENGLISH_LBM / wing_loading
        )
        J[Aircraft.Wing.AREA, Aircraft.Wing.LOADING] = dWA_dWL = (
            -gross_mass_initial * GRAV_ENGLISH_LBM / wing_loading**2
        )

        J[Aircraft.Wing.SPAN, Aircraft.Wing.ASPECT_RATIO] = 0.5 * wing_area**0.5 * AR ** (-0.5)
        J[Aircraft.Wing.SPAN, Mission.Design.GROSS_MASS] = (
            0.5 * AR**0.5 * wing_area ** (-0.5) * dWA_dGMT
        )
        J[Aircraft.Wing.SPAN, Aircraft.Wing.LOADING] = 0.5 * AR**0.5 * wing_area ** (-0.5) * dWA_dWL


class WingParameters(om.ExplicitComponent):
    """Computation of various wing parameters for GASP-based geometry."""

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP, units='unitless')

        add_aviary_output(self, Aircraft.Wing.CENTER_CHORD, units='ft')
        add_aviary_output(self, Aircraft.Wing.AVERAGE_CHORD, units='ft')
        add_aviary_output(self, Aircraft.Wing.ROOT_CHORD, units='ft')
        add_aviary_output(self, Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, units='unitless')
        add_aviary_output(self, Aircraft.Wing.LEADING_EDGE_SWEEP, units='rad')

    def setup_partials(self):
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

        # For BWB, this formula might need correction
        FHP = (
            2.0
            * (tc_ratio_root * center_chord * (cabin_width - (tc_ratio_root * center_chord))) ** 0.5
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
            * (tc_ratio_root * center_chord * (cabin_width - (tc_ratio_root * center_chord))) ** 0.5
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
        dFHP_dtc_ratio_rootatioRoot = (
            tc_ratio_root * center_chord * (cabin_width - tc_ratio_root * center_chord)
        ) ** (-0.5) * (center_chord * cabin_width - 2 * tc_ratio_root * center_chord**2)
        dFHP_dCabinWidth = (
            (tc_ratio_root * center_chord * (cabin_width - tc_ratio_root * center_chord)) ** (-0.5)
            * tc_ratio_root
            * center_chord
        )
        dFHP_dWingArea = (
            (tc_ratio_root * center_chord * (cabin_width - tc_ratio_root * center_chord)) ** (-0.5)
            * (tc_ratio_root * cabin_width - 2 * tc_ratio_root**2 * center_chord)
            * dCenterChord_dWingArea
        )
        dFHP_dWingspan = (
            (tc_ratio_root * center_chord * (cabin_width - tc_ratio_root * center_chord)) ** (-0.5)
            * (tc_ratio_root * cabin_width - 2 * tc_ratio_root**2 * center_chord)
            * dCenterChord_dWingspan
        )
        dFHP_dTaperRatio = (
            (tc_ratio_root * center_chord * (cabin_width - tc_ratio_root * center_chord)) ** (-0.5)
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
            3 * ((1 + taper_ratio) * AR - (taper_ratio - 1) * AR) / ((1 + taper_ratio) ** 2 * AR**2)
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
        dRootChord_dtc_ratio_rootatioRoot = dFHP_dtc_ratio_rootatioRoot * (
            -tan_sweep_LE / 2 + tan_sweep_TE / 2
        )
        dRootChord_dCabinWidth = dFHP_dCabinWidth * (-tan_sweep_LE / 2 + tan_sweep_TE / 2)
        dRootChord_dAR = FHP * (-dTanSweepLE_dAR / 2 + dTanSweepTE_dAR / 2)
        dRootChord_dSweepC4 = FHP * 0.5 * (-dTanSweepLE_dSweepC4 + dTanSweepTE_dSweepC4)

        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.AREA] = dRootChord_dWingArea
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.SPAN] = dRootChord_dWingspan
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.TAPER_RATIO] = dRootChord_dTaperRatio
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
            dRootChord_dtc_ratio_rootatioRoot
        )
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Fuselage.AVG_DIAMETER] = dRootChord_dCabinWidth
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.ASPECT_RATIO] = dRootChord_dAR
        J[Aircraft.Wing.ROOT_CHORD, Aircraft.Wing.SWEEP] = dRootChord_dSweepC4

        J[Aircraft.Wing.CENTER_CHORD, Aircraft.Wing.AREA] = dCenterChord_dWingArea
        J[Aircraft.Wing.CENTER_CHORD, Aircraft.Wing.SPAN] = dCenterChord_dWingspan
        J[Aircraft.Wing.CENTER_CHORD, Aircraft.Wing.TAPER_RATIO] = dCenterChord_dTaperRatio

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
        J[Aircraft.Wing.AVERAGE_CHORD, Aircraft.Wing.TAPER_RATIO] = dCenterChord_dTaperRatio * (
            2.0 / 3.0
        ) * ((1.0 + taper_ratio) - (taper_ratio / (1.0 + taper_ratio))) + (
            2.0 * center_chord / 3.0
        ) * (1 - ((1 + taper_ratio) - taper_ratio) / (1.0 + taper_ratio) ** 2)

        a = tc_ratio_root - cabin_width / wingspan * (tc_ratio_root - tc_ratio_tip)
        b = 1.0 - cabin_width / wingspan * (1.0 - taper_ratio)
        c = taper_ratio * tc_ratio_tip
        d = 1.0 + taper_ratio - cabin_width / wingspan * (1.0 - taper_ratio)

        dAB_dCabW = a * (taper_ratio - 1) / wingspan + b * (tc_ratio_tip - tc_ratio_root) / wingspan
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
        ] = (1 - cabin_width / wingspan) * b / d
        J[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, Aircraft.Fuselage.AVG_DIAMETER] = (
            d * dAB_dCabW - (a * b + c) * dD_dCabW
        ) / d**2
        J[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, Aircraft.Wing.SPAN] = (
            d * dAB_dWingspan - (a * b + c) * dD_dWingspan
        ) / d**2
        J[
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
        ] = (cabin_width / wingspan * b + taper_ratio) / d
        J[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, Aircraft.Wing.TAPER_RATIO] = (
            d * dABC_dTR - (a * b + c) * dD_dTR
        ) / d**2

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


class WingVolume(om.ExplicitComponent):
    """
    Computation of Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX without considering folds
    for GASP-based geometry. If there are folds, it will be computed again in WingFold.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless')

        add_aviary_output(self, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, units='ft**3')

    def setup_partials(self):
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

    def compute(self, inputs, outputs):
        if self.options[Aircraft.Wing.HAS_FOLD]:
            print('Warning: Aircraft.Wing.HAS_FOLD should be False.')
        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]

        tc_ratio_avg = (
            (tc_ratio_root - cabin_width / wingspan * (tc_ratio_root - tc_ratio_tip))
            * (1.0 - cabin_width / wingspan * (1.0 - taper_ratio))
            + taper_ratio * tc_ratio_tip
        ) / (1.0 + taper_ratio - cabin_width / wingspan * (1.0 - taper_ratio))

        geometric_fuel_vol = (
            fuel_vol_frac * 0.888889 * tc_ratio_avg * (wing_area**1.5) * (2.0 * taper_ratio + 1.0)
        ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
        outputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = geometric_fuel_vol

    def compute_partials(self, inputs, J):
        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]

        tc_ratio_avg = (
            (tc_ratio_root - cabin_width / wingspan * (tc_ratio_root - tc_ratio_tip))
            * (1.0 - cabin_width / wingspan * (1.0 - taper_ratio))
            + taper_ratio * tc_ratio_tip
        ) / (1.0 + taper_ratio - cabin_width / wingspan * (1.0 - taper_ratio))
        a = tc_ratio_root - cabin_width / wingspan * (tc_ratio_root - tc_ratio_tip)
        b = 1.0 - cabin_width / wingspan * (1.0 - taper_ratio)
        c = taper_ratio * tc_ratio_tip
        d = 1.0 + taper_ratio - cabin_width / wingspan * (1.0 - taper_ratio)

        dAB_dCabW = a * (taper_ratio - 1) / wingspan + b * (tc_ratio_tip - tc_ratio_root) / wingspan
        dD_dCabW = (taper_ratio - 1) / wingspan
        dAB_dWingspan = (
            a * cabin_width * (1 - taper_ratio) / wingspan**2
            + b * cabin_width * (tc_ratio_root - tc_ratio_tip) / wingspan**2
        )
        dD_dWingspan = cabin_width * (1 - taper_ratio) / wingspan**2
        dABC_dTR = a * cabin_width / wingspan + tc_ratio_tip
        dD_dTR = 1 + cabin_width / wingspan

        dTCA_dtc_ratio_root = (1 - cabin_width / wingspan) * b / d
        dTCA_dCabW = (d * dAB_dCabW - (a * b + c) * dD_dCabW) / d**2
        dTCA_dWingspan = (d * dAB_dWingspan - (a * b + c) * dD_dWingspan) / d**2
        dTCA_dtc_ratio_tip = (cabin_width / wingspan * b + taper_ratio) / d
        dTCA_dTR = (d * dABC_dTR - (a * b + c) * dD_dTR) / d**2

        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        num = fuel_vol_frac * 0.888889 * tc_ratio_avg * (wing_area**1.5) * (2.0 * taper_ratio + 1.0)
        den = (AR**0.5) * ((taper_ratio + 1.0) ** 2.0)
        dNum_dTR = (
            fuel_vol_frac * 0.888889 * dTCA_dTR * (wing_area**1.5) * (2.0 * taper_ratio + 1.0)
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
            * dTCA_dtc_ratio_root
            * (wing_area**1.5)
            * (2.0 * taper_ratio + 1.0)
        ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Fuselage.AVG_DIAMETER] = (
            fuel_vol_frac * 0.888889 * dTCA_dCabW * (wing_area**1.5) * (2.0 * taper_ratio + 1.0)
        ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.SPAN] = (
            fuel_vol_frac * 0.888889 * dTCA_dWingspan * (wing_area**1.5) * (2.0 * taper_ratio + 1.0)
        ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = (
            fuel_vol_frac
            * 0.888889
            * dTCA_dtc_ratio_tip
            * (wing_area**1.5)
            * (2.0 * taper_ratio + 1.0)
        ) / ((AR**0.5) * ((taper_ratio + 1.0) ** 2.0))


class BWBWingVolume(om.ExplicitComponent):
    """
    Computation of Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX for BWB
    without considering fold for GASP-based geometry. If HAS_FOLD is True,
    Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX will be updated using this one.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        self.options.declare('mu', default=10.0, types=float)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(
            self, Aircraft.LandingGear.MAIN_GEAR_LOCATION, units='unitless', desc='YMG'
        )
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless', desc='TCR')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP, units='unitless', desc='TCT')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft', desc='SWF')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft', desc='B')
        # In GASP, the variable is CROOT. In Aviary, it is Aircraft.Wing.CENTER_CHORD
        add_aviary_input(self, Aircraft.Wing.CENTER_CHORD, units='ft', desc='CROOT')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless', desc='SLM')
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless', desc='SKWF')

        if self.options[Aircraft.Wing.HAS_FOLD]:
            self.add_output('wing_volume_no_fold', units='ft**3', desc='FVOLW_GEOMX')
        else:
            add_aviary_output(
                self, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, units='ft**3', desc='FVOLW_GEOM'
            )

    def setup_partials(self):
        if self.options[Aircraft.Wing.HAS_FOLD]:
            self.declare_partials(
                'wing_volume_no_fold',
                [
                    Aircraft.LandingGear.MAIN_GEAR_LOCATION,
                    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                    Aircraft.Wing.SPAN,
                    Aircraft.Fuselage.AVG_DIAMETER,
                    Aircraft.Wing.CENTER_CHORD,
                    Aircraft.Wing.TAPER_RATIO,
                    Aircraft.Fuel.WING_FUEL_FRACTION,
                ],
            )
        else:
            self.declare_partials(
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
                [
                    Aircraft.LandingGear.MAIN_GEAR_LOCATION,
                    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                    Aircraft.Wing.SPAN,
                    Aircraft.Fuselage.AVG_DIAMETER,
                    Aircraft.Wing.CENTER_CHORD,
                    Aircraft.Wing.TAPER_RATIO,
                    Aircraft.Fuel.WING_FUEL_FRACTION,
                ],
            )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        mu = self.options['mu']

        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        loc_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_LOCATION]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuselage_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        wingspan = inputs[Aircraft.Wing.SPAN]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        center_chort = inputs[Aircraft.Wing.CENTER_CHORD]

        if smooth:
            WID_GRX = 6 * sigmoidX(loc_main_gear, 0.0, mu)
        else:
            if loc_main_gear > 0:
                WID_GRX = 6
            else:
                WID_GRX = 0

        CTIP = taper_ratio * center_chort
        TCF = (
            tc_ratio_root
            - 2.0 * (0.5 * fuselage_width + WID_GRX) * (tc_ratio_root - tc_ratio_tip) / wingspan
        )
        CFUEL = (
            center_chort + 2.0 * (0.5 * fuselage_width + WID_GRX) * (CTIP - center_chort) / wingspan
        )
        if CFUEL == 0.0:
            if verbosity > Verbosity.BRIEF:
                print('warning: CFUEL is 0.0')
        SW_NFX = (0.5 * wingspan - (0.5 * fuselage_width + WID_GRX)) * (CFUEL + CTIP)
        if SW_NFX == 0.0:
            if verbosity > Verbosity.BRIEF:
                print('warning: SW_NFX is 0.0')
        SLM_NFX = CTIP / CFUEL
        AR_NFX = (wingspan - fuselage_width - 2.0 * WID_GRX) ** 2 / SW_NFX
        if AR_NFX == 0.0:
            if verbosity > Verbosity.BRIEF:
                print('warning: AR_NFX is 0.0')
        TCM = 0.5 * (TCF + tc_ratio_tip)
        # 0.888889 is some empirical factor derived from historical data to estimate
        # the wing fuel volume based on the wing geometric characteristics.
        numer = fuel_vol_frac * 0.888889 * TCM * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
        denom = AR_NFX**0.5 * (SLM_NFX + 1.0) ** 2
        FVOLW_GEOM = numer / denom
        if self.options[Aircraft.Wing.HAS_FOLD]:
            outputs['wing_volume_no_fold'] = FVOLW_GEOM
        else:
            outputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = FVOLW_GEOM

    def compute_partials(self, inputs, J):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        mu = self.options['mu']

        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        loc_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_LOCATION]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuselage_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        wingspan = inputs[Aircraft.Wing.SPAN]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        center_chort = inputs[Aircraft.Wing.CENTER_CHORD]

        if smooth:
            WID_GRX = 6 * sigmoidX(loc_main_gear, 0.0, mu)
            dWID_GRX_dloc_main_gear = dSigmoidXdx(loc_main_gear, 0.0, mu) / mu
        else:
            if loc_main_gear > 0:
                WID_GRX = 6.0
            else:
                WID_GRX = 0.0
            dWID_GRX_dloc_main_gear = 0.0

        TCF = (
            tc_ratio_root
            - 2.0 * (0.5 * fuselage_width + WID_GRX) * (tc_ratio_root - tc_ratio_tip) / wingspan
        )
        dTCF_dloc_main_gear = (
            -2.0 * dWID_GRX_dloc_main_gear * (tc_ratio_root - tc_ratio_tip) / wingspan
        )
        dTCF_dtaper_ratio = 0.0
        dTCF_dcenter_chord_wing = 0.0
        dTCF_dtc_ratio_root = 1.0 - 2.0 * (0.5 * fuselage_width + WID_GRX) / wingspan
        dTCF_dtc_ratio_tip = 2.0 * (0.5 * fuselage_width + WID_GRX) / wingspan
        dTCF_dSWF = -(tc_ratio_root - tc_ratio_tip) / wingspan
        dTCF_dwingspan = (
            2.0 * (0.5 * fuselage_width + WID_GRX) * (tc_ratio_root - tc_ratio_tip) / wingspan**2
        )

        CFUEL = (
            center_chort
            + 2.0 * (0.5 * fuselage_width + WID_GRX) * (taper_ratio - 1.0) * center_chort / wingspan
        )
        dCFUEL_dcenter_chord_wing = (
            1.0 + 2.0 * (0.5 * fuselage_width + WID_GRX) * (taper_ratio - 1.0) / wingspan
        )
        dCFUEL_dSWF = (taper_ratio - 1.0) * center_chort / wingspan
        dCFUEL_dloc_main_gear = 2.0 * dWID_GRX_dloc_main_gear * (taper_ratio - 1.0) / wingspan
        dCFUEL_dtaper_ratio = 2.0 * (0.5 * fuselage_width + WID_GRX) * center_chort / wingspan
        dCFUEL_dwingspan = (
            -2.0
            * (0.5 * fuselage_width + WID_GRX)
            * (taper_ratio - 1.0)
            * center_chort
            / wingspan**2
        )

        SW_NFX = (0.5 * wingspan - (0.5 * fuselage_width + WID_GRX)) * (
            CFUEL + taper_ratio * center_chort
        )
        dSW_NFX_dcenter_chord_wing = (0.5 * wingspan - (0.5 * fuselage_width + WID_GRX)) * (
            dCFUEL_dcenter_chord_wing + taper_ratio
        )
        dSW_NFX_dSWF = (
            -0.5 * (CFUEL + taper_ratio * center_chort)
            + (0.5 * wingspan - (0.5 * fuselage_width + WID_GRX)) * dCFUEL_dSWF
        )
        dSW_NFX_dloc_main_gear = -dWID_GRX_dloc_main_gear * (CFUEL + taper_ratio * center_chort)
        dSW_NFX_dtaper_ratio = (0.5 * wingspan - (0.5 * fuselage_width + WID_GRX)) * (
            dCFUEL_dtaper_ratio + center_chort
        )
        dSW_NFX_dwingspan = (
            0.5 * (CFUEL + taper_ratio * center_chort)
            + (0.5 * wingspan - (0.5 * fuselage_width + WID_GRX)) * dCFUEL_dwingspan
        )

        SLM_NFX = taper_ratio * center_chort / CFUEL
        dtaper_ratio_NFX_dtaper_ratio = (
            center_chort / CFUEL - taper_ratio * center_chort * dCFUEL_dtaper_ratio / CFUEL**2
        )
        dtaper_ratio_NFX_dcenter_chord_wing = (
            taper_ratio / CFUEL - taper_ratio * center_chort * dCFUEL_dcenter_chord_wing / CFUEL**2
        )
        dtaper_ratio_NFX_dSWF = -taper_ratio * center_chort * dCFUEL_dSWF / CFUEL**2
        dtaper_ratio_NFX_dloc_main_gear = (
            -taper_ratio * center_chort * dCFUEL_dloc_main_gear / CFUEL**2
        )
        dtaper_ratio_NFX_dwingspan = -taper_ratio * center_chort * dCFUEL_dwingspan / CFUEL**2

        AR_NFX = (wingspan - fuselage_width - 2.0 * WID_GRX) ** 2 / SW_NFX
        dAR_NFX_dtaper_ratio = (
            -((wingspan - fuselage_width - 2.0 * WID_GRX) ** 2) * dSW_NFX_dtaper_ratio / SW_NFX**2
        )
        dAR_NFX_dcenter_chord_wing = (
            -((wingspan - fuselage_width - 2.0 * WID_GRX) ** 2)
            * dSW_NFX_dcenter_chord_wing
            / SW_NFX**2
        )
        dAR_NFX_dSWF = (
            -2 * (wingspan - fuselage_width - 2.0 * WID_GRX) / SW_NFX
            - (wingspan - fuselage_width - 2.0 * WID_GRX) ** 2 * dSW_NFX_dSWF / SW_NFX**2
        )
        dAR_NFX_dloc_main_gear = (
            -((wingspan - fuselage_width - 2.0 * WID_GRX) ** 2) * dSW_NFX_dloc_main_gear / SW_NFX**2
        )
        dAR_NFX_dwingspan = (
            2 * (wingspan - fuselage_width - 2.0 * WID_GRX) / SW_NFX
            - (wingspan - fuselage_width - 2.0 * WID_GRX) ** 2 * dSW_NFX_dwingspan / SW_NFX**2
        )

        TCM = 0.5 * (TCF + tc_ratio_tip)
        dTCM_dloc_main_gear = 0.5 * dTCF_dloc_main_gear
        dTCM_dtaper_ratio = 0.5 * dTCF_dtaper_ratio
        dTCM_dcenter_chord_wing = 0.5 * dTCF_dcenter_chord_wing
        dTCM_dtc_ratio_root = 0.5 * dTCF_dtc_ratio_root
        dTCM_dtc_ratio_tip = 0.5 * dTCF_dtc_ratio_tip + 0.5
        dTCM_dSWF = 0.5 * dTCF_dSWF
        dTCM_dwingspan = 0.5 * dTCF_dwingspan

        numer = fuel_vol_frac * 0.888889 * TCM * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
        dnumer_dfuel_vol_frac = 0.888889 * TCM * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
        dnumer_dloc_main_gear = (
            fuel_vol_frac * 0.888889 * dTCM_dloc_main_gear * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac
            * 0.888889
            * TCM
            * 1.5
            * dSW_NFX_dloc_main_gear
            * SW_NFX**0.5
            * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac * 0.888889 * TCM * SW_NFX**1.5 * (2.0 * dtaper_ratio_NFX_dloc_main_gear)
        )
        dnumer_dtaper_ratio = (
            fuel_vol_frac * 0.888889 * dTCM_dtaper_ratio * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac
            * 0.888889
            * TCM
            * 1.5
            * dSW_NFX_dtaper_ratio
            * SW_NFX**0.5
            * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac * 0.888889 * TCM * SW_NFX**1.5 * (2.0 * dtaper_ratio_NFX_dtaper_ratio)
        )
        dnumer_dcenter_chord_wing = (
            fuel_vol_frac * 0.888889 * dTCM_dcenter_chord_wing * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac
            * 0.888889
            * TCM
            * 1.5
            * dSW_NFX_dcenter_chord_wing
            * SW_NFX**0.5
            * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac
            * 0.888889
            * TCM
            * SW_NFX**1.5
            * (2.0 * dtaper_ratio_NFX_dcenter_chord_wing)
        )
        dnumer_dtc_ratio_root = (
            fuel_vol_frac * 0.888889 * dTCM_dtc_ratio_root * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
        )
        dnumer_dtc_ratio_tip = (
            fuel_vol_frac * 0.888889 * dTCM_dtc_ratio_tip * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
        )
        dnumer_dSWF = (
            fuel_vol_frac * 0.888889 * dTCM_dSWF * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac
            * 0.888889
            * TCM
            * 1.5
            * dSW_NFX_dSWF
            * SW_NFX**0.5
            * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac * 0.888889 * TCM * SW_NFX**1.5 * (2.0 * dtaper_ratio_NFX_dSWF)
        )
        dnumer_dwingspan = (
            fuel_vol_frac * 0.888889 * dTCM_dwingspan * SW_NFX**1.5 * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac
            * 0.888889
            * TCM
            * 1.5
            * dSW_NFX_dwingspan
            * SW_NFX**0.5
            * (2.0 * SLM_NFX + 1.0)
            + fuel_vol_frac * 0.888889 * TCM * SW_NFX**1.5 * (2.0 * dtaper_ratio_NFX_dwingspan)
        )

        denom = AR_NFX**0.5 * (SLM_NFX + 1.0) ** 2
        ddenom_dtaper_ratio = 0.5 * dAR_NFX_dtaper_ratio / AR_NFX**0.5 * (
            SLM_NFX + 1.0
        ) ** 2 + 2 * AR_NFX**0.5 * dtaper_ratio_NFX_dtaper_ratio * (SLM_NFX + 1.0)
        ddenom_dcenter_chord_wing = 0.5 * dAR_NFX_dcenter_chord_wing / AR_NFX**0.5 * (
            SLM_NFX + 1.0
        ) ** 2 + 2 * AR_NFX**0.5 * dtaper_ratio_NFX_dcenter_chord_wing * (SLM_NFX + 1.0)
        ddenom_dSWF = 0.5 * dAR_NFX_dSWF / AR_NFX**0.5 * (
            SLM_NFX + 1.0
        ) ** 2 + 2 * AR_NFX**0.5 * dtaper_ratio_NFX_dSWF * (SLM_NFX + 1.0)
        ddenom_dloc_main_gear = 0.5 * dAR_NFX_dloc_main_gear / AR_NFX**0.5 * (
            SLM_NFX + 1.0
        ) ** 2 + 2 * AR_NFX**0.5 * dtaper_ratio_NFX_dloc_main_gear * (SLM_NFX + 1.0)
        ddenom_dwingspan = 0.5 * dAR_NFX_dwingspan / AR_NFX**0.5 * (
            SLM_NFX + 1.0
        ) ** 2 + 2 * AR_NFX**0.5 * dtaper_ratio_NFX_dwingspan * (SLM_NFX + 1.0)
        ddenom_dfuel_vol_frac = 0.0
        ddenom_dtc_ratio_root = 0.0
        ddenom_dtc_ratio_tip = 0.0

        # FVOLW_GEOM = numer / denom
        dFVOLW_GEOM_dfuel_vol_frac = (
            dnumer_dfuel_vol_frac * denom - numer * ddenom_dfuel_vol_frac
        ) / denom**2
        dFVOLW_GEOM_dtc_ratio_root = (
            dnumer_dtc_ratio_root * denom - numer * ddenom_dtc_ratio_root
        ) / denom**2
        dFVOLW_GEOM_dtc_ratio_tip = (
            dnumer_dtc_ratio_tip * denom - numer * ddenom_dtc_ratio_tip
        ) / denom**2
        dFVOLW_GEOM_dloc_main_gear = (
            dnumer_dloc_main_gear * denom - numer * ddenom_dloc_main_gear
        ) / denom**2
        dFVOLW_GEOM_dtaper_ratio = (
            dnumer_dtaper_ratio * denom - numer * ddenom_dtaper_ratio
        ) / denom**2
        dFVOLW_GEOM_dcenter_chord_wing = (
            dnumer_dcenter_chord_wing * denom - numer * ddenom_dcenter_chord_wing
        ) / denom**2
        dFVOLW_GEOM_dSWF = (dnumer_dSWF * denom - numer * ddenom_dSWF) / denom**2
        dFVOLW_GEOM_dwingspan = (dnumer_dwingspan * denom - numer * ddenom_dwingspan) / denom**2

        if self.options[Aircraft.Wing.HAS_FOLD]:
            J['wing_volume_no_fold', Aircraft.Fuel.WING_FUEL_FRACTION] = dFVOLW_GEOM_dfuel_vol_frac
            J['wing_volume_no_fold', Aircraft.LandingGear.MAIN_GEAR_LOCATION] = (
                dFVOLW_GEOM_dloc_main_gear
            )
            J['wing_volume_no_fold', Aircraft.Wing.TAPER_RATIO] = dFVOLW_GEOM_dtaper_ratio
            J['wing_volume_no_fold', Aircraft.Wing.CENTER_CHORD] = dFVOLW_GEOM_dcenter_chord_wing
            J['wing_volume_no_fold', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
                dFVOLW_GEOM_dtc_ratio_root
            )
            J['wing_volume_no_fold', Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = (
                dFVOLW_GEOM_dtc_ratio_tip
            )
            J['wing_volume_no_fold', Aircraft.Fuselage.AVG_DIAMETER] = dFVOLW_GEOM_dSWF
            J['wing_volume_no_fold', Aircraft.Wing.SPAN] = dFVOLW_GEOM_dwingspan
        else:
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Fuel.WING_FUEL_FRACTION] = (
                dFVOLW_GEOM_dfuel_vol_frac
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.LandingGear.MAIN_GEAR_LOCATION] = (
                dFVOLW_GEOM_dloc_main_gear
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.TAPER_RATIO] = (
                dFVOLW_GEOM_dtaper_ratio
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.CENTER_CHORD] = (
                dFVOLW_GEOM_dcenter_chord_wing
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
                dFVOLW_GEOM_dtc_ratio_root
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = (
                dFVOLW_GEOM_dtc_ratio_tip
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Fuselage.AVG_DIAMETER] = (
                dFVOLW_GEOM_dSWF
            )
            J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.SPAN] = dFVOLW_GEOM_dwingspan


class WingFoldArea(om.ExplicitComponent):
    """
    Computation of folding area.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.CHOOSE_FOLD_LOCATION)

    def setup(self):
        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            self.add_input(
                'strut_y', val=25, units='ft', desc='YSTRUT: attachment location of strut'
            )
        else:
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN, units='ft')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')

        add_aviary_output(self, Aircraft.Wing.FOLDING_AREA, units='ft**2')

    def setup_partials(self):
        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            self.declare_partials(Aircraft.Wing.FOLDING_AREA, 'strut_y')
        else:
            self.declare_partials(Aircraft.Wing.FOLDING_AREA, Aircraft.Wing.FOLDED_SPAN)

        self.declare_partials(
            Aircraft.Wing.FOLDING_AREA,
            [Aircraft.Wing.SPAN, Aircraft.Wing.AREA, Aircraft.Wing.TAPER_RATIO],
        )

    def compute(self, inputs, outputs):
        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]

        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            strut_y = inputs['strut_y']
            location = strut_y
        else:
            fold_y = inputs[Aircraft.Wing.FOLDED_SPAN]
            location = fold_y / 2.0

        root_chord_wing = 2 * wing_area / (wingspan * (1 + taper_ratio))
        tip_chord = taper_ratio * root_chord_wing
        fold_chord = root_chord_wing + location * (tip_chord - root_chord_wing) / (wingspan / 2.0)
        folding_area = (wingspan / 2.0 - location) * (fold_chord + tip_chord)

        if (wingspan / 2.0) < location:
            raise ValueError(
                'Error: The wingspan provided is less than the wingspan of the wing fold.'
            )

        outputs[Aircraft.Wing.FOLDING_AREA] = folding_area

    def compute_partials(self, inputs, J):
        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]

        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            strut_y = inputs['strut_y']
            location = strut_y
            dLoc_dWingspan = 0
            dLoc_dy = 1
            wrt = 'strut_y'

        else:
            fold_y = inputs[Aircraft.Wing.FOLDED_SPAN]
            wrt = Aircraft.Wing.FOLDED_SPAN

            location = fold_y / 2.0
            dLoc_dWingspan = 0
            dLoc_dy = 1 / 2

        root_chord_wing = 2 * wing_area / (wingspan * (1 + taper_ratio))
        tip_chord = taper_ratio * root_chord_wing
        fold_chord = root_chord_wing + location * (tip_chord - root_chord_wing) / (wingspan / 2.0)

        dRootChordWing_dWingArea = 2 / (wingspan * (1 + taper_ratio))
        dRootChordWing_dWingspan = -2 * wing_area / (wingspan**2 * (1 + taper_ratio))
        dRootChordWing_dTaperRatio = -2 * wing_area / (wingspan * (1 + taper_ratio) ** 2)

        dTipChord_dTaperRatio = taper_ratio * dRootChordWing_dTaperRatio + root_chord_wing
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

        J[Aircraft.Wing.FOLDING_AREA, Aircraft.Wing.AREA] = (wingspan / 2.0 - location) * (
            dFoldChord_dWingArea + dTipChord_dWingArea
        )
        J[Aircraft.Wing.FOLDING_AREA, Aircraft.Wing.SPAN] = (wingspan / 2.0 - location) * (
            dFoldChord_dWingspan + dTipChord_dWingspan
        ) + (fold_chord + tip_chord) * (0.5 - dLoc_dWingspan)
        J[Aircraft.Wing.FOLDING_AREA, Aircraft.Wing.TAPER_RATIO] = (wingspan / 2.0 - location) * (
            dFoldChord_dTaperRatio + dTipChord_dTaperRatio
        )
        J[Aircraft.Wing.FOLDING_AREA, wrt] = -dLoc_dy * (fold_chord + tip_chord) + (
            wingspan / 2 - location
        ) * dLoc_dy * (tip_chord - root_chord_wing) / (wingspan / 2.0)


class WingFoldVolume(om.ExplicitComponent):
    """
    Computation of taper ratio between wing root and fold location, wing area of
    part of wings that does not fold, mean value of thickess to chord ratio between
    root and fold, aspect ratio of non-folding part of wing, wing tank fuel volume.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.CHOOSE_FOLD_LOCATION)

    def setup(self):
        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            self.add_input(
                'strut_y', val=25, units='ft', desc='YSTRUT: attachment location of strut'
            )
        else:
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN)

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless')
        add_aviary_input(self, Aircraft.Wing.FOLDING_AREA, units='ft**2')

        # all of the non-Aviary variable outputs are not needed.
        self.add_output(
            'nonfolded_taper_ratio',
            val=0.1,
            units='unitless',
            desc='SLM_NF: taper ratio between wing root and fold location',
        )
        self.add_output(
            'nonfolded_wing_area',
            val=150,
            units='ft**2',
            desc='SW_NF: wing area of part of wings that does not fold',
        )
        self.add_output(
            'tc_ratio_mean_folded',
            val=0.12,
            units='unitless',
            desc='TCM: mean value of thickess to chord ratio between root and fold',
        )
        self.add_output(
            'nonfolded_AR',
            val=10,
            units='unitless',
            desc='AR_NF: aspect ratio of non-folding part of wing',
        )

        add_aviary_output(self, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, units='ft**3')

    def setup_partials(self):
        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            self.declare_partials('nonfolded_taper_ratio', 'strut_y')
            self.declare_partials('nonfolded_wing_area', 'strut_y')
            self.declare_partials('tc_ratio_mean_folded', 'strut_y')
            self.declare_partials('nonfolded_AR', 'strut_y')
            self.declare_partials(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 'strut_y')
        else:
            self.declare_partials('nonfolded_taper_ratio', Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials('nonfolded_wing_area', Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials('tc_ratio_mean_folded', Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials('nonfolded_AR', Aircraft.Wing.FOLDED_SPAN)
            self.declare_partials(
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.FOLDED_SPAN
            )

        self.declare_partials(
            'nonfolded_taper_ratio',
            [
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
            ],
        )
        self.declare_partials(
            'nonfolded_wing_area',
            [
                Aircraft.Wing.AREA,
                Aircraft.Wing.FOLDING_AREA,
            ],
        )
        self.declare_partials(
            'tc_ratio_mean_folded',
            [
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                Aircraft.Wing.SPAN,
            ],
        )
        self.declare_partials(
            'nonfolded_AR',
            [
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.FOLDING_AREA,
            ],
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
                Aircraft.Wing.FOLDING_AREA,
            ],
        )

    def compute(self, inputs, outputs):
        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        folding_area = inputs[Aircraft.Wing.FOLDING_AREA]

        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            strut_y = inputs['strut_y']
            location = strut_y

        else:
            fold_y = inputs[Aircraft.Wing.FOLDED_SPAN]
            location = fold_y / 2.0

        root_chord_wing = 2 * wing_area / (wingspan * (1 + taper_ratio))
        tip_chord = taper_ratio * root_chord_wing
        fold_chord = root_chord_wing + location * (tip_chord - root_chord_wing) / (wingspan / 2.0)
        nonfolded_taper_ratio = fold_chord / root_chord_wing
        nonfolded_wing_area = wing_area - folding_area

        tc_ratio_fold = tc_ratio_root + location * (tc_ratio_tip - tc_ratio_root) / (wingspan / 2.0)
        tc_ratio_mean_folded = 0.5 * (tc_ratio_root + tc_ratio_fold)
        nonfolded_AR = 4.0 * location**2 / nonfolded_wing_area
        geometric_fuel_vol = (
            fuel_vol_frac
            * 0.888889
            * tc_ratio_mean_folded
            * (nonfolded_wing_area**1.5)
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) / ((nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0))

        outputs['nonfolded_taper_ratio'] = nonfolded_taper_ratio
        outputs['nonfolded_wing_area'] = nonfolded_wing_area
        outputs['tc_ratio_mean_folded'] = tc_ratio_mean_folded
        outputs['nonfolded_AR'] = nonfolded_AR
        outputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = geometric_fuel_vol

    def compute_partials(self, inputs, J):
        wing_area = inputs[Aircraft.Wing.AREA]
        wingspan = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        folding_area = inputs[Aircraft.Wing.FOLDING_AREA]

        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            strut_y = inputs['strut_y']
            location = strut_y
            dLoc_dWingspan = 0
            dLoc_dy = 1
            wrt = 'strut_y'

        else:
            fold_y = inputs[Aircraft.Wing.FOLDED_SPAN]
            wrt = Aircraft.Wing.FOLDED_SPAN

            location = fold_y / 2.0
            dLoc_dWingspan = 0
            dLoc_dy = 1 / 2

        root_chord_wing = 2 * wing_area / (wingspan * (1 + taper_ratio))
        dRootChordWing_dWingArea = 2 / (wingspan * (1 + taper_ratio))
        dRootChordWing_dWingspan = -2 * wing_area / (wingspan**2 * (1 + taper_ratio))
        dRootChordWing_dTaperRatio = -2 * wing_area / (wingspan * (1 + taper_ratio) ** 2)

        tip_chord = taper_ratio * root_chord_wing
        dTipChord_dTaperRatio = taper_ratio * dRootChordWing_dTaperRatio + root_chord_wing
        dTipChord_dWingArea = taper_ratio * dRootChordWing_dWingArea
        dTipChord_dWingspan = taper_ratio * dRootChordWing_dWingspan

        fold_chord = root_chord_wing + location * (tip_chord - root_chord_wing) / (wingspan / 2.0)
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

        nonfolded_taper_ratio = fold_chord / root_chord_wing
        dNFTR_dWingArea = (
            root_chord_wing * dFoldChord_dWingArea - fold_chord * dRootChordWing_dWingArea
        ) / root_chord_wing**2
        dNFTR_dWingspan = (
            root_chord_wing * dFoldChord_dWingspan - fold_chord * dRootChordWing_dWingspan
        ) / root_chord_wing**2
        dNFTR_dTaperRatio = (
            root_chord_wing * dFoldChord_dTaperRatio - fold_chord * dRootChordWing_dTaperRatio
        ) / root_chord_wing**2
        dNFTR_dy = dFoldChord_dy / root_chord_wing

        J['nonfolded_taper_ratio', Aircraft.Wing.AREA] = dNFTR_dWingArea
        J['nonfolded_taper_ratio', Aircraft.Wing.SPAN] = dNFTR_dWingspan
        J['nonfolded_taper_ratio', Aircraft.Wing.TAPER_RATIO] = dNFTR_dTaperRatio
        J['nonfolded_taper_ratio', wrt] = dNFTR_dy

        nonfolded_wing_area = wing_area - folding_area
        dNFWA_dWingArea = 1
        dNFWA_dWingFoldArea = -1
        J['nonfolded_wing_area', Aircraft.Wing.AREA] = dNFWA_dWingArea
        J['nonfolded_wing_area', Aircraft.Wing.FOLDING_AREA] = dNFWA_dWingFoldArea

        tc_ratio_fold = tc_ratio_root + location * (tc_ratio_tip - tc_ratio_root) / (wingspan / 2.0)
        dtc_ratio_rootatioFold_dtc_ratio_root = 1 + location * (-1) / (wingspan / 2)
        dtc_ratio_rootatioFold_dtc_ratio_tip = location / (wingspan / 2)
        dtc_ratio_rootatioFold_dWingspan = (
            (tc_ratio_tip - tc_ratio_root)
            * ((wingspan / 2) * dLoc_dWingspan - location * 0.5)
            / (wingspan / 2) ** 2
        )
        dtc_ratio_rootatioFold_dy = dLoc_dy * (tc_ratio_tip - tc_ratio_root) / (wingspan / 2.0)

        tc_ratio_mean_folded = 0.5 * (tc_ratio_root + tc_ratio_fold)
        dtc_ratio_rootMeanFolded_dtc_ratio_root = 0.5 * (1 + dtc_ratio_rootatioFold_dtc_ratio_root)
        dtc_ratio_rootMeanFolded_dtc_ratio_tip = 0.5 * dtc_ratio_rootatioFold_dtc_ratio_tip
        dtc_ratio_rootMeanFolded_dWingspan = 0.5 * dtc_ratio_rootatioFold_dWingspan
        dtc_ratio_rootMeanFolded_dy = 0.5 * dtc_ratio_rootatioFold_dy

        J['tc_ratio_mean_folded', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
            dtc_ratio_rootMeanFolded_dtc_ratio_root
        )
        J['tc_ratio_mean_folded', Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = (
            dtc_ratio_rootMeanFolded_dtc_ratio_tip
        )
        J['tc_ratio_mean_folded', Aircraft.Wing.SPAN] = dtc_ratio_rootMeanFolded_dWingspan
        J['tc_ratio_mean_folded', wrt] = dtc_ratio_rootMeanFolded_dy

        nonfolded_AR = 4.0 * location**2 / nonfolded_wing_area
        dNFAR_dWingArea = -4.0 * location**2 / nonfolded_wing_area**2
        dNFAR_dWingFoldArea = 4.0 * location**2 / nonfolded_wing_area**2
        dNFAR_dWingspan = 4.0 * (2.0 * location * dLoc_dWingspan) / nonfolded_wing_area
        dNFAR_dTaperRatio = 0
        dNFAR_dy = 4 * (2.0 * location * dLoc_dy) / nonfolded_wing_area

        J['nonfolded_AR', Aircraft.Wing.AREA] = dNFAR_dWingArea
        J['nonfolded_AR', Aircraft.Wing.FOLDING_AREA] = dNFAR_dWingFoldArea
        J['nonfolded_AR', Aircraft.Wing.SPAN] = dNFAR_dWingspan
        J['nonfolded_AR', wrt] = dNFAR_dy

        a = (nonfolded_wing_area**1.5) * (2.0 * nonfolded_taper_ratio + 1.0)
        dA_dWingspan = nonfolded_wing_area**1.5 * 2 * dNFTR_dWingspan
        dA_dy = nonfolded_wing_area**1.5 * 2 * dNFTR_dy

        num = fuel_vol_frac * 0.888889 * tc_ratio_mean_folded * a
        dNum_dWingspan = (
            fuel_vol_frac * 0.888889 * dtc_ratio_rootMeanFolded_dWingspan * a
            + fuel_vol_frac * 0.888889 * tc_ratio_mean_folded * dA_dWingspan
        )
        dNum_dy = (
            fuel_vol_frac * 0.888889 * dtc_ratio_rootMeanFolded_dy * a
            + fuel_vol_frac * 0.888889 * tc_ratio_mean_folded * dA_dy
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
            fuel_vol_frac * 0.888889 * tc_ratio_mean_folded * (nonfolded_wing_area**1.5) * 2
        ) * dNFTR_dWingArea
        dNum_dWingFoldArea = (
            fuel_vol_frac
            * 0.888889
            * tc_ratio_mean_folded
            * 1.5
            * (nonfolded_wing_area**0.5)
            * dNFWA_dWingFoldArea
            * (2.0 * nonfolded_taper_ratio + 1.0)
        )

        dNum_dTaperRatio = (
            fuel_vol_frac * 0.888889 * tc_ratio_mean_folded * (nonfolded_wing_area**1.5) * 2
        ) * dNFTR_dTaperRatio

        den = (nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0)

        dDen_dWingspan = (
            0.5
            * (nonfolded_AR ** (-0.5))
            * dNFAR_dWingspan
            * ((nonfolded_taper_ratio + 1.0) ** 2.0)
            + ((nonfolded_AR**0.5) * 2 * (nonfolded_taper_ratio + 1.0)) * dNFTR_dWingspan
        )
        dDen_dy = (
            0.5 * (nonfolded_AR ** (-0.5)) * dNFAR_dy * ((nonfolded_taper_ratio + 1.0) ** 2.0)
            + ((nonfolded_AR**0.5) * 2 * (nonfolded_taper_ratio + 1.0)) * dNFTR_dy
        )
        dDen_dWingArea = (
            0.5
            * (nonfolded_AR ** (-0.5))
            * dNFAR_dWingArea
            * ((nonfolded_taper_ratio + 1.0) ** 2.0)
            + ((nonfolded_AR**0.5) * 2 * (nonfolded_taper_ratio + 1.0)) * dNFTR_dWingArea
        )
        dDen_dWingFoldArea = (
            0.5
            * (nonfolded_AR ** (-0.5))
            * dNFAR_dWingFoldArea
            * ((nonfolded_taper_ratio + 1.0) ** 2.0)
        )
        dDen_dTaperRatio = (
            0.5
            * (nonfolded_AR ** (-0.5))
            * dNFAR_dTaperRatio
            * ((nonfolded_taper_ratio + 1.0) ** 2.0)
            + ((nonfolded_AR**0.5) * 2 * (nonfolded_taper_ratio + 1.0)) * dNFTR_dTaperRatio
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
            * dtc_ratio_rootMeanFolded_dtc_ratio_root
            * (nonfolded_wing_area**1.5)
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) / ((nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0))
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = (
            fuel_vol_frac
            * 0.888889
            * dtc_ratio_rootMeanFolded_dtc_ratio_tip
            * (nonfolded_wing_area**1.5)
            * (2.0 * nonfolded_taper_ratio + 1.0)
        ) / ((nonfolded_AR**0.5) * ((nonfolded_taper_ratio + 1.0) ** 2.0))
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.SPAN] = (
            den * dNum_dWingspan - num * dDen_dWingspan
        ) / den**2
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.AREA] = (
            den * dNum_dWingArea - num * dDen_dWingArea
        ) / den**2

        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.FOLDING_AREA] = (
            den * dNum_dWingFoldArea - num * dDen_dWingFoldArea
        ) / den**2

        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.TAPER_RATIO] = (
            den * dNum_dTaperRatio - num * dDen_dTaperRatio
        ) / den**2
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, wrt] = (den * dNum_dy - num * dDen_dy) / den**2


class BWBWingFoldVolume(om.ExplicitComponent):
    """
    Computation of wing tank fuel volume.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.CHOOSE_FOLD_LOCATION)

    def setup(self):
        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            self.add_input(
                'strut_y', val=25, units='ft', desc='YSTRUT: attachment location of strut'
            )
        else:
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN)

        self.add_input('wing_volume_no_fold', units='ft**3', desc='FVOLW_GEOMX')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        add_aviary_output(self, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, units='ft**3')

    def setup_partials(self):
        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            self.declare_partials(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 'strut_y')
        else:
            self.declare_partials(
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.FOLDED_SPAN
            )

        self.declare_partials(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
            [
                'wing_volume_no_fold',
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                Aircraft.Wing.SPAN,
            ],
        )

    def compute(self, inputs, outputs):
        wingspan = inputs[Aircraft.Wing.SPAN]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuselage_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        wing_volume_no_fold = inputs['wing_volume_no_fold']

        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            strut_span = inputs['strut_y']  # assume fold at strut location
            fold_span = strut_span * 2.0
        else:
            fold_span = inputs[Aircraft.Wing.FOLDED_SPAN]
        location = fold_span / 2.0

        tc_ratio_fold = tc_ratio_root + 2.0 * location * (tc_ratio_tip - tc_ratio_root) / wingspan
        tc_fac = (1.0 + tc_ratio_fold / tc_ratio_root) / (1.0 + tc_ratio_tip / tc_ratio_root)
        geometric_fuel_vol = (
            tc_fac
            * (fold_span - fuselage_width)
            / (wingspan - fuselage_width)
            * wing_volume_no_fold
        )
        outputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = geometric_fuel_vol

    def compute_partials(self, inputs, J):
        wingspan = inputs[Aircraft.Wing.SPAN]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        tc_ratio_tip = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_TIP]
        fuselage_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        wing_volume_no_fold = inputs['wing_volume_no_fold']

        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            strut_span = inputs['strut_y']
            fold_span = strut_span * 2.0
            wrt = 'strut_y'
            dLoc_dspan = 1
        else:
            fold_span = inputs[Aircraft.Wing.FOLDED_SPAN]
            wrt = Aircraft.Wing.FOLDED_SPAN
            dLoc_dspan = 1 / 2
        location = fold_span / 2.0
        dlocation_wrt = dLoc_dspan

        tc_ratio_fold = tc_ratio_root + 2.0 * location * (tc_ratio_tip - tc_ratio_root) / wingspan
        dtc_ratio_fold_wrt = 2.0 * dlocation_wrt * (tc_ratio_tip - tc_ratio_root) / wingspan
        dtc_ratio_fold_dwingspan = -2.0 * location * (tc_ratio_tip - tc_ratio_root) / wingspan**2
        dtc_ratio_fold_dtc_ratio_root = 1 - 2.0 * location / wingspan
        dtc_ratio_fold_dtc_ratio_tip = 2.0 * location / wingspan

        u = 1.0 + tc_ratio_fold / tc_ratio_root
        v = 1.0 + tc_ratio_tip / tc_ratio_root
        tc_fac = (1.0 + tc_ratio_fold / tc_ratio_root) / (1.0 + tc_ratio_tip / tc_ratio_root)
        dtc_fac_wrt = dtc_ratio_fold_wrt / tc_ratio_root / (1.0 + tc_ratio_tip / tc_ratio_root)
        dtc_fac_dwingspan = (
            dtc_ratio_fold_dwingspan / tc_ratio_root / (1.0 + tc_ratio_tip / tc_ratio_root)
        )

        du_tip = dtc_ratio_fold_dtc_ratio_tip / tc_ratio_root
        dv_tip = 1.0 / tc_ratio_root
        dtc_fac_dtc_ratio_tip = (du_tip * v - u * dv_tip) / v**2
        du_root = (dtc_ratio_fold_dtc_ratio_root * tc_ratio_root - tc_ratio_fold) / tc_ratio_root**2
        dv_root = -tc_ratio_tip / tc_ratio_root**2
        dtc_fac_dtc_ratio_root = (du_root * v - u * dv_root) / v**2

        dgeometric_fuel_vol_dtc_ratio_root = (
            dtc_fac_dtc_ratio_root
            * (fold_span - fuselage_width)
            / (wingspan - fuselage_width)
            * wing_volume_no_fold
        )
        dgeometric_fuel_vol_dtc_ratio_tip = (
            dtc_fac_dtc_ratio_tip
            * (fold_span - fuselage_width)
            / (wingspan - fuselage_width)
            * wing_volume_no_fold
        )
        dgeometric_fuel_vol_dwingspan = (
            dtc_fac_dwingspan
            * (fold_span - fuselage_width)
            / (wingspan - fuselage_width)
            * wing_volume_no_fold
            - tc_fac
            * (fold_span - fuselage_width)
            / (wingspan - fuselage_width) ** 2
            * wing_volume_no_fold
        )
        dgeometric_fuel_vol_wrt = (
            dtc_fac_wrt
            * (fold_span - fuselage_width)
            / (wingspan - fuselage_width)
            * wing_volume_no_fold
            + tc_fac * 2 * dlocation_wrt / (wingspan - fuselage_width) * wing_volume_no_fold
        )
        dgeometric_fuel_vol_dwing_volume_no_fold = (
            tc_fac * (fold_span - fuselage_width) / (wingspan - fuselage_width)
        )
        dgeometric_fuel_vol_dfuselage_width = (
            -tc_fac
            * (wingspan - fold_span)
            / (wingspan - fuselage_width) ** 2
            * wing_volume_no_fold
        )

        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 'wing_volume_no_fold'] = (
            dgeometric_fuel_vol_dwing_volume_no_fold
        )
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Fuselage.AVG_DIAMETER] = (
            dgeometric_fuel_vol_dfuselage_width
        )
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, wrt] = dgeometric_fuel_vol_wrt
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.SPAN] = (
            dgeometric_fuel_vol_dwingspan
        )

        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
            dgeometric_fuel_vol_dtc_ratio_root
        )
        J[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = (
            dgeometric_fuel_vol_dtc_ratio_tip
        )


class WingGroup(om.Group):
    """Group of WingSize, WingParameters, and WingFold for wing parameter computations."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.CHOOSE_FOLD_LOCATION)
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):
        has_fold = self.options[Aircraft.Wing.HAS_FOLD]
        has_strut = self.options[Aircraft.Wing.HAS_STRUT]

        self.add_subsystem(
            'size',
            WingSize(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )

        if has_fold or has_strut:
            self.add_subsystem(
                'dimensionless_calcs',
                DimensionalNonDimensionalInterchange(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

        self.add_subsystem(
            'parameters',
            WingParameters(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )

        if not has_fold:
            self.add_subsystem(
                'wing_vol',
                WingVolume(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

        if has_strut:
            self.add_subsystem(
                'strut',
                StrutGeom(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

        if has_fold:
            self.add_subsystem(
                'fold_area',
                WingFoldArea(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )
            self.add_subsystem(
                'fold_vol',
                WingFoldVolume(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

            choose_fold_location = self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]
            if not choose_fold_location:
                check_fold_location_definition(choose_fold_location, has_strut)
                self.promotes('strut', outputs=['strut_y'])
                self.promotes('fold_area', inputs=['strut_y'])
                self.promotes('fold_vol', inputs=['strut_y'])


class BWBWingGroup(om.Group):
    """
    Group of WingSize, WingParameters, WingFoldArea and BWBWingVolumeArea for wing parameter computations.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.CHOOSE_FOLD_LOCATION)
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        verbosity = self.options[Settings.VERBOSITY]
        has_fold = self.options[Aircraft.Wing.HAS_FOLD]
        has_strut = self.options[Aircraft.Wing.HAS_STRUT]

        if has_strut:
            if verbosity >= 1:
                print('BWB does not have strut implemented.')
        if not self.options[Aircraft.Wing.CHOOSE_FOLD_LOCATION]:
            raise ('There is no strut. Aircraft.Wing.CHOOSE_FOLD_LOCATION must be True.')

        self.add_subsystem(
            'size',
            WingSize(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )

        if has_fold:
            self.add_subsystem(
                'dimensionless_calcs',
                DimensionalNonDimensionalInterchange(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

        self.add_subsystem(
            'parameters',
            WingParameters(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )

        self.add_subsystem(
            'wing_vol',
            BWBWingVolume(),
            promotes_inputs=['aircraft:*'],
        )
        if has_fold:
            self.promotes('wing_vol', outputs=['wing_volume_no_fold'])
        else:
            self.promotes('wing_vol', outputs=['aircraft:*'])

        if has_fold:
            self.add_subsystem(
                'fold_area',
                WingFoldArea(),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )
            self.add_subsystem(
                'fold_vol',
                BWBWingFoldVolume(),
                promotes_inputs=['aircraft:*', 'wing_volume_no_fold'],
                promotes_outputs=['aircraft:*'],
            )

        self.add_subsystem(
            'exposed_wing',
            ExposedWing(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )


epsilon = 0.05


def f(x):
    """Valid for x in [0.0, 1.0]."""
    diff = 0.5 - x
    y = np.sqrt(0.25 - diff**2)
    return y


def df(x):
    """First derivative of f(x), valid for x in (0.0, 1.0)."""
    diff = 0.5 - x
    dy = (0.5 - x) / np.sqrt(0.25 - diff**2)
    return dy


def d2f(x):
    """Second derivative of f(x), valid for x in (0.0, 1.0)."""
    diff = 0.5 - x
    d2y = -0.25 / np.sqrt(0.25 - diff**2) ** 3
    return d2y


def g1(x):
    """
    Returns a cubic function g1(x) such that:
    g1(0.0) = 0.0
    g1(ε) = f(ε)
    g1'(ε) = f'(ε)
    g1"(ε) = f"(ε).
    """
    A1 = f(epsilon)
    B1 = df(epsilon)
    C1 = d2f(epsilon)
    d1 = (A1 - epsilon * B1 + 0.5 * epsilon**2 * C1) / epsilon**3
    c1 = (C1 - 6.0 * d1 * epsilon) / 2.0
    b1 = B1 - epsilon * C1 + 3 * d1 * epsilon**2
    a1 = 0.0
    y = a1 + b1 * x + c1 * x**2 + d1 * x**3
    return y


def dg1(x):
    """First derivative of g1(x)."""
    A1 = f(epsilon)
    B1 = df(epsilon)
    C1 = d2f(epsilon)
    d1 = (A1 - epsilon * B1 + 0.5 * epsilon**2 * C1) / epsilon**3
    c1 = (C1 - 6.0 * d1 * epsilon) / 2.0
    b1 = B1 - epsilon * C1 + 3.0 * d1 * epsilon**2
    dy = b1 + 2.0 * c1 * x + 3.0 * d1 * x**2
    return dy


def g2(x):
    """
    Returns a cubic function g2(x) such that:
    g2(1.0) = 0.0
    g2(ε) = f(1.0-ε)
    g2'(ε) = f'(1.0-ε)
    g2"(ε) = f"(1.0-ε).
    """
    delta = 1.0 - epsilon
    A2 = f(delta)
    B2 = df(delta)
    C2 = d2f(delta)
    d2 = -(A2 + B2 * epsilon + 0.5 * C2 * epsilon**2) / epsilon**3
    c2 = (C2 - 6.0 * d2 * delta) / 2.0
    b2 = B2 - C2 * delta + 3.0 * d2 * delta**2
    a2 = -(b2 + c2 + d2)
    y = a2 + b2 * x + c2 * x**2 + d2 * x**3
    return y


def dg2(x):
    """First derivative of g2(x)."""
    delta = 1.0 - epsilon
    A2 = f(delta)
    B2 = df(delta)
    C2 = d2f(delta)
    d2 = -(A2 + B2 * epsilon + 0.5 * C2 * epsilon**2) / epsilon**3
    c2 = (C2 - 6 * d2 * delta) / 2.0
    b2 = B2 - C2 * delta + 3.0 * d2 * delta**2
    dy = b2 + 2.0 * c2 * x + 3.0 * d2 * x**2
    return dy


class ExposedWing(om.ExplicitComponent):
    """
    Computation of exposed wing area. This is useful for BWB,
    but is available to tube + wing model too.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.TYPE)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(
            self, Aircraft.Wing.VERTICAL_MOUNT_LOCATION, units='unitless', desc='HWING'
        )
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft', desc='SWF')
        add_aviary_input(
            self, Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, units='unitless', desc='HGTqWID'
        )
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft', desc='B')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless', desc='SLM')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2', desc='SW')

        add_aviary_output(self, Aircraft.Wing.EXPOSED_AREA, units='ft**2', desc='SW_EXP')

    def setup_partials(self):
        design_type = self.options[Aircraft.Design.TYPE]

        self.declare_partials(
            Aircraft.Wing.EXPOSED_AREA,
            [
                Aircraft.Wing.VERTICAL_MOUNT_LOCATION,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.AREA,
            ],
        )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.declare_partials(
                Aircraft.Wing.EXPOSED_AREA,
                [
                    Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO,
                ],
            )

    def compute(self, inputs, outputs):
        design_type = self.options[Aircraft.Design.TYPE]
        verbosity = self.options[Settings.VERBOSITY]

        h_wing = inputs[Aircraft.Wing.VERTICAL_MOUNT_LOCATION]
        body_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]

        if h_wing >= epsilon and h_wing <= 1.0 - epsilon:
            sqt = np.sqrt(0.25 - (0.5 - h_wing) ** 2)
        elif h_wing >= 0.0 and h_wing < epsilon:
            sqt = g1(h_wing)
        elif h_wing <= 1.0 and h_wing > 1.0 - epsilon:
            sqt = g2(h_wing)
        else:
            raise 'The given parameter Aircraft.Wing.VERTICAL_MOUNT_LOCATION is out of range.'

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            cabin_height = body_width * inputs[Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO]
            b_fus = 0.5 * (body_width - cabin_height) + cabin_height * sqt
        else:
            b_fus = body_width * sqt

        wingspan = inputs[Aircraft.Wing.SPAN]
        wing_area = inputs[Aircraft.Wing.AREA]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        if wingspan <= 0:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Wing.SPAN must be positive.')
        if taper_ratio <= 0:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Wing.TAPER_RATIO must be positive.')

        root_chord_wing = 2.0 * wing_area / (wingspan * (1.0 + taper_ratio))
        tip_chord = taper_ratio * root_chord_wing
        c_fus = root_chord_wing + 2.0 * b_fus * (tip_chord - root_chord_wing) / wingspan

        exp_wing_area = (wingspan / 2.0 - b_fus) * (c_fus + tip_chord)
        outputs[Aircraft.Wing.EXPOSED_AREA] = exp_wing_area

    def compute_partials(self, inputs, J):
        design_type = self.options[Aircraft.Design.TYPE]

        h_wing = inputs[Aircraft.Wing.VERTICAL_MOUNT_LOCATION]
        body_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        height_to_width = inputs[Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO]

        if h_wing >= epsilon and h_wing <= 1.0 - epsilon:
            sqt = np.sqrt(0.25 - (0.5 - h_wing) ** 2)
            d_sqt = df(h_wing)
        elif h_wing >= 0.0 and h_wing < epsilon:
            sqt = g1(h_wing)
            d_sqt = dg1(h_wing)
        elif h_wing <= 1.0 and h_wing > 1.0 - epsilon:
            sqt = g2(h_wing)
            d_sqt = dg2(h_wing)
        else:
            raise 'The given parameter Aircraft.Wing.VERTICAL_MOUNT_LOCATION is out of range.'

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            cabin_height = body_width * height_to_width
            b_fus = 0.5 * (body_width - cabin_height) + cabin_height * sqt
        else:
            b_fus = body_width * sqt

        wingspan = inputs[Aircraft.Wing.SPAN]
        wing_area = inputs[Aircraft.Wing.AREA]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]

        root_chord_wing = 2.0 * wing_area / (wingspan * (1.0 + taper_ratio))
        tip_chord = taper_ratio * root_chord_wing
        c_fus = root_chord_wing + 2.0 * b_fus * (tip_chord - root_chord_wing) / wingspan

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            d_b_fus_d_hwing = body_width * height_to_width * d_sqt
            d_b_fus_d_body_width = 0.5 * (1.0 - height_to_width) + height_to_width * sqt
            d_b_fus_d_height_to_width = -0.5 * body_width + body_width * sqt
            d_c_fus_d_height_to_width = (
                4
                * d_b_fus_d_height_to_width
                * (taper_ratio - 1.0)
                / (taper_ratio + 1.0)
                * wing_area
                / wingspan**2
            )

            d_exp_area_d_height_to_width = (
                -d_b_fus_d_height_to_width * c_fus
                + (wingspan * 0.5 - b_fus) * d_c_fus_d_height_to_width
                - d_b_fus_d_height_to_width
                * 2.0
                * wing_area
                * taper_ratio
                / wingspan
                / (1.0 + taper_ratio)
            )
            J[Aircraft.Wing.EXPOSED_AREA, Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO] = (
                d_exp_area_d_height_to_width
            )

        else:
            d_b_fus_d_hwing = body_width * d_sqt
            d_b_fus_d_body_width = sqt

        d_c_fus_d_hwing = (
            4.0
            * d_b_fus_d_hwing
            * (taper_ratio - 1.0)
            / (taper_ratio + 1.0)
            * wing_area
            / wingspan**2
        )
        d_c_fus_d_body_width = (
            4.0
            * d_b_fus_d_body_width
            * (taper_ratio - 1.0)
            / (taper_ratio + 1.0)
            * wing_area
            / wingspan**2
        )
        d_c_fus_d_wing_area = (
            2.0
            / (1.0 + taper_ratio)
            * (1.0 / wingspan + b_fus * 2 * (taper_ratio - 1.0) / wingspan**2)
        )
        d_c_fus_d_wingspan = (
            -2.0
            * wing_area
            / (1.0 + taper_ratio)
            * (1.0 / wingspan**2 + b_fus * 4.0 * (taper_ratio - 1.0) / wingspan**3)
        )
        d_c_fus_d_taper_ratio = (
            -2.0 * wing_area / (1.0 + taper_ratio) ** 2 / wingspan
            + b_fus * 8.0 * wing_area / wingspan**2 / (taper_ratio + 1) ** 2
        )

        d_exp_area_d_hwing = (
            -d_b_fus_d_hwing * c_fus
            + (wingspan * 0.5 - b_fus) * d_c_fus_d_hwing
            - d_b_fus_d_hwing * 2.0 * wing_area * taper_ratio / wingspan / (1.0 + taper_ratio)
        )
        J[Aircraft.Wing.EXPOSED_AREA, Aircraft.Wing.VERTICAL_MOUNT_LOCATION] = d_exp_area_d_hwing

        d_exp_area_d_body_width = (
            -d_b_fus_d_body_width * c_fus
            + (wingspan * 0.5 - b_fus) * d_c_fus_d_body_width
            - d_b_fus_d_body_width * 2.0 * wing_area * taper_ratio / wingspan / (1 + taper_ratio)
        )
        J[Aircraft.Wing.EXPOSED_AREA, Aircraft.Fuselage.AVG_DIAMETER] = d_exp_area_d_body_width

        d_exp_area_d_wingspan = (
            0.5 * c_fus
            + (0.5 * wingspan - b_fus) * d_c_fus_d_wingspan
            + b_fus * 2.0 * wing_area * taper_ratio / wingspan**2 / (1.0 + taper_ratio)
        )
        J[Aircraft.Wing.EXPOSED_AREA, Aircraft.Wing.SPAN] = d_exp_area_d_wingspan

        d_exp_area_d_taper_ratio = (
            (0.5 * wingspan - b_fus) * d_c_fus_d_taper_ratio
            + wing_area / (1.0 + taper_ratio) ** 2
            - b_fus * 2.0 * wing_area / wingspan / (1.0 + taper_ratio) ** 2
        )
        J[Aircraft.Wing.EXPOSED_AREA, Aircraft.Wing.TAPER_RATIO] = d_exp_area_d_taper_ratio

        d_exp_area_d_wing_area = (
            (0.5 * wingspan - b_fus) * d_c_fus_d_wing_area
            + taper_ratio / (1.0 + taper_ratio)
            - b_fus * 2.0 * taper_ratio / wingspan / (1.0 + taper_ratio)
        )
        J[Aircraft.Wing.EXPOSED_AREA, Aircraft.Wing.AREA] = d_exp_area_d_wing_area
