import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class BasicFlapsCalculations(om.ExplicitComponent):
    """
    Intermediate calculations for flaps model of GASP-based aerodynamics
    """

    def setup(self):

        # inputs

        add_aviary_input(self, Aircraft.Wing.SWEEP, val=25.0)
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=10.13)
        add_aviary_input(self, Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, val=0.33)
        add_aviary_input(self, Aircraft.Wing.CENTER_CHORD, val=17.48974)
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, val=13.1)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15)
        add_aviary_input(self, Aircraft.Wing.SPAN, val=117.8)
        add_aviary_input(self, Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.add_input(
            "slat_defl",
            val=10.0,
            units="deg",
            desc="DELLED: leading edge slat deflection",
        )
        add_aviary_input(self, Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION, val=20.0)
        self.add_input(
            "flap_defl",
            val=10.0,
            units="deg",
            desc="DFLPTO | DFLPLD: takeoff or landing flap deflection",
        )
        add_aviary_input(self, Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION, val=55.0)
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, val=16.406626)
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, val=129.4)
        add_aviary_input(self, Aircraft.Wing.LEADING_EDGE_SWEEP,
                         val=0.47639, units="rad")

        # outputs

        self.add_output(
            "VLAM8",
            val=0.74444322,
            units='unitless',
            desc="VLAM8: sensitivity of flap clean wing maximum lift coefficient to wing sweep angle",
        )
        self.add_output(
            "VDEL4",
            val=0.93577864,
            units='unitless',
            desc="VDEL4: sensitivity of flap minimum drag coefficient to flap hinge line sweep",
        )
        self.add_output(
            "VDEL5",
            val=0.90759603,
            units='unitless',
            desc="VDEL5: sensitivity of minimum drag coefficient to fuselage width to span ratio",
        )
        self.add_output(
            "VLAM9",
            val=0.9975,
            units='unitless',
            desc="VLAM9: sensitivity of slat clean wing maximum lift coefficient to slat chord",
        )
        self.add_output(
            "slat_defl_ratio",
            val=0.5,
            units='unitless',
            desc="RDELL: ratio of leading edge slat deflection to optimum deflection angle",
        )
        self.add_output(
            "flap_defl_ratio",
            val=0.5,
            units='unitless',
            desc="RDELF: ratio of trailing edge flap deflection to optimum deflection angle",
        )
        add_aviary_output(self, Aircraft.Wing.SLAT_SPAN_RATIO, 0.89759603)
        self.add_output(
            "chord_to_body_ratio",
            val=0.12679,
            units='unitless',
            desc="CROBL: ratio of root chord to fuselage length",
        )
        self.add_output(
            "body_to_span_ratio",
            val=0.09240397,
            units='unitless',
            desc="FWOB: ratio of body diameter at quarter chord to wing span",
        )
        self.add_output(
            "VLAM12",
            val=0.79207395,
            units='unitless',
            desc="VLAM12: sensitivity of slat clean wing maximum lift coefficient to leading edge sweepback",
        )

    def setup_partials(self):

        # output partials
        self.declare_partials(
            "VLAM8", [Aircraft.Wing.SWEEP], dependent=True, method="cs", step=1e-8
        )
        self.declare_partials(
            "VDEL4",
            [
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.FLAP_CHORD_RATIO,
                Aircraft.Wing.TAPER_RATIO,
            ],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            "VDEL5",
            [
                Aircraft.Wing.SPAN,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.CENTER_CHORD,
                Aircraft.Fuselage.AVG_DIAMETER,
            ],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            "VLAM9", [Aircraft.Wing.SLAT_CHORD_RATIO], dependent=True, method="cs", step=1e-8
        )
        self.declare_partials(
            "slat_defl_ratio",
            ["slat_defl", Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            "flap_defl_ratio", ["flap_defl", Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION], method="cs"
        )
        self.declare_partials(
            Aircraft.Wing.SLAT_SPAN_RATIO,
            [
                Aircraft.Wing.SPAN,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.CENTER_CHORD,
                Aircraft.Fuselage.AVG_DIAMETER,
            ],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            "chord_to_body_ratio",
            [Aircraft.Wing.ROOT_CHORD, Aircraft.Fuselage.LENGTH],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            "body_to_span_ratio",
            [
                Aircraft.Wing.SPAN,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.CENTER_CHORD,
                Aircraft.Fuselage.AVG_DIAMETER,
            ],
            dependent=True,
            method="cs",
            step=1e-8,
        )
        self.declare_partials(
            "VLAM12",
            [Aircraft.Wing.LEADING_EDGE_SWEEP],
            dependent=True,
            method="cs",
            step=1e-8,
        )

    def compute(self, inputs, outputs):

        sweep_c4 = inputs[Aircraft.Wing.SWEEP]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        flap_chord_ratio = inputs[Aircraft.Wing.FLAP_CHORD_RATIO]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        center_chord = inputs[Aircraft.Wing.CENTER_CHORD]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        wingspan = inputs[Aircraft.Wing.SPAN]
        slat_chord_ratio = inputs[Aircraft.Wing.SLAT_CHORD_RATIO]
        slat_defl = inputs["slat_defl"]
        optimum_slat_defl = inputs[Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION]
        flap_defl = inputs["flap_defl"]
        optimum_flap_defl = inputs[Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION]
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        sweep_LE = inputs[Aircraft.Wing.LEADING_EDGE_SWEEP]

        # intermediate values equations
        RLMC4 = sweep_c4 * 0.017453

        TSWPFH = (np.tan(RLMC4)) - (4.0 / AR) * (
            (0.75 - flap_chord_ratio) * (1.0 - taper_ratio) / (1.0 + taper_ratio)
        )
        SWPFHL = np.arctan(TSWPFH)

        DBALE = (
            2.0
            * (
                tc_ratio_root
                * center_chord
                * (cabin_width - (tc_ratio_root * center_chord))
            )
            ** 0.5
        ) + 0.4

        SWPL12 = sweep_LE - 5.0 / 57.296

        # outputs equations
        outputs["VLAM8"] = VLAM8 = (np.cos(RLMC4)) ** 3
        outputs["body_to_span_ratio"] = body_to_span_ratio = DBALE / wingspan
        outputs["VDEL4"] = VDEL4 = np.cos(SWPFHL)
        outputs["VDEL5"] = VDEL5 = 1.0 - body_to_span_ratio
        outputs["VLAM9"] = VLAM9 = 6.65 * slat_chord_ratio
        outputs["slat_defl_ratio"] = slat_defl_ratio = slat_defl / optimum_slat_defl
        outputs["flap_defl_ratio"] = flap_defl / optimum_flap_defl
        outputs[Aircraft.Wing.SLAT_SPAN_RATIO] = slat_span_ratio = 0.99 - DBALE / wingspan
        outputs["chord_to_body_ratio"] = chord_to_body_ratio = root_chord / fus_len
        outputs["VLAM12"] = VLAM12 = (np.cos(SWPL12)) ** 3
