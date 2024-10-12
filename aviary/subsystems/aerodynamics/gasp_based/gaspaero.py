import numpy as np
import warnings

import openmdao.api as om
from openmdao.utils import cs_safe as cs

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.aerodynamics.gasp_based.common import (AeroForces,
                                                              CLFromLift,
                                                              TanhRampComp)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.utils.aviary_values import AviaryValues
from aviary.subsystems.aerodynamics.gasp_based.interference import WingFuselageInterferenceMission


#
# data from EAERO
#
sig1 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.1833, 0.1621, 0.1429, 0.1256, 0.1101, 0.0966],
        [0.4, 0.3600, 0.3186, 0.2801, 0.2454, 0.2147, 0.1879],
        [0.6, 0.5319, 0.4654, 0.4053, 0.3526, 0.3070, 0.2681],
        [0.8, 0.6896, 0.5900, 0.5063, 0.4368, 0.3791, 0.3309],
        [1.0, 0.7857, 0.6575, 0.5613, 0.4850, 0.4228, 0.3712],
    ]
)
sig2 = np.array(
    [
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.04, 0.7971, 0.9314, 0.9722, 0.9874, 0.9939, 0.9969],
        [0.16, 0.7040, 0.8681, 0.9373, 0.9688, 0.9839, 0.9914],
        [0.36, 0.7476, 0.8767, 0.9363, 0.9659, 0.9812, 0.9893],
        [0.64, 0.8709, 0.9338, 0.9625, 0.9778, 0.9865, 0.9917],
        [1.0, 0.9852, 0.9852, 0.9880, 0.9910, 0.9935, 0.9954],
    ]
)
xbbar = np.linspace(0, 1.0, sig1.shape[0])
xhbar = np.linspace(0, 0.3, sig1.shape[1])

#
# data from DRAG
#
# flap deflection angles, deg
adelfd = np.array(
    [
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        38.0,
        40.0,
        42.0,
        44.0,
        50.0,
        55.0,
        60.0,
    ]
)
# flap angle correction of oswald efficiency factor
adel6 = np.array(
    [
        1.0,
        0.995,
        0.99,
        0.98,
        0.97,
        0.955,
        0.935,
        0.90,
        0.875,
        0.855,
        0.83,
        0.80,
        0.70,
        0.54,
        0.30,
    ]
)
# induced drag correction factors
asigma = np.array(
    [
        0.0,
        0.16,
        0.285,
        0.375,
        0.435,
        0.48,
        0.52,
        0.55,
        0.575,
        0.58,
        0.59,
        0.60,
        0.62,
        0.635,
        0.65,
    ]
)


def deg2rad(d):
    """Complex step safe deg2rad"""
    return d * np.pi / 180.0


def rad2deg(r):
    """Complex step safe rad2deg"""
    return r * 180.0 / np.pi


def cla(ar, sweep, mach):
    """Lift-curve slope of 3D wings from Seckel equation

    Parameters
    ----------
    ar : float
        Aspect ratio
    sweep : float
        Quarter-chord sweep angle, in radians
    mach : float
        Mach number.
    """
    return (
        np.pi
        * ar
        / (
            1
            + np.sqrt(
                1
                + (
                    ((ar / (2 * np.cos(sweep))) ** 2)
                    * (1 - (mach * np.cos(sweep)) ** 2)
                )
            )
        )
    )


def sigmoid(x, x0, alpha=0.1):
    """Sigmoid used to smoothly transition between piecewise functions"""
    if alpha == 0:
        raise ValueError("alpha must be non-zero")
    return 1 / (1 + np.exp(-(x - x0) / alpha))


class WingTailRatios(om.ExplicitComponent):
    """Pre-mission calculation of ratios between tail and wing parameters"""

    def setup(self):

        add_aviary_input(self, Aircraft.Wing.AREA, val=1370.3)

        add_aviary_input(self, Aircraft.Wing.SPAN, val=0.0)

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, val=0.33)

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15)

        add_aviary_input(self, Aircraft.Wing.MOUNTING_TYPE, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.SPAN, val=0.0)

        add_aviary_input(self, Aircraft.VerticalTail.SPAN, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.AREA, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.AVERAGE_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, val=0.0)

        self.add_output(
            "hbar", val=0.0, units="unitless",
            desc="HBAR: Ratio of HGAP(?) to wing span")
        self.add_output(
            "bbar", units="unitless", desc="BBAR: Ratio of H tail area to wing area")
        self.add_output(
            "sbar", units="unitless", desc="SBAR: Ratio of H tail area to wing area")
        self.add_output(
            "cbar", units="unitless", desc="SBAR: Ratio of H tail chord to wing chord")

    def setup_partials(self):
        self.declare_partials(
            "hbar",
            [
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.VerticalTail.SPAN,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.MOUNTING_TYPE,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
            ],
            method="cs",
        )
        self.declare_partials(
            "bbar", [Aircraft.HorizontalTail.SPAN, Aircraft.Wing.SPAN], method="cs"
        )
        self.declare_partials(
            "sbar", [Aircraft.HorizontalTail.AREA, Aircraft.Wing.AREA], method="cs"
        )
        self.declare_partials(
            "cbar",
            [Aircraft.HorizontalTail.AVERAGE_CHORD, Aircraft.Wing.AVERAGE_CHORD],
            method="cs",
        )

    def compute(self, inputs, outputs):
        (
            wing_area,
            wingspan,
            avg_chord,
            taper_ratio,
            tc_ratio_root,
            wing_loc,
            htail_loc,
            span_htail,
            span_vtail,
            htail_area,
            htail_chord,
            cabin_width,
        ) = inputs.values()

        trtw = tc_ratio_root * 2 * wing_area / wingspan / (1 + taper_ratio)
        hgap = cs.abs(
            htail_loc * span_vtail - 0.5 * (cabin_width - trtw) * (2 * wing_loc - 1)
        )
        outputs["hbar"] = hgap / wingspan
        outputs["bbar"] = span_htail / wingspan
        outputs["sbar"] = htail_area / wing_area
        outputs["cbar"] = htail_chord / avg_chord


class Xlifts(om.ExplicitComponent):
    """Compute lift ratio and lift-curve slope for given stability margin"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # mission inputs
        self.add_input(Dynamic.Mission.MACH, val=0.0, units="unitless",
                       shape=nn, desc="Mach number")

        # stability inputs

        add_aviary_input(self, Aircraft.Design.STATIC_MARGIN, val=0.03)

        add_aviary_input(self, Aircraft.Design.CG_DELTA, val=0.25)

        # geometry inputs

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=10.13)

        add_aviary_input(self, Aircraft.Wing.SWEEP, val=25.0)

        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.SWEEP, val=25.0)

        add_aviary_input(self, Aircraft.HorizontalTail.MOMENT_RATIO, val=0.0)

        # geometry from wing-tail ratios
        self.add_input(
            "sbar", units="unitless", desc="SBAR: Ratio of H tail area to wing area")
        self.add_input(
            "cbar", units="unitless", desc="CBAR: Ratio of H tail chord to wing chord")
        self.add_input(
            "hbar", units="unitless", desc="HBAR: Ratio of HGAP(?) to wing span")
        self.add_input(
            "bbar", units="unitless", desc="BBAR: Ratio of H tail area to wing area")

        self.add_output("lift_curve_slope", units="unitless",
                        shape=nn, desc="Lift-curve slope")
        self.add_output("lift_ratio", units="unitless", shape=nn, desc="Lift ratio")

    def setup_partials(self):
        ar = np.arange(self.options["num_nodes"])

        self.declare_partials("lift_ratio", "*", method="cs")
        self.declare_partials("lift_ratio", Dynamic.Mission.MACH,
                              rows=ar, cols=ar, method="cs")
        self.declare_partials("lift_curve_slope", "*", method="cs")
        self.declare_partials(
            "lift_curve_slope",
            [
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.HorizontalTail.SWEEP,
                Aircraft.HorizontalTail.MOMENT_RATIO,
                "sbar",
                "cbar",
                "hbar",
                "bbar",
            ],
            method="cs",
        )
        self.declare_partials("lift_curve_slope", Dynamic.Mission.MACH,
                              rows=ar, cols=ar, method="cs")

    def compute(self, inputs, outputs):
        (
            mach,
            static_margin,
            delta_cg,
            AR,
            sweep_c4,
            htail_loc,
            htail_sweep,
            h_tail_moment,
            sbar,
            cbar,
            hbar,
            bbar,
        ) = inputs.values()

        delta = (static_margin + delta_cg) * h_tail_moment

        # TODO handle xt < 0?
        xt = 1 / h_tail_moment

        art = AR * bbar**2 / sbar
        h = hbar * AR

        # stability contribution from each surface
        claw0 = cla(AR, deg2rad(sweep_c4), mach)
        clat0 = cla(art, deg2rad(htail_sweep), mach) * (0.9 + 0.1 * htail_loc)

        # Hayes reverse flow theorem to estimate downwash effects on wing and canard
        eps1 = 1 / (4 * np.pi * np.sqrt(xt**2 + h**2))
        eps2 = 1 / np.pi / AR
        eps3 = cs.abs(xt) / (np.pi * AR * np.sqrt(xt**2 + h**2 + AR**2 / 4))
        eps4 = 1 / np.pi / art
        eps5 = cs.abs(xt) / (
            np.pi * art * np.sqrt(xt**2 + h**2 + art**2 * cbar**2 / 4)
        )

        claw = (
            claw0
            * (1 - clat0 * (eps4 - eps5 - cbar * eps1))
            / (1 - clat0 * claw0 * (eps1 + eps2 + eps3) * (eps4 - eps5 - cbar * eps1))
        )

        clat = clat0 * (1 - claw * (eps1 + eps2 + eps3))

        abar = clat / claw
        c = 1 / (1 + 1 / abar / sbar)
        lift_ratio = (c - delta) / (1 + delta - c)

        outputs["lift_curve_slope"] = claw
        outputs["lift_ratio"] = lift_ratio


class AeroGeom(om.ExplicitComponent):
    """Compute drag parameters from cruise conditions and geometric parameters.

    This corresponds to the AERO subroutine in GASP. The primary outputs are parameters
    SA* which build up the total aircraft drag coefficient.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        nn = self.options["num_nodes"]
        num_engine_type = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))

        self.add_input(
            Dynamic.Mission.MACH, val=0.0, units="unitless", shape=nn, desc="Current Mach number")
        self.add_input(
            Dynamic.Mission.SPEED_OF_SOUND,
            val=1.0,
            units="ft/s",
            shape=nn,
            desc="Speed of sound at current altitude",
        )
        self.add_input(
            Dynamic.Mission.KINEMATIC_VISCOSITY,
            val=1.0,
            units="ft**2/s",
            shape=nn,
            desc="Kinematic viscosity at current altitude",
        )

        self.add_input("ufac", units="unitless", shape=nn, desc="UFAC")

        # form factors
        # user could input these directly or use functions to estimate from geometry

        add_aviary_input(self, Aircraft.Wing.FORM_FACTOR, val=1.25)

        add_aviary_input(self, Aircraft.Fuselage.FORM_FACTOR, val=1.25)

        add_aviary_input(self, Aircraft.Nacelle.FORM_FACTOR,
                         val=np.full(num_engine_type, 1.5))

        add_aviary_input(self, Aircraft.VerticalTail.FORM_FACTOR, val=1.25)

        add_aviary_input(self, Aircraft.HorizontalTail.FORM_FACTOR, val=1.25)

        add_aviary_input(self, Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, val=1.1)

        add_aviary_input(self, Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR, val=0.0)

        # miscellaneous top-level inputs

        add_aviary_input(self, Aircraft.Design.DRAG_COEFFICIENT_INCREMENT, val=0.00175)

        add_aviary_input(self, Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT, val=0.25)

        add_aviary_input(self, Aircraft.Wing.MIN_PRESSURE_LOCATION, val=0.3)

        add_aviary_input(self, Aircraft.Wing.MAX_THICKNESS_LOCATION, val=0.4)

        # geometric user inputs

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=10.13)

        add_aviary_input(self, Aircraft.Wing.SWEEP, val=25.0)

        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, val=0.33)

        add_aviary_input(self, Aircraft.Strut.AREA_RATIO, val=0.0)

        # geometric data from sizing

        add_aviary_input(self, Aircraft.Wing.SPAN, val=0.0)

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.AVERAGE_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.VerticalTail.AVERAGE_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.Fuselage.LENGTH, val=0.0)

        add_aviary_input(self, Aircraft.Nacelle.AVG_LENGTH,
                         val=np.zeros(num_engine_type))

        add_aviary_input(self, Aircraft.HorizontalTail.AREA, val=0.0)

        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, val=0.0)

        add_aviary_input(self, Aircraft.Nacelle.SURFACE_AREA,
                         val=np.zeros(num_engine_type))

        add_aviary_input(self, Aircraft.Wing.AREA, val=1370.3)

        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, val=0.0)

        add_aviary_input(self, Aircraft.VerticalTail.AREA, val=0.0)

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, val=0.0)

        add_aviary_input(self, Aircraft.Strut.CHORD, val=0.0)

        self.add_input('interference_independent_of_shielded_area')
        self.add_input('drag_loss_due_to_shielded_wing_area')

        # outputs
        for i in range(7):
            name = f"SA{i+1}"
            self.add_output(name, units="unitless", shape=nn, desc=f"{name}: Drag param")

        self.add_output(
            "cf", units="unitless", shape=nn,
            desc="CFIN: Skin friction coefficient at Re=1e7"
        )

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials("*", "*", dependent=False)
        ar = np.arange(self.options["num_nodes"])

        self.declare_partials(
            "SA1",
            [
                Aircraft.Wing.MIN_PRESSURE_LOCATION,
                Aircraft.Wing.MAX_THICKNESS_LOCATION,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            ],
            method="cs",
        )
        self.declare_partials(
            "SA2",
            [
                Aircraft.Wing.MIN_PRESSURE_LOCATION,
                Aircraft.Wing.MAX_THICKNESS_LOCATION,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
            ],
            method="cs",
        )
        self.declare_partials(
            "SA3",
            [
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            ],
            method="cs",
        )
        self.declare_partials(
            "SA4", [Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], method="cs"
        )
        self.declare_partials("cf", [Dynamic.Mission.MACH],
                              rows=ar, cols=ar, method="cs")

        # diag partials for SA5-SA7
        self.declare_partials(
            "SA5", [Dynamic.Mission.MACH, Dynamic.Mission.SPEED_OF_SOUND,
                    Dynamic.Mission.KINEMATIC_VISCOSITY], rows=ar, cols=ar, method="cs"
        )
        self.declare_partials(
            "SA6", [Dynamic.Mission.MACH, Dynamic.Mission.SPEED_OF_SOUND,
                    Dynamic.Mission.KINEMATIC_VISCOSITY], rows=ar, cols=ar, method="cs"
        )
        self.declare_partials(
            "SA7", [Dynamic.Mission.MACH, Dynamic.Mission.SPEED_OF_SOUND,
                    Dynamic.Mission.KINEMATIC_VISCOSITY, "ufac"], rows=ar, cols=ar, method="cs"
        )

        # dense partials for SA5-SA7
        most_params = [
            Aircraft.Wing.FORM_FACTOR,
            Aircraft.Fuselage.FORM_FACTOR,
            Aircraft.Nacelle.FORM_FACTOR,
            Aircraft.VerticalTail.FORM_FACTOR,
            Aircraft.HorizontalTail.FORM_FACTOR,
            Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR,
            Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR,
            Aircraft.Design.DRAG_COEFFICIENT_INCREMENT,
            Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT,
            Aircraft.Wing.TAPER_RATIO,
            Aircraft.Strut.AREA_RATIO,
            Aircraft.Wing.SPAN,
            Aircraft.Wing.AVERAGE_CHORD,
            Aircraft.HorizontalTail.AVERAGE_CHORD,
            Aircraft.VerticalTail.AVERAGE_CHORD,
            Aircraft.Fuselage.LENGTH,
            Aircraft.Nacelle.AVG_LENGTH,
            Aircraft.HorizontalTail.AREA,
            Aircraft.Fuselage.WETTED_AREA,
            Aircraft.Nacelle.SURFACE_AREA,
            Aircraft.Wing.AREA,
            Aircraft.Fuselage.AVG_DIAMETER,
            Aircraft.VerticalTail.AREA,
            'interference_independent_of_shielded_area',
            'drag_loss_due_to_shielded_wing_area',
        ]
        self.declare_partials("SA5", most_params, method="cs")
        self.declare_partials(
            "SA6", [Aircraft.Wing.FORM_FACTOR, Aircraft.Wing.AVERAGE_CHORD], method="cs"
        )
        self.declare_partials(
            "SA7",
            most_params + [Aircraft.Wing.ASPECT_RATIO, Aircraft.Wing.SWEEP],
            method="cs",
        )

    def compute(self, inputs, outputs):
        (
            mach,
            sos,
            nu,
            ufac,
            ff_wing,
            ff_fus,
            ff_nac,
            ff_vtail,
            ff_htail,
            wing_fus_intf,
            strut_fus_intf,
            cd0_inc,
            fe_fus_inc,
            wing_min_pressure_loc,
            wing_max_thickness_loc,
            AR,
            sweep_c4,
            taper_ratio,
            strut_wing_area_ratio,
            wingspan,
            avg_chord,
            htail_chord,
            vtail_chord,
            fus_len,
            nac_len,
            htail_area,
            fus_SA,
            nacelle_area,
            wing_area,
            cabin_width,
            vtail_area,
            tc_ratio,
            strut_chord,
            feintwf,
            areashieldwf,
        ) = inputs.values()
        # skin friction coeff at Re = 10**7
        cf = 0.455 / 7**2.58 / (1 + 0.144 * mach**2) ** 0.65

        t = cs.abs(np.tan(deg2rad(sweep_c4)))
        yale05 = (1 - taper_ratio) / (1 + taper_ratio)
        # sweep angle to min pressure point
        dlmps = rad2deg(
            cs.arctan2(AR * t - 4 * (wing_min_pressure_loc - 0.25) * yale05, AR)
        )
        # sweep angle to max thickness point
        dlmtcx = rad2deg(
            cs.arctan2(AR * t - 4 * (wing_max_thickness_loc - 0.25) * yale05, AR)
        )
        # sweep angle of the leading edge
        rlmle = cs.arctan2(AR * t + yale05, AR)
        fk = 1 / (1 + yale05 / AR * 4 * taper_ratio**2)

        # Reynolds number per foot
        # here we make a smooth transition between a minimum reli (approximately
        # corresponding to Mach 0.1 at SLS) to help with takeoff. GASP doesn't call AERO
        # before takeoff, so the RELI used corresponds to the cruise point, and this
        # isn't a problem.
        reli_y1 = 700000 * np.ones(self.options["num_nodes"])
        reli_y2 = sos * mach / nu
        sig = sigmoid(mach, 0.1, alpha=0.005)
        reli = (1 - sig) * reli_y1 + sig * reli_y2

        # Re correction factors: fuselage, wing, nacelle, vtail, htail, strut, tip tank
        # protect against Mach 0, any other small Mach should be ok
        dtype = complex if self.under_complex_step else float
        ffre, fwre, fnre, fvtre, fhtre, fstrtre = np.ones(
            (6, self.options["num_nodes"]), dtype=dtype
        )
        if self.under_complex_step:
            good_mask = reli.real > 1
        else:
            good_mask = reli > 1
        ffre[good_mask] = (np.log10(reli[good_mask] * fus_len) / 7) ** -2.6
        fwre[good_mask] = (np.log10(reli[good_mask] * avg_chord) / 7) ** -2.6
        fnre[good_mask] = (np.log10(reli[good_mask] * nac_len) / 7) ** -2.6
        fvtre[good_mask] = (np.log10(reli[good_mask] * vtail_chord) / 7) ** -2.6
        fhtre[good_mask] = (np.log10(reli[good_mask] * htail_chord) / 7) ** -2.6
        include_strut = self.options["aviary_options"].get_val(
            Aircraft.Wing.HAS_STRUT, units='unitless')
        if include_strut:
            fstrtre = (np.log10(reli[good_mask] * strut_chord) / 7) ** -2.6

        # fuselage form drag factor
        fffus = (
            1 + 1.5 * (cabin_width / fus_len) ** 1.5 + 7 * (cabin_width / fus_len) ** 3
        )

        # flat plate equivalent areas
        fef = ff_fus * fus_SA * cf * ffre * fffus + fe_fus_inc
        few = ff_wing * wing_area * cf * fwre
        # TODO replace 2 with num_engines
        fen = 2 * ff_nac * nacelle_area * cf * fnre
        fevt = ff_vtail * vtail_area * cf * fvtre
        feht = ff_htail * htail_area * cf * fhtre
        festrt = strut_fus_intf * strut_wing_area_ratio * wing_area * cf * fstrtre

        # begin INTERFERENCE - get flat plate equivalent for wing-fuselage interference
        # wing profile drag coefficient
        cdw0 = few / wing_area
        # interference drag independent of shielded area
        feshieldwf = cdw0 * areashieldwf
        feiwf = wing_fus_intf * (feintwf - feshieldwf)
        # end INTERFERENCE

        # total flat plate equivalent area
        fe = few + fef + fevt + feht + fen + feiwf + festrt + cd0_inc * wing_area

        wfob = cabin_width / wingspan
        siwb = (
            1
            - 0.0088 * wfob
            - 1.7364 * wfob**2
            - 2.303 * wfob**3
            + 6.0606 * wfob**4
        )

        # wing-free profile drag coefficient
        cdpo = (fe - few) / wing_area

        # Oswald efficiency
        see = 1.0 / (
            (1 / ufac / siwb)
            + 1.1938 * AR * (cdw0 / (np.cos(deg2rad(sweep_c4))) ** 2 + cdpo)
        )

        # compressibility drag parameters
        # sa1--4 are static, depending only on geometry
        sa1 = (1 + 0.0033 * (4 * dlmps - 3 * dlmtcx)) * (
            1 - 1.4 * tc_ratio - 0.06 * (1 - wing_min_pressure_loc)
        ) - 0.0368
        sa2 = (
            -0.33
            * (0.65 - wing_min_pressure_loc)
            * (1 + 0.0033 * (4 * dlmps - 3 * dlmtcx))
        )
        sa3 = (1.5 - 2 * fk**2 * np.sin(rlmle) ** 2) * tc_ratio ** (5 / 3.0)
        sa4 = 0.75 * tc_ratio

        # profile drag of everything but the wing
        sa5 = cdpo
        # wing profile drag
        sa6 = ff_wing * fwre
        # induced drag
        sa7 = 1.0 / (np.pi * see * AR)

        outputs["SA1"] = sa1
        outputs["SA2"] = sa2
        outputs["SA3"] = sa3
        outputs["SA4"] = sa4
        outputs["SA5"] = sa5
        outputs["SA6"] = sa6
        outputs["SA7"] = sa7
        outputs["cf"] = cf


class AeroSetup(om.Group):
    """Calculations for setting up aero"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')
        self.options.declare(
            "input_atmos",
            default=False,
            types=bool,
            desc="Directly input speed of sound and kinematic viscosity instead of "
            "computing them with an atmospherics component. For testing.",
        )
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        nn = self.options["num_nodes"]
        aviary_options = self.options['aviary_options']

        self.add_subsystem("ratios", WingTailRatios(), promotes=["*"])
        self.add_subsystem("xlifts", Xlifts(num_nodes=nn), promotes=["*"])

        # implements EAERO
        interp = om.MetaModelStructuredComp(method="2D-slinear")
        interp.add_input("bbar", 0.0, units="unitless", training_data=xbbar)
        interp.add_input("hbar", 0.0, units="unitless", training_data=xhbar)
        interp.add_output("sigma", 0.0, units="unitless", training_data=sig1)
        interp.add_output("sigstr", 0.0, units="unitless", training_data=sig2)
        self.add_subsystem("interp", interp, promotes=["*"])

        self.add_subsystem(
            "ufac_calc",
            om.ExecComp(
                "ufac=(1 + lift_ratio)**2 / (sigstr*(lift_ratio/bbar)**2 + 2*sigma*lift_ratio/bbar + 1)",
                lift_ratio={'units': "unitless", "shape": nn},
                bbar={'units': "unitless"},
                sigma={'units': "unitless"},
                sigstr={'units': "unitless"},
                ufac={'units': "unitless", "shape": nn},
                has_diag_partials=True
            ),
            promotes=["*"],
        )

        if not self.options["input_atmos"]:
            # self.add_subsystem(
            #     "atmos",
            #     USatm1976Comp(num_nodes=nn),
            #     promotes_inputs=[("h", Dynamic.Mission.ALTITUDE)],
            #     promotes_outputs=["rho", Dynamic.Mission.SPEED_OF_SOUND, "viscosity"],
            # )
            self.add_subsystem(
                "kin_visc",
                om.ExecComp(
                    "nu = viscosity / rho",
                    viscosity={"units": "lbf*s/ft**2", "shape": nn},
                    rho={"units": "slug/ft**3", "shape": nn},
                    nu={"units": "ft**2/s", "shape": nn},
                    has_diag_partials=True,
                ),
                promotes=["*", ('rho', Dynamic.Mission.DENSITY),
                          ('nu', Dynamic.Mission.KINEMATIC_VISCOSITY)],
            )

        self.add_subsystem("geom", AeroGeom(
            num_nodes=nn, aviary_options=aviary_options), promotes=["*"])


class DragCoef(om.ExplicitComponent):
    """GASP lift coefficient calculation for low-speed near-ground flight.

    Drag for low-speed flight is affected by ground effects, landing gear, and flap
    deflection. Flaps are treated separately as an increment so their effects may be
    gradually added/removed over time.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # mission inputs
        self.add_input(Dynamic.Mission.ALTITUDE, val=0.0,
                       units="ft", shape=nn, desc="Altitude")
        self.add_input(
            "CL", val=1.0, units="unitless", shape=nn, desc="Lift coefficient")

        # user inputs

        add_aviary_input(self, Mission.Design.GROSS_MASS, val=175400.0)

        self.add_input("flap_defl", val=10.0, units="deg", desc="Full flap deflection")

        add_aviary_input(self, Aircraft.Wing.HEIGHT, val=8.0)

        self.add_input(
            "airport_alt", val=0.0, units="ft", desc="HPORT: Airport altitude"
        )

        add_aviary_input(self, Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)

        # from flaps
        self.add_input(
            "dCL_flaps_model", val=0.0, units="unitless",
            desc="Delta CL from flaps model")
        self.add_input(
            "dCD_flaps_model", val=0.0, units="unitless",
            desc="Delta CD from flaps model")
        self.add_input(
            "dCL_flaps_coef",
            val=1.0, units="unitless",
            desc="SIGMTO | SIGMLD: Coefficient applied to delta CL from flaps model",
        )
        self.add_input(
            "CDI_factor",
            val=1.0, units="unitless",
            desc="VDEL6T | VDEL6L: Factor applied to induced drag with flaps",
        )

        # from sizing

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.Wing.SPAN, val=0.0)

        add_aviary_input(self, Aircraft.Wing.AREA, val=1370.3)

        # from aero setup
        self.add_input(
            "cf", units="unitless", shape=nn,
            desc="CFIN: Skin friction coefficient at Re=1e7")
        self.add_input("SA5", units="unitless", shape=nn, desc="SA5: Drag param")
        self.add_input("SA6", units="unitless", shape=nn, desc="SA6: Drag param")
        self.add_input("SA7", units="unitless", shape=nn, desc="SA7: Drag param")

        self.add_output("CD_base", units="unitless", shape=nn, desc="Drag coefficient")
        self.add_output(
            "dCD_flaps_full", units="unitless", shape=nn,
            desc="CD increment with full flap deflection")
        self.add_output(
            "dCD_gear_full", units="unitless",
            shape=nn, desc="CD increment with landing gear down")

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials("*", "*", dependent=False)
        ar = np.arange(self.options["num_nodes"])

        self.declare_partials("CD_base", ["*"], method="cs")
        self.declare_partials(
            "CD_base",
            [Dynamic.Mission.ALTITUDE, "CL", "cf", "SA5", "SA6", "SA7"],
            rows=ar,
            cols=ar,
            method="cs",
        )
        # self.declare_partials(
        #     "CD_base", [Mission.Design.GROSS_MASS, "dCD_flaps_model", "wing_area"], val=0
        # )

        self.declare_partials("dCD_flaps_full", ["dCD_flaps_model"], val=1)

        self.declare_partials(
            "dCD_gear_full",
            [Mission.Design.GROSS_MASS, Aircraft.Wing.AREA, "flap_defl"],
            method="cs",
        )

    def compute(self, inputs, outputs):
        (
            alt,
            CL,
            gross_mass_initial,
            flap_defl,
            wing_height,
            airport_alt,
            flap_chord_ratio,
            dCL_flaps_model,
            dCD_flaps_model,
            dCL_flaps_coef,
            CDI_factor,
            avg_chord,
            wingspan,
            wing_area,
            cf,
            SA5,
            SA6,
            SA7,
        ) = inputs.values()
        gross_wt_initial = gross_mass_initial * GRAV_ENGLISH_LBM

        # profile drag
        cd0 = SA5 + SA6 * cf

        # induced drag
        cdi = SA7 * (CL - dCL_flaps_coef * dCL_flaps_model) ** 2 / CDI_factor

        # ground effects - direct increment to CD
        hac = wing_height + alt - airport_alt
        heff = 2 * hac - np.sin(deg2rad(flap_defl)) * flap_chord_ratio * avg_chord
        sig = np.exp(-2.48 * (heff / wingspan) ** 0.768)
        betag = np.sqrt(1 + (heff / wingspan) ** 2) - heff / wingspan
        c1 = betag * CL / (12.5664 * hac)
        dcd_ground = -(sig - c1) * cdi / (1.0 - c1) - c1 * SA6 * cf

        # landing gear
        grfe = 0.0033 * gross_wt_initial**0.785
        dcd_gear = (grfe / wing_area) * (1 - 0.454545 * flap_defl / 50)

        outputs["CD_base"] = cd0 + cdi + dcd_ground
        outputs["dCD_flaps_full"] = dCD_flaps_model
        outputs["dCD_gear_full"] = dcd_gear


class DragCoefClean(om.ExplicitComponent):
    """Clean drag coefficient for high-speed flight"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # mission inputs
        self.add_input(Dynamic.Mission.MACH, val=0.0, units="unitless",
                       shape=nn, desc="Mach number")
        self.add_input(
            "CL", val=1.0, units="unitless", shape=nn, desc="Lift coefficient")

        # user inputs

        add_aviary_input(self, Aircraft.Design.SUPERCRITICAL_DIVERGENCE_SHIFT, val=0.033)

        # from aero setup
        self.add_input(
            "cf", units="unitless", shape=nn,
            desc="CFIN: Skin friction coefficient at Re=1e7")
        self.add_input("SA1", units="unitless", shape=nn, desc="SA1: Drag param")
        self.add_input("SA2", units="unitless", shape=nn, desc="SA2: Drag param")
        self.add_input("SA5", units="unitless", shape=nn, desc="SA5: Drag param")
        self.add_input("SA6", units="unitless", shape=nn, desc="SA6: Drag param")
        self.add_input("SA7", units="unitless", shape=nn, desc="SA7: Drag param")

        self.add_output("CD", units="unitless", shape=nn, desc="Drag coefficient")

    def setup_partials(self):
        ar = np.arange(self.options["num_nodes"])

        self.declare_partials(
            "CD",
            [Dynamic.Mission.MACH, "CL", "cf", "SA1", "SA2", "SA5", "SA6", "SA7"],
            rows=ar,
            cols=ar,
            method="cs",
        )
        self.declare_partials(
            "CD", [Aircraft.Design.SUPERCRITICAL_DIVERGENCE_SHIFT], method="cs"
        )

    def compute(self, inputs, outputs):
        mach, CL, div_drag_supercrit, cf, SA1, SA2, SA5, SA6, SA7 = inputs.values()

        mach_div = SA1 + SA2 * CL + div_drag_supercrit

        sig = sigmoid(mach, mach_div, alpha=0.005)
        delcdm = sig * (10 * (mach - mach_div) ** 3)

        # delcdm = np.zeros_like(mach)
        # mask = np.bitwise_and(mach >= mach_div, mach > 0.6)
        # delcdm[mask] = 10 * (mach[mask] - mach_div[mask]) ** 3

        # profile drag
        cd0 = SA5 + SA6 * cf
        # induced drag
        cdi = SA7 * CL**2

        outputs["CD"] = cd0 + cdi + delcdm


class LiftCoeff(om.ExplicitComponent):
    """GASP lift coefficient calculation for low-speed near-ground flight"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # mission inputs
        self.add_input("alpha", val=0.0, units="deg", shape=nn, desc="Angle of attack")
        self.add_input(Dynamic.Mission.ALTITUDE, val=0.0,
                       units="ft", shape=nn, desc="Altitude")
        self.add_input("lift_curve_slope", units="unitless",
                       shape=nn, desc="Lift-curve slope")
        self.add_input("lift_ratio", units="unitless", shape=nn, desc="Lift ratio")

        # user inputs

        add_aviary_input(self, Aircraft.Wing.ZERO_LIFT_ANGLE, val=-1.2)

        add_aviary_input(self, Aircraft.Wing.SWEEP, val=25.0)

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=10.13)

        add_aviary_input(self, Aircraft.Wing.HEIGHT, val=8.0)

        self.add_input(
            "airport_alt", val=0.0, units="ft", desc="HPORT: Airport altitude"
        )

        self.add_input("flap_defl", val=10.0, units="deg", desc="Full flap deflection")

        add_aviary_input(self, Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)

        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, val=0.33)

        # from flaps
        self.add_input(
            "CL_max_flaps", units="unitless",
            desc="CLMWTO | CLMWLD: Max lift coefficient from flaps model",
        )
        self.add_input(
            "dCL_flaps_model", val=0.0, units="unitless",
            desc="Delta CL from flaps model")

        # from sizing

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.Wing.SPAN, val=0.0)

        self.add_output(
            "CL_base", units="unitless", shape=nn, desc="Base lift coefficient")
        self.add_output(
            "dCL_flaps_full", units="unitless", shape=nn,
            desc="CL increment with full flap deflection"
        )
        self.add_output(
            "alpha_stall", units="deg", shape=nn, desc="Stall angle of attack"
        )
        self.add_output(
            "CL_max", units="unitless", shape=nn, desc="Max lift coefficient")

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials("*", "*", dependent=False)
        ar = np.arange(self.options["num_nodes"])

        dynvars = ["alpha", Dynamic.Mission.ALTITUDE, "lift_curve_slope", "lift_ratio"]

        self.declare_partials("CL_base", ["*"], method="cs")
        self.declare_partials("CL_base", dynvars, rows=ar, cols=ar, method="cs")

        self.declare_partials("dCL_flaps_full", ["dCL_flaps_model"], method="cs")
        self.declare_partials(
            "dCL_flaps_full", ["lift_ratio"], rows=ar, cols=ar, method="cs")

        self.declare_partials("alpha_stall", ["*"], method="cs")
        self.declare_partials("alpha_stall", dynvars, rows=ar, cols=ar, method="cs")

        self.declare_partials("CL_max", ["CL_max_flaps"], method="cs")
        self.declare_partials("CL_max", ["lift_ratio"], rows=ar, cols=ar, method="cs")

    def compute(self, inputs, outputs):
        (
            alpha,
            alt,
            lift_curve_slope,
            lift_ratio,
            alpha0,
            sweep_c4,
            AR,
            wing_height,
            airport_alt,
            flap_defl,
            flap_chord_ratio,
            taper_ratio,
            CL_max_flaps,
            dCL_flaps_model,
            avg_chord,
            wingspan,
        ) = inputs.values()

        # ground effects - factor on lift-curve slope
        hac = wing_height + alt - airport_alt
        heff = 2 * hac - np.sin(deg2rad(flap_defl)) * flap_chord_ratio * avg_chord
        sig = np.exp(-2.48 * (heff / wingspan) ** 0.768)
        betag = (1 + (heff / wingspan) ** 2) ** 0.5 - heff / wingspan
        rlmc2 = cs.arctan2(
            AR * np.tan(deg2rad(sweep_c4)) - ((1 - taper_ratio) / (1 + taper_ratio)), AR
        )
        c3 = 2 * np.cos(rlmc2) + np.sqrt(AR**2 + (2 * np.cos(rlmc2)) ** 2)
        c4 = betag / (12.5664 * hac / avg_chord)
        cloge = lift_curve_slope * deg2rad(alpha - alpha0) + dCL_flaps_model
        kclge = (
            1
            + sig
            - sig * AR * np.cos(rlmc2) / c3
            - c4 * (cloge - lift_curve_slope / (16 * hac / avg_chord))
        )
        kclge = np.clip(kclge, 1.0, None)

        outputs["CL_base"] = kclge * lift_curve_slope * \
            deg2rad(alpha - alpha0) * (1 + lift_ratio)
        outputs["dCL_flaps_full"] = dCL_flaps_model * (1 + lift_ratio)

        outputs["alpha_stall"] = (
            rad2deg((CL_max_flaps - dCL_flaps_model) /
                    (kclge * lift_curve_slope)) + alpha0
        )
        outputs["CL_max"] = CL_max_flaps * (1 + lift_ratio)


class LiftCoeffClean(om.ExplicitComponent):
    """Clean wing lift coefficient for high-speed flight"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare(
            "output_alpha",
            default=False,
            types=bool,
            desc="If True, output alpha for a given input CL",
        )

    def setup(self):
        nn = self.options["num_nodes"]

        if self.options["output_alpha"]:
            self.add_output(
                "alpha", val=0.0, units="deg", shape=nn, desc="Angle of attack"
            )
            self.add_input(
                "CL", val=1.0, units="unitless", shape=nn, desc="Lift coefficient")
        else:
            self.add_input(
                "alpha", val=0.0, units="deg", shape=nn, desc="Angle of attack"
            )
            self.add_output(
                "CL", val=1.0, units="unitless", shape=nn, desc="Lift coefficient")

        self.add_input("lift_curve_slope", units="unitless",
                       shape=nn, desc="Lift-curve slope")
        self.add_input("lift_ratio", units="unitless", shape=nn, desc="Lift ratio")

        add_aviary_input(self, Aircraft.Wing.ZERO_LIFT_ANGLE, val=-1.2)

        add_aviary_input(self, Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP, val=0.0)

        self.add_output("alpha_stall", shape=nn, desc="Stall angle of attack")
        self.add_output(
            "CL_max", units="unitless", shape=nn, desc="Max lift coefficient")

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials("*", "*", dependent=False)
        ar = np.arange(self.options["num_nodes"])

        if self.options["output_alpha"]:
            self.declare_partials(
                "alpha", ["CL", "lift_ratio", "lift_curve_slope"], rows=ar, cols=ar, method="cs"
            )
            self.declare_partials("alpha", [Aircraft.Wing.ZERO_LIFT_ANGLE], method="cs")
        else:
            self.declare_partials(
                "CL", ["lift_curve_slope", "alpha", "lift_ratio"], rows=ar, cols=ar, method="cs"
            )
            self.declare_partials("CL", [Aircraft.Wing.ZERO_LIFT_ANGLE], method="cs")

        self.declare_partials(
            "alpha_stall", ["lift_curve_slope"], rows=ar, cols=ar, method="cs")
        self.declare_partials(
            "alpha_stall",
            [
                Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP,
                Aircraft.Wing.ZERO_LIFT_ANGLE,
            ],
            method="cs",
        )

        self.declare_partials("CL_max", ["lift_ratio"], rows=ar, cols=ar, method="cs")
        self.declare_partials(
            "CL_max", [Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP], method="cs"
        )

    def compute(self, inputs, outputs):
        _, lift_curve_slope, lift_ratio, alpha0, CL_max_flaps = inputs.values()
        if self.options["output_alpha"]:
            CL = inputs["CL"]
            clw = CL / (1 + lift_ratio)
            outputs["alpha"] = rad2deg(clw / lift_curve_slope) + alpha0
        else:
            alpha = inputs["alpha"]
            outputs["CL"] = lift_curve_slope * deg2rad(alpha - alpha0) * (1 + lift_ratio)

        outputs["alpha_stall"] = rad2deg(CL_max_flaps / lift_curve_slope) + alpha0
        outputs["CL_max"] = CL_max_flaps * (1 + lift_ratio)


class CruiseAero(om.Group):
    """Top-level aerodynamics group for cruise (no flaps, no landing gear)"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

        self.options.declare(
            "output_alpha",
            default=False,
            types=bool,
            desc="If True, output alpha for a given input CL",
        )
        self.options.declare(
            "input_atmos",
            default=False,
            types=bool,
            desc="Directly input speed of sound and kinematic viscosity instead of "
            "computing them with an atmospherics component. For testing.",
        )
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        nn = self.options["num_nodes"]
        aviary_options = self.options["aviary_options"]
        self.add_subsystem(
            "aero_setup",
            AeroSetup(num_nodes=nn, aviary_options=aviary_options,
                      input_atmos=self.options["input_atmos"]),
            promotes=["*"],
        )
        if self.options["output_alpha"]:
            # lift_req -> CL
            self.add_subsystem("lift2cl", CLFromLift(num_nodes=nn), promotes=["*"])
        self.add_subsystem(
            "lift_coef",
            LiftCoeffClean(output_alpha=self.options["output_alpha"], num_nodes=nn),
            promotes=["*"],
        )
        self.add_subsystem("drag_coef", DragCoefClean(num_nodes=nn), promotes=["*"])
        self.add_subsystem("forces", AeroForces(num_nodes=nn), promotes=["*"])


class LowSpeedAero(om.Group):
    """Top-level aerodynamics group for near-ground flight"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')
        self.options.declare(
            "retract_gear",
            default=True,
            types=bool,
            desc="True to start with gear landing gear down, False for reverse",
        )
        self.options.declare(
            "retract_flaps",
            default=True,
            types=bool,
            desc="True to start with flaps applied, False for reverse",
        )
        # TODO this option does not really exist for LowSpeed and should be renamed
        # (the value of having identical option set to cruise aero not worth the added
        #  confusion of having a mislabeled option here)
        self.options.declare(
            "output_alpha",
            default=False,
            types=bool,
            desc="If True, output alpha for a given input CL",
        )
        self.options.declare(
            "input_atmos",
            default=False,
            types=bool,
            desc="Directly input speed of sound and kinematic viscosity instead of "
            "computing them with an atmospherics component. For testing.",
        )
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        nn = self.options["num_nodes"]
        output_alpha = self.options["output_alpha"]
        aviary_options = self.options["aviary_options"]
        self.add_subsystem(
            "aero_setup",
            AeroSetup(num_nodes=nn, aviary_options=aviary_options,
                      input_atmos=self.options["input_atmos"]),
            promotes=["*"],
        )

        aero_ramps = TanhRampComp(time_units='s', num_nodes=nn)
        aero_ramps.add_ramp('flap_factor', output_units='unitless',
                            initial_val=1.0 if self.options['retract_flaps'] else 0.0,
                            final_val=0.0 if self.options['retract_flaps'] else 1.0)
        aero_ramps.add_ramp('gear_factor', output_units='unitless',
                            initial_val=1.0 if self.options['retract_gear'] else 0.0,
                            final_val=0.0 if self.options['retract_gear'] else 1.0)

        self.add_subsystem("aero_ramps",
                           aero_ramps,
                           promotes_inputs=[("time", "t_curr"),
                                            ("flap_factor:t_init", "t_init_flaps"),
                                            ("flap_factor:t_duration", "dt_flaps"),
                                            ("gear_factor:t_init", "t_init_gear"),
                                            ("gear_factor:t_duration", "dt_gear")],
                           promotes_outputs=['flap_factor',
                                             'gear_factor'])

        if output_alpha:
            # lift_req -> CL
            self.add_subsystem(
                "lift2cl",
                CLFromLift(num_nodes=nn),
                promotes_inputs=["*"],
                # little bit of a hack here - input CL bypasses CL increment ramp
                # so ensure this is what's passed to DragCoef
                promotes_outputs=[("CL", "CL_full_flaps")],
            )
            warnings.warn("Alpha is NOT an output from LowSpeedAero.")
        else:
            self.add_subsystem(
                "lift_coef",
                LiftCoeff(num_nodes=nn),
                promotes_inputs=["*"],
                promotes_outputs=["*"]
            )

            self.add_subsystem(
                "total_cl",
                om.ExecComp(
                    [
                        # "CL = CL_base + dCL_flaps",
                        "CL_full_flaps = CL_base + dCL_flaps_full",
                        "CL = CL_base + flap_factor * dCL_flaps_full",
                    ],
                    CL=dict(shape=nn, units='unitless'),
                    CL_full_flaps=dict(shape=nn, units='unitless'),
                    CL_base=dict(shape=nn, units='unitless'),
                    # dCL_flaps=dict(shape=nn, units='unitless'),
                    flap_factor=dict(shape=nn, units='unitless'),
                    dCL_flaps_full=dict(shape=nn, units='unitless'),
                    has_diag_partials=True
                ),
                promotes=["*"],
            )

        interp = om.MetaModelStructuredComp(method="slinear")
        interp.add_input("flap_defl", 10.0, units="deg", training_data=adelfd)
        interp.add_output("dCL_flaps_coef", 0.0, units="unitless", training_data=asigma)
        interp.add_output("CDI_factor", 1.0, units="unitless", training_data=adel6)
        self.add_subsystem("cdi_flap_interp", interp, promotes=["*"])

        self.add_subsystem(
            "drag_coef", DragCoef(num_nodes=nn), promotes=["*", ("CL", "CL_full_flaps")]
        )

        self.add_subsystem(
            "total_cd",
            om.ExecComp(
                "CD = CD_base + flap_factor * dCD_flaps_full + gear_factor"
                " * dCD_gear_full",
                # "CD = CD_base + dCD_flaps + dCD_gear",
                CD=dict(shape=nn, units='unitless'),
                CD_base=dict(shape=nn, units='unitless'),
                # dCD_flaps=dict(shape=nn, units='unitless'),
                # dCD_gear=dict(shape=nn, units='unitless'),
                flap_factor=dict(shape=nn, units='unitless'),
                gear_factor=dict(shape=nn, units='unitless'),
                dCD_gear_full=dict(shape=nn, units='unitless'),
                dCD_flaps_full=dict(shape=nn, units='unitless'),
                has_diag_partials=True
            ),
            promotes=["*"],
        )

        self.add_subsystem("forces", AeroForces(num_nodes=nn), promotes=["*"])

        self.set_input_defaults(Dynamic.Mission.ALTITUDE, np.zeros(nn))

        if self.options["retract_gear"]:
            # takeoff defaults
            self.set_input_defaults("dt_gear", 7)
        # gear not dynamically extended during landing

        if self.options["retract_flaps"]:
            # takeoff defaults
            self.set_input_defaults("flap_defl", 10)
            self.set_input_defaults("dt_flaps", 3)
        else:
            # landing defaults
            self.set_input_defaults("flap_defl", 40)
            # flaps not dynamically extended during landing
