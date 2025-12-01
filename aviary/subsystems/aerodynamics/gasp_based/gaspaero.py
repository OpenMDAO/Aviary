import warnings
import numpy as np
import openmdao.api as om
from openmdao.utils import cs_safe as cs

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.aerodynamics.gasp_based.common import AeroForces, CLFromLift, TanhRampComp
from aviary.utils.functions import sigmoidX, smooth_min, d_smooth_min
from aviary.variable_info.enums import AircraftTypes, Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

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
# block auto-formatting of tables
# autopep8: off
# fmt: off
adelfd = np.array(
    [
        0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
        38.0, 40.0, 42.0, 44.0, 50.0, 55.0, 60.0,
    ]
)
# flap angle correction of oswald efficiency factor
adel6 = np.array(
    [
        1.0, 0.995, 0.99, 0.98, 0.97, 0.955, 0.935, 0.90,
        0.875, 0.855, 0.83, 0.80, 0.70, 0.54, 0.30,
    ]
)
# induced drag correction factors
asigma = np.array(
    [
        0.0, 0.16, 0.285, 0.375, 0.435, 0.48, 0.52, 0.55,
        0.575, 0.58, 0.59, 0.60, 0.62, 0.635, 0.65,
    ]
)
# autopep8: off
# fmt: off

def deg2rad(d):
    """Complex step safe deg2rad."""
    return d * np.pi / 180.0


def rad2deg(r):
    """Complex step safe rad2deg."""
    return r * 180.0 / np.pi


def cla(ar, sweep, mach):
    """Lift-curve slope of 3D wings from Seckel equation. cla is per radian.

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
        / (1 + np.sqrt(1 + (((ar / (2 * np.cos(sweep))) ** 2) * (1 - (mach * np.cos(sweep)) ** 2))))
    )


class WingTailRatios(om.ExplicitComponent):
    # NOTE this is actually getting added in mission, not pre-mission. Which place is
    # intended for this component??
    """Pre-mission calculation of ratios between tail and wing parameters."""

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')

        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, units='ft')

        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless')

        add_aviary_input(self, Aircraft.Wing.VERTICAL_MOUNT_LOCATION, units='unitless')

        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, units='unitless')

        add_aviary_input(self, Aircraft.HorizontalTail.SPAN, units='ft')

        add_aviary_input(self, Aircraft.VerticalTail.SPAN, units='ft')

        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')

        add_aviary_input(self, Aircraft.HorizontalTail.AVERAGE_CHORD, units='ft')

        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')

        self.add_output(
            'hbar',
            val=0.0,
            units='unitless',
            desc='HBAR: Ratio of HGAP(?) to wing span',
        )
        self.add_output('bbar', units='unitless', desc='BBAR: Ratio of H tail area to wing span')
        self.add_output('sbar', units='unitless', desc='SBAR: Ratio of H tail area to wing area')
        self.add_output('cbar', units='unitless', desc='CBAR: Ratio of H tail chord to wing chord')
        self.add_output(
            'bbar_alt', units='unitless', desc='BHTqB: Ratio of H tail area to wing span'
        )

    def setup_partials(self):
        self.declare_partials(
            'hbar',
            [
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.VerticalTail.SPAN,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.VERTICAL_MOUNT_LOCATION,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
            ],
            method='cs',
        )
        self.declare_partials(
            'bbar', [Aircraft.HorizontalTail.SPAN, Aircraft.Wing.SPAN], method='cs'
        )
        self.declare_partials(
            'sbar', [Aircraft.HorizontalTail.AREA, Aircraft.Wing.AREA], method='cs'
        )
        self.declare_partials(
            'cbar',
            [Aircraft.HorizontalTail.AVERAGE_CHORD, Aircraft.Wing.AVERAGE_CHORD],
            method='cs',
        )
        self.declare_partials(
            'bbar_alt', [Aircraft.HorizontalTail.SPAN, Aircraft.Wing.SPAN], method='cs'
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
        verbosity = self.options[Settings.VERBOSITY]

        if wing_area == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.Wing.AREA is 0.0 in WingTailRatios.')
        if wingspan == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.Wing.SPAN is 0.0 in WingTailRatios.')
        if avg_chord == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.Wing.AVERAGE_CHORD is 0.0 in WingTailRatios.')

        trtw = tc_ratio_root * 2 * wing_area / wingspan / (1 + taper_ratio)
        hgap = cs.abs(htail_loc * span_vtail - 0.5 * (cabin_width - trtw) * (2 * wing_loc - 1))
        outputs['hbar'] = hgap / wingspan
        outputs['bbar'] = span_htail / wingspan
        outputs['sbar'] = htail_area / wing_area
        outputs['cbar'] = htail_chord / avg_chord
        # bbar_alt is a modified bbar for tailless BWB
        if span_htail < 0.01 * wingspan:
            outputs['bbar_alt'] = 1.0
        else:
            outputs['bbar_alt'] = outputs['bbar']


class BWBBodyLiftCurveSlope(om.ExplicitComponent):
    """Compute body lift curve slope of BWB."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        nn = self.options['num_nodes']
        # mission inputs
        add_aviary_input(self, Dynamic.Atmosphere.MACH, shape=nn, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.LIFT_CURVE_SLOPE_MACH0, units='1/rad')

        self.add_output(
            'body_lift_curve_slope',
            units='unitless',
            shape=nn,
            desc='CLALPH_B: Lift-curve slope of fuselage for the given Mach',
        )

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(
            'body_lift_curve_slope',
            Dynamic.Atmosphere.MACH,
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'body_lift_curve_slope',
            Aircraft.Fuselage.LIFT_CURVE_SLOPE_MACH0,
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        mach = inputs[Dynamic.Atmosphere.MACH]

        # Provide a little buffer for being close to the limit.
        eps = 1e-8

        if any(x < -eps or x >= 1.0 + eps for x in mach.real):
            raise om.AnalysisError('Mach number must be within the range (0, 1).')
        elif any(x > 0.8 + eps for x in mach.real):
            if verbosity > Verbosity.BRIEF:
                warnings.warn(
                    f"Mach range should be less or equal to 0.8. You've provided a Mach number {mach}."
                )
        CLALPH_B0 = inputs[Aircraft.Fuselage.LIFT_CURVE_SLOPE_MACH0]
        CLALPH_B = CLALPH_B0 / np.sqrt(1.0 - mach**2)
        outputs['body_lift_curve_slope'] = CLALPH_B

    def compute_partials(self, inputs, J):
        mach = inputs[Dynamic.Atmosphere.MACH]
        CLALPH_B0 = inputs[Aircraft.Fuselage.LIFT_CURVE_SLOPE_MACH0]
        J['body_lift_curve_slope', Dynamic.Atmosphere.MACH] = (
            CLALPH_B0 * mach / (np.sqrt(1.0 - mach**2)) ** 3
        )
        J['body_lift_curve_slope', Aircraft.Fuselage.LIFT_CURVE_SLOPE_MACH0] = 1 / np.sqrt(
            1.0 - mach**2
        )


class Xlifts(om.ExplicitComponent):
    """Compute lift ratio and lift-curve slope for given stability margin."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        nn = self.options['num_nodes']

        # mission inputs
        add_aviary_input(self, Dynamic.Atmosphere.MACH, shape=nn, units='unitless')

        # stability inputs

        add_aviary_input(self, Aircraft.Design.STATIC_MARGIN, units='unitless')

        add_aviary_input(self, Aircraft.Design.CG_DELTA, units='unitless')

        # geometry inputs

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')

        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')

        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, units='unitless')

        add_aviary_input(self, Aircraft.HorizontalTail.SWEEP, units='deg')

        add_aviary_input(self, Aircraft.HorizontalTail.MOMENT_RATIO, units='unitless')

        # geometry from wing-tail ratios
        self.add_input('sbar', units='unitless', desc='SBAR: Ratio of H tail area to wing area')
        self.add_input('cbar', units='unitless', desc='CBAR: Ratio of H tail chord to wing chord')
        self.add_input('hbar', units='unitless', desc='HBAR: Ratio of HGAP(?) to wing span')
        self.add_input('bbar', units='unitless', desc='BBAR: Ratio of H tail area to wing span')

        self.add_output(
            'lift_curve_slope', units='unitless', shape=nn, desc='CLAW: Lift-curve slope'
        )
        self.add_output('lift_ratio', units='unitless', shape=nn, desc='BARL: Lift ratio')

    def setup_partials(self):
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials('lift_ratio', '*', method='cs')
        self.declare_partials('lift_ratio', Dynamic.Atmosphere.MACH, rows=ar, cols=ar, method='cs')
        self.declare_partials('lift_curve_slope', '*', method='cs')
        self.declare_partials(
            'lift_curve_slope',
            [
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.HorizontalTail.SWEEP,
                Aircraft.HorizontalTail.MOMENT_RATIO,
                'sbar',
                'cbar',
                'hbar',
                'bbar',
            ],
            method='cs',
        )
        self.declare_partials(
            'lift_curve_slope', Dynamic.Atmosphere.MACH, rows=ar, cols=ar, method='cs'
        )

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
        verbosity = self.options[Settings.VERBOSITY]

        if h_tail_moment == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.HorizontalTail.MOMENT_RATIO should not be 0.0 in Xlifts.')
        if sbar == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('sbar should not be 0.0 in Xlifts.')
        if bbar == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('bbar should not be 0.0 in Xlifts.')
        if AR == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.Wing.ASPECT_RATIO should not be 0.0 in Xlifts.')

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
        eps5 = cs.abs(xt) / (np.pi * art * np.sqrt(xt**2 + h**2 + art**2 * cbar**2 / 4))

        denom = 1 - clat0 * claw0 * (eps1 + eps2 + eps3) * (eps4 - eps5 - cbar * eps1)
        if np.any(denom) == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn(
                    'lift_curve_slope is undefined because the denominator is 0.0 in Xlift.'
                )
        claw = claw0 * (1 - clat0 * (eps4 - eps5 - cbar * eps1)) / denom

        clat = clat0 * (1 - claw * (eps1 + eps2 + eps3))

        if np.any(claw) == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('lift_ratio is undefined because the denominator is 0.0 in Xlift.')
        abar = clat / claw
        c = 1 / (1 + 1 / abar / sbar)
        lift_ratio = (c - delta) / (1 + delta - c)  # for AFT Tails

        outputs['lift_curve_slope'] = claw
        outputs['lift_ratio'] = lift_ratio


class FormFactorAndSIWB(om.ExplicitComponent):
    """
    Compute body form factor and SIWB for tube+wing aircraft
    Incompressible form factor for streamlined bodies. From Hoerner's "Fluid Dynamic Drag", p. 6-17.
    """

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft', desc='SWF')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft', desc='ELF')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft', desc='B')

        add_aviary_output(
            self,
            Aircraft.Fuselage.FORM_FACTOR,
            units='unitless',
            desc='FFFUS: fuselage form factor',
        )
        self.add_output(
            'siwb',
            units='unitless',
            desc='SIWB: curve fitting correction factor for Oswald efficiency',
        )

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.FORM_FACTOR,
            [
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.LENGTH,
            ],
        )
        self.declare_partials(
            'siwb',
            [
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.SPAN,
            ],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        wingspan = inputs[Aircraft.Wing.SPAN]

        if fus_len == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.Fuselage.LENGTH should not be 0.0 in FormFactorAndSIWB.')
        if wingspan == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.Wing.SPAN should not be 0.0 in FormFactorAndSIWB.')

        # fuselage form drag factor
        fffus = 1 + 1.5 * (cabin_width / fus_len) ** 1.5 + 7 * (cabin_width / fus_len) ** 3
        outputs[Aircraft.Fuselage.FORM_FACTOR] = fffus

        # fuselage width over wing span
        wfob = cabin_width / wingspan
        siwb = 1 - 0.0088 * wfob - 1.7364 * wfob**2 - 2.303 * wfob**3 + 6.0606 * wfob**4
        outputs['siwb'] = siwb

    def compute_partials(self, inputs, J):
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        wingspan = inputs[Aircraft.Wing.SPAN]

        dfffus_dcabin_width = (
            2.25 * (cabin_width / fus_len) ** 0.5 / fus_len
            + 21 * (cabin_width / fus_len) ** 2.0 / fus_len
        )
        dfffus_dfus_len = (
            -2.25 * (cabin_width / fus_len) ** 0.5 * cabin_width / fus_len**2.0
            - 21.0 * (cabin_width / fus_len) ** 2.0 * cabin_width / fus_len**2.0
        )

        wfob = cabin_width / wingspan
        # siwb = 1 - 0.0088 * wfob - 1.7364 * wfob**2 - 2.303 * wfob**3 + 6.0606 * wfob**4

        dsiwb_dcabin_width = (
            -0.0088 / wingspan
            - 2 * 1.7364 * wfob / wingspan
            - 3 * 2.303 * wfob**2 / wingspan
            + 4 * 6.0606 * wfob**3 / wingspan
        )

        dsiwb_dwingspan = (
            0.0088 * cabin_width / wingspan**2
            + 2 * 1.7364 * wfob * cabin_width / wingspan**2
            + 3 * 2.303 * wfob**2 * cabin_width / wingspan**2
            - 4 * 6.0606 * wfob**3 * cabin_width / wingspan**2
        )

        J[Aircraft.Fuselage.FORM_FACTOR, Aircraft.Fuselage.AVG_DIAMETER] = dfffus_dcabin_width
        J[Aircraft.Fuselage.FORM_FACTOR, Aircraft.Fuselage.LENGTH] = dfffus_dfus_len
        J['siwb', Aircraft.Fuselage.AVG_DIAMETER] = dsiwb_dcabin_width
        J['siwb', Aircraft.Wing.SPAN] = dsiwb_dwingspan


class BWBFormFactorAndSIWB(om.ExplicitComponent):
    """
    Compute body form factor and SIWB for BWB aircraft
    Incompressible form factor for streamlined bodies. From Hoerner's "Fluid Dynamic Drag", p. 6-17.
    """

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.HYDRAULIC_DIAMETER, units='ft', desc='DHYDRAL')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft', desc='ELF')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft', desc='B')

        self.add_output(
            Aircraft.Fuselage.FORM_FACTOR,
            units='unitless',
            desc='FFFUS: fuselage form factor',
        )
        self.add_output(
            'siwb',
            units='unitless',
            desc='SIWB: curve fitting correction factor for Oswald efficiency',
        )

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.FORM_FACTOR,
            [
                Aircraft.Fuselage.HYDRAULIC_DIAMETER,
                Aircraft.Fuselage.LENGTH,
            ],
        )
        self.declare_partials(
            'siwb',
            [
                Aircraft.Fuselage.HYDRAULIC_DIAMETER,
                Aircraft.Wing.SPAN,
            ],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        diam = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        wingspan = inputs[Aircraft.Wing.SPAN]

        if fus_len == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.Fuselage.LENGTH should not be 0.0 in FormFactorAndSIWB.')
        if wingspan == 0.0:
            if verbosity > Verbosity.BRIEF:
                warnings.warn('Aircraft.Wing.SPAN should not be 0.0 in FormFactorAndSIWB.')

        # fuselage form drag factor
        fffus = 1 + 1.5 * (diam / fus_len) ** 1.5 + 7 * (diam / fus_len) ** 3
        outputs[Aircraft.Fuselage.FORM_FACTOR] = fffus

        # hydraulic diameter over wing span
        wfob = diam / wingspan
        siwb = 1 - 0.0088 * wfob - 1.7364 * wfob**2 - 2.303 * wfob**3 + 6.0606 * wfob**4
        outputs['siwb'] = siwb

    def compute_partials(self, inputs, J):
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        diam = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        wingspan = inputs[Aircraft.Wing.SPAN]

        dfffus_ddiam = (
            2.25 * (diam / fus_len) ** 0.5 / fus_len + 21 * (diam / fus_len) ** 2.0 / fus_len
        )
        dfffus_dfus_len = (
            -2.25 * (diam / fus_len) ** 0.5 * diam / fus_len**2.0
            - 21.0 * (diam / fus_len) ** 2.0 * diam / fus_len**2.0
        )

        wfob = diam / wingspan
        # siwb = 1 - 0.0088 * wfob - 1.7364 * wfob**2 - 2.303 * wfob**3 + 6.0606 * wfob**4

        dsiwb_ddiam = (
            -0.0088 / wingspan
            - 2 * 1.7364 * wfob / wingspan
            - 3 * 2.303 * wfob**2 / wingspan
            + 4 * 6.0606 * wfob**3 / wingspan
        )

        dsiwb_dwingspan = (
            0.0088 * diam / wingspan**2
            + 2 * 1.7364 * wfob * diam / wingspan**2
            + 3 * 2.303 * wfob**2 * diam / wingspan**2
            - 4 * 6.0606 * wfob**3 * diam / wingspan**2
        )

        J[Aircraft.Fuselage.FORM_FACTOR, Aircraft.Fuselage.HYDRAULIC_DIAMETER] = dfffus_ddiam
        J[Aircraft.Fuselage.FORM_FACTOR, Aircraft.Fuselage.LENGTH] = dfffus_dfus_len
        J['siwb', Aircraft.Fuselage.HYDRAULIC_DIAMETER] = dsiwb_ddiam
        J['siwb', Aircraft.Wing.SPAN] = dsiwb_dwingspan


class UFac(om.ExplicitComponent):
    """GASP EAERO subroutine"""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        add_aviary_option(self, Aircraft.Design.TYPE)
        self.options.declare('smooth_ufac', default=False, types=bool)  # only for BWB
        self.options.declare('mu', default=100.0, types=float)  # only for BWB

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('lift_ratio', units='unitless', shape=nn, desc='BARL: Lift ratio')
        # bbar is the ratio of horizontal tail span over wing span.
        self.add_input('bbar_alt', units='unitless', desc='BBAR')
        self.add_input('sigma', units='unitless', desc='interference factor')
        self.add_input('sigstr', units='unitless', desc='nonelliptic interference factor')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft', desc='B')
        self.add_output('ufac', shape=nn, units='unitless')

    def setup_partials(self):
        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(
            'ufac',
            [
                'lift_ratio',
            ],
            rows=ar,
            cols=ar,
        )
        self.declare_partials(
            'ufac',
            [
                'bbar_alt',
                'sigma',
                'sigstr',
            ],
        )

    def compute(self, inputs, outputs):
        design_type = self.options[Aircraft.Design.TYPE]
        lift_ratio = inputs['lift_ratio']
        bbar = inputs['bbar_alt']
        sigma = inputs['sigma']
        sigstr = inputs['sigstr']
        wingspan = inputs[Aircraft.Wing.SPAN]

        if design_type is AircraftTypes.TRANSPORT:
            ufac = (1 + lift_ratio) ** 2 / (
                sigstr * (lift_ratio / bbar) ** 2 + 2 * sigma * lift_ratio / bbar + 1
            )
        else:
            # Modify for tailless "BWB"
            if bbar < 0.01 * wingspan:
                bbar = 1.0
            ufac = (1 + lift_ratio) ** 2 / (
                sigstr * (lift_ratio / bbar) ** 2 + 2 * sigma * lift_ratio / bbar + 1
            )
            # For BWB, GASP has to limit UFAC under 0.975 for good result
            smooth = self.options['smooth_ufac']
            if smooth:
                mu = self.options['mu']
                ufac = smooth_min(ufac, 0.975, mu)
            else:
                ufac = np.minimum(ufac, 0.975)

        outputs['ufac'] = ufac

    def compute_partials(self, inputs, J):
        design_type = self.options[Aircraft.Design.TYPE]
        lift_ratio = inputs['lift_ratio']
        bbar = inputs['bbar_alt']
        sigma = inputs['sigma']
        sigstr = inputs['sigstr']

        numer = (1 + lift_ratio) ** 2
        denom = sigstr * (lift_ratio / bbar) ** 2 + 2 * sigma * lift_ratio / bbar + 1
        ufac = numer / denom

        # w.r.t. lift_ratio
        dnumer = 2 * (1 + lift_ratio) / bbar
        ddenom = 2 * sigstr * (lift_ratio / bbar) / bbar + 2 * sigma / bbar
        dufac_dlift_ratio = (dnumer * denom - numer * ddenom) / denom**2

        dufac_dsigstr = -numer * (lift_ratio / bbar) ** 2 / denom**2
        dufac_dsigma = -numer * 2 * lift_ratio / bbar / denom**2

        ddenom_dbbar = (
            -2 * sigstr * (lift_ratio / bbar) * lift_ratio / bbar**2
            - 2 * sigma * lift_ratio / bbar**2
        )
        dufac_dbbar = -numer * ddenom_dbbar / denom**2

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            smooth = self.options['smooth_ufac']
            if smooth:
                mu = self.options['mu']
                dufac_dsigstr = d_smooth_min(ufac, 0.975, mu) * dufac_dsigstr
                dufac_dsigma = d_smooth_min(ufac, 0.975, mu) * dufac_dsigma
                dufac_dbbar = d_smooth_min(ufac, 0.975, mu) * dufac_dbbar
            else:
                ufac = np.minimum(ufac, 0.975)
                dufac_dsigstr = (
                    np.piecewise(ufac, [ufac < 0.975, ufac >= 0.975], [1, 0]) * dufac_dsigstr
                )
                dufac_dsigma = (
                    np.piecewise(ufac, [ufac < 0.975, ufac >= 0.975], [1, 0]) * dufac_dsigma
                )
                dufac_dbbar = (
                    np.piecewise(ufac, [ufac < 0.975, ufac >= 0.975], [1, 0]) * dufac_dbbar
                )

        J['ufac', 'lift_ratio'] = dufac_dlift_ratio
        J['ufac', 'bbar_alt'] = dufac_dbbar
        J['ufac', 'sigma'] = dufac_dsigma
        J['ufac', 'sigstr'] = dufac_dsigstr


class AeroGeom(om.ExplicitComponent):
    """Compute drag parameters from cruise conditions and geometric parameters.

    This corresponds to the AERO subroutine in GASP. The primary outputs are parameters
    SA* which build up the total aircraft drag coefficient.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):
        nn = self.options['num_nodes']
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Dynamic.Atmosphere.MACH, shape=nn, units='unitless')
        add_aviary_input(self, Dynamic.Atmosphere.SPEED_OF_SOUND, shape=nn, units='ft/s')
        add_aviary_input(
            self, Dynamic.Atmosphere.KINEMATIC_VISCOSITY, val=1.0, shape=nn, units='ft**2/s'
        )

        self.add_input('ufac', units='unitless', shape=nn, desc='UFAC')

        # form factors
        # user could input these directly or use functions to estimate from geometry

        add_aviary_input(self, Aircraft.Wing.FORM_FACTOR, units='unitless')

        add_aviary_input(
            self,
            Aircraft.Fuselage.FORM_FACTOR,
            units='unitless',
            desc='FFFUS: fuselage form drag factor',
        )

        add_aviary_input(
            self, Aircraft.Nacelle.FORM_FACTOR, shape=num_engine_type, units='unitless'
        )

        add_aviary_input(self, Aircraft.VerticalTail.FORM_FACTOR, units='unitless')

        add_aviary_input(self, Aircraft.HorizontalTail.FORM_FACTOR, units='unitless')

        add_aviary_input(self, Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, units='unitless')

        add_aviary_input(self, Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR, units='unitless')

        # miscellaneous top-level inputs

        add_aviary_input(self, Aircraft.Design.DRAG_COEFFICIENT_INCREMENT, units='unitless')

        add_aviary_input(self, Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT, units='ft**2')

        add_aviary_input(self, Aircraft.Wing.MIN_PRESSURE_LOCATION, units='unitless')

        add_aviary_input(self, Aircraft.Wing.MAX_THICKNESS_LOCATION, units='unitless')

        # geometric user inputs

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')

        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')

        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')

        add_aviary_input(self, Aircraft.Strut.AREA_RATIO, units='unitless')

        # geometric data from sizing

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, units='ft')

        add_aviary_input(self, Aircraft.HorizontalTail.AVERAGE_CHORD, units='ft')

        add_aviary_input(self, Aircraft.VerticalTail.AVERAGE_CHORD, units='ft')

        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')

        add_aviary_input(self, Aircraft.Nacelle.AVG_LENGTH, shape=num_engine_type, units='ft')

        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')

        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')

        add_aviary_input(self, Aircraft.Nacelle.SURFACE_AREA, shape=num_engine_type, units='ft**2')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')

        add_aviary_input(self, Aircraft.VerticalTail.AREA, units='ft**2')

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, units='unitless')

        add_aviary_input(self, Aircraft.Strut.CHORD, units='ft')

        self.add_input(
            'interference_independent_of_shielded_area', units='unitless'
        )  # Is this used?
        (self.add_input('drag_loss_due_to_shielded_wing_area', units='unitless'),)  # Is this used?
        self.add_input(
            'siwb',
            units='unitless',
            desc='SIWB: curve fitting correction factor for Oswald efficiency',
        )

        # outputs
        self.add_output('SA1', units='unitless', shape=nn, desc='SA1: drag param')
        self.add_output('SA2', units='unitless', shape=nn, desc='SA2: drag param')
        self.add_output('SA3', units='unitless', shape=nn, desc='SA3: drag param, not used')
        self.add_output('SA4', units='unitless', shape=nn, desc='SA4: drag param, not used')
        self.add_output('SA5', units='unitless', shape=nn, desc='SA5: drag param')
        self.add_output('SA6', units='unitless', shape=nn, desc='SA6: drag param')
        self.add_output('SA7', units='unitless', shape=nn, desc='SA7: induced drag param')

        self.add_output(
            'cf',
            units='unitless',
            shape=nn,
            desc='CFIN: Skin friction coefficient at Re=1e7',
        )

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials('*', '*', dependent=False)
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(
            'SA1',
            [
                Aircraft.Wing.MIN_PRESSURE_LOCATION,
                Aircraft.Wing.MAX_THICKNESS_LOCATION,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            ],
            method='cs',
        )
        self.declare_partials(
            'SA2',
            [
                Aircraft.Wing.MIN_PRESSURE_LOCATION,
                Aircraft.Wing.MAX_THICKNESS_LOCATION,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
            ],
            method='cs',
        )
        self.declare_partials(
            'SA3',
            [
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
            ],
            method='cs',
        )
        self.declare_partials('SA4', [Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], method='cs')
        self.declare_partials('cf', [Dynamic.Atmosphere.MACH], rows=ar, cols=ar, method='cs')

        # diag partials for SA5-SA7
        self.declare_partials(
            'SA5',
            [
                Dynamic.Atmosphere.MACH,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
                Dynamic.Atmosphere.KINEMATIC_VISCOSITY,
            ],
            rows=ar,
            cols=ar,
            method='cs',
        )
        self.declare_partials(
            'SA6',
            [
                Dynamic.Atmosphere.MACH,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
                Dynamic.Atmosphere.KINEMATIC_VISCOSITY,
            ],
            rows=ar,
            cols=ar,
            method='cs',
        )
        self.declare_partials(
            'SA7',
            [
                Dynamic.Atmosphere.MACH,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
                Dynamic.Atmosphere.KINEMATIC_VISCOSITY,
                'ufac',
            ],
            rows=ar,
            cols=ar,
            method='cs',
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
            Aircraft.Wing.AVERAGE_CHORD,
            Aircraft.HorizontalTail.AVERAGE_CHORD,
            Aircraft.VerticalTail.AVERAGE_CHORD,
            Aircraft.Fuselage.LENGTH,
            Aircraft.Nacelle.AVG_LENGTH,
            Aircraft.HorizontalTail.AREA,
            Aircraft.Fuselage.WETTED_AREA,
            Aircraft.Nacelle.SURFACE_AREA,
            Aircraft.Wing.AREA,
            Aircraft.VerticalTail.AREA,
            'interference_independent_of_shielded_area',
            'drag_loss_due_to_shielded_wing_area',
        ]
        self.declare_partials('SA5', most_params, method='cs')
        self.declare_partials(
            'SA6', [Aircraft.Wing.FORM_FACTOR, Aircraft.Wing.AVERAGE_CHORD], method='cs'
        )
        self.declare_partials(
            'SA7',
            most_params + [Aircraft.Wing.ASPECT_RATIO, Aircraft.Wing.SWEEP] + ['siwb'],
            method='cs',
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
            avg_chord,
            htail_chord,
            vtail_chord,
            fus_len,
            nac_len,
            htail_area,
            fus_SA,
            nacelle_area,
            wing_area,
            vtail_area,
            tc_ratio,
            strut_chord,
            feintwf,
            areashieldwf,
            siwb,
        ) = inputs.values()
        # skin friction coeff at Re = 10**7
        cf = 0.455 / 7**2.58 / (1 + 0.144 * mach**2) ** 0.65

        t = cs.abs(np.tan(deg2rad(sweep_c4)))
        yale05 = (1 - taper_ratio) / (1 + taper_ratio)
        # sweep angle to min pressure point
        dlmps = rad2deg(cs.arctan2(AR * t - 4 * (wing_min_pressure_loc - 0.25) * yale05, AR))
        # sweep angle to max thickness point
        dlmtcx = rad2deg(cs.arctan2(AR * t - 4 * (wing_max_thickness_loc - 0.25) * yale05, AR))
        # sweep angle of the leading edge
        rlmle = cs.arctan2(AR * t + yale05, AR)
        fk = 1 / (1 + yale05 / AR * 4 * taper_ratio**2)

        # Reynolds number per foot
        # here we make a smooth transition between a minimum reli (approximately
        # corresponding to Mach 0.1 at SLS) to help with takeoff. GASP doesn't call AERO
        # before takeoff, so the RELI used corresponds to the cruise point, and this
        # isn't a problem.
        reli_y1 = 700000 * np.ones(self.options['num_nodes'])
        reli_y2 = sos * mach / nu
        sig = sigmoidX(mach, 0.1, mu=0.005)
        reli = (1 - sig) * reli_y1 + sig * reli_y2

        # Re correction factors: fuselage, wing, nacelle, vtail, htail, strut, tip tank
        # protect against Mach 0, any other small Mach should be ok
        dtype = complex if self.under_complex_step else float
        ffre, fwre, fnre, fvtre, fhtre, fstrtre = np.ones(
            (6, self.options['num_nodes']), dtype=dtype
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
        include_strut = self.options[Aircraft.Wing.HAS_STRUT]
        if include_strut:
            fstrtre = (np.log10(reli[good_mask] * strut_chord) / 7) ** -2.6

        # fuselage form drag factor
        # fffus = 1 + 1.5 * (cabin_width / fus_len) ** 1.5 + 7 * (cabin_width / fus_len) ** 3

        # flat plate equivalent areas
        # GASP uses different values of cf for wing, nacelle, fuselage, etc.
        fef = fus_SA * cf * ffre * ff_fus + fe_fus_inc
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
        # In GASP, nacelle is excluded.
        fe = few + fef + fevt + feht + fen + feiwf + festrt + cd0_inc * wing_area

        # wfob = cabin_width / wingspan
        # siwb = 1 - 0.0088 * wfob - 1.7364 * wfob**2 - 2.303 * wfob**3 + 6.0606 * wfob**4

        # wing-free profile drag coefficient
        cdpo = (fe - few) / wing_area

        # Oswald efficiency
        see = 1.0 / (
            (1 / ufac / siwb) + 1.1938 * AR * (cdw0 / (np.cos(deg2rad(sweep_c4))) ** 2 + cdpo)
        )

        # compressibility drag parameters
        # sa1--4 are static, depending only on geometry
        sa1 = (1 + 0.0033 * (4 * dlmps - 3 * dlmtcx)) * (
            1 - 1.4 * tc_ratio - 0.06 * (1 - wing_min_pressure_loc)
        ) - 0.0368
        sa2 = -0.33 * (0.65 - wing_min_pressure_loc) * (1 + 0.0033 * (4 * dlmps - 3 * dlmtcx))
        sa3 = (1.5 - 2 * fk**2 * np.sin(rlmle) ** 2) * tc_ratio ** (5 / 3.0)
        sa4 = 0.75 * tc_ratio

        # profile drag of everything but the wing
        sa5 = cdpo
        # wing profile drag
        sa6 = ff_wing * fwre
        # induced drag
        sa7 = 1.0 / (np.pi * see * AR)

        outputs['SA1'] = sa1
        outputs['SA2'] = sa2
        outputs['SA3'] = sa3
        outputs['SA4'] = sa4
        outputs['SA5'] = sa5
        outputs['SA6'] = sa6
        outputs['SA7'] = sa7
        outputs['cf'] = cf


class AeroSetup(om.Group):
    """Calculations for setting up aero."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare(
            'input_atmos',
            default=False,
            types=bool,
            desc='Directly input speed of sound and kinematic viscosity instead of '
            'computing them with an atmospherics component. For testing.',
        )

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('ratios', WingTailRatios(), promotes=['*'])
        self.add_subsystem('xlifts', Xlifts(num_nodes=nn), promotes=['*'])

        # implements EAERO
        interp = om.MetaModelStructuredComp(method='2D-slinear')
        interp.add_input('bbar', 0.0, units='unitless', training_data=xbbar)
        interp.add_input('hbar', 0.0, units='unitless', training_data=xhbar)
        interp.add_output('sigma', 0.0, units='unitless', training_data=sig1)
        interp.add_output('sigstr', 0.0, units='unitless', training_data=sig2)
        self.add_subsystem('interp', interp, promotes=['*'])

        # Note: It should hold ufac <= 0.975 for BWB
        self.add_subsystem(
            'ufac_calc',
            om.ExecComp(
                'ufac=(1 + lift_ratio)**2 / (sigstr*(lift_ratio/bbar)**2 + 2*sigma*lift_ratio/bbar + 1)',
                lift_ratio={'units': 'unitless', 'shape': nn},
                bbar={'units': 'unitless'},
                sigma={'units': 'unitless'},
                sigstr={'units': 'unitless'},
                ufac={'units': 'unitless', 'shape': nn},
                has_diag_partials=True,
            ),
            promotes=['*'],
        )

        if not self.options['input_atmos']:
            # self.add_subsystem(
            #     "atmos",
            #     USatm1976Comp(num_nodes=nn),
            #     promotes_inputs=[("h", Dynamic.Mission.ALTITUDE)],
            #     promotes_outputs=["rho", Dynamic.Atmosphere.SPEED_OF_SOUND, "viscosity"],
            # )
            self.add_subsystem(
                'kin_visc',
                om.ExecComp(
                    'nu = viscosity / rho',
                    viscosity={'units': 'lbf*s/ft**2', 'shape': nn},
                    rho={'units': 'slug/ft**3', 'shape': nn},
                    nu={'units': 'ft**2/s', 'shape': nn},
                    has_diag_partials=True,
                ),
                promotes=[
                    '*',
                    ('rho', Dynamic.Atmosphere.DENSITY),
                    ('nu', Dynamic.Atmosphere.KINEMATIC_VISCOSITY),
                ],
            )

        self.add_subsystem('form_factor', FormFactorAndSIWB(), promotes=['*'])
        self.add_subsystem('geom', AeroGeom(num_nodes=nn), promotes=['*'])


class BWBAeroSetup(om.Group):
    """Calculations for setting up aero."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare(
            'input_atmos',
            default=False,
            types=bool,
            desc='Directly input speed of sound and kinematic viscosity instead of '
            'computing them with an atmospherics component. For testing.',
        )
        add_aviary_option(self, Aircraft.Design.TYPE)

    def setup(self):
        nn = self.options['num_nodes']
        design_type = self.options[Aircraft.Design.TYPE]

        self.add_subsystem('ratios', WingTailRatios(), promotes=['*'])
        self.add_subsystem('xlifts', Xlifts(num_nodes=nn), promotes=['*'])

        # implements EAERO
        interp = om.MetaModelStructuredComp(method='2D-slinear')
        interp.add_input('bbar_alt', 0.0, units='unitless', training_data=xbbar)
        interp.add_input('hbar', 0.0, units='unitless', training_data=xhbar)
        interp.add_output('sigma', 0.0, units='unitless', training_data=sig1)
        interp.add_output('sigstr', 0.0, units='unitless', training_data=sig2)
        self.add_subsystem('interp', interp, promotes=['*'])

        options = {
            Aircraft.Design.TYPE: design_type,
        }
        self.add_subsystem(
            'ufac_calc',
            UFac(num_nodes=nn, **options),
            promotes=['*'],
        )

        if not self.options['input_atmos']:
            self.add_subsystem(
                'kin_visc',
                om.ExecComp(
                    'nu = viscosity / rho',
                    viscosity={'units': 'lbf*s/ft**2', 'shape': nn},
                    rho={'units': 'slug/ft**3', 'shape': nn},
                    nu={'units': 'ft**2/s', 'shape': nn},
                    has_diag_partials=True,
                ),
                promotes=[
                    '*',
                    ('rho', Dynamic.Atmosphere.DENSITY),
                    ('nu', Dynamic.Atmosphere.KINEMATIC_VISCOSITY),
                ],
            )

        self.add_subsystem('form_factor', BWBFormFactorAndSIWB(), promotes=['*'])
        self.add_subsystem('geom', AeroGeom(num_nodes=nn), promotes=['*'])


class DragCoef(om.ExplicitComponent):
    """GASP lift coefficient calculation for low-speed near-ground flight.

    Drag for low-speed flight is affected by ground effects, landing gear, and flap
    deflection. Flaps are treated separately as an increment so their effects may be
    gradually added/removed over time.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # mission inputs
        add_aviary_input(self, Dynamic.Mission.ALTITUDE, shape=nn, units='ft')
        self.add_input('CL', val=1.0, units='unitless', shape=nn, desc='Lift coefficient')

        # user inputs

        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')

        self.add_input('flap_defl', val=10.0, units='deg', desc='Full flap deflection')

        add_aviary_input(self, Aircraft.Wing.HEIGHT, units='ft')

        self.add_input('airport_alt', val=0.0, units='ft', desc='HPORT: Airport altitude')

        add_aviary_input(self, Aircraft.Wing.FLAP_CHORD_RATIO, units='unitless')

        # from flaps
        self.add_input(
            'dCL_flaps_model',
            val=0.0,
            units='unitless',
            desc='DELCL: Delta CL from flaps model',
        )
        self.add_input(
            'dCD_flaps_model',
            val=0.0,
            units='unitless',
            desc='Delta CD from flaps model',
        )
        self.add_input(
            'dCL_flaps_coef',
            val=1.0,
            units='unitless',
            desc='SIGMTO | SIGMLD: Coefficient applied to delta CL from flaps model',
        )
        self.add_input(
            'CDI_factor',
            val=1.0,
            units='unitless',
            desc='VDEL6T | VDEL6L: Factor applied to induced drag with flaps',
        )

        # from sizing

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, units='ft')

        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')

        # from aero setup
        self.add_input(
            'cf',
            units='unitless',
            shape=nn,
            desc='CFIN: Skin friction coefficient at Re=1e7',
        )
        self.add_input('SA5', units='unitless', shape=nn, desc='SA5: Drag param')
        self.add_input('SA6', units='unitless', shape=nn, desc='SA6: Drag param')
        self.add_input('SA7', units='unitless', shape=nn, desc='SA7: Drag param')

        self.add_output('CD_base', units='unitless', shape=nn, desc='Drag coefficient')
        self.add_output(
            'dCD_flaps_full',
            units='unitless',
            shape=nn,
            desc='CD increment with full flap deflection',
        )
        self.add_output(
            'dCD_gear_full',
            units='unitless',
            shape=nn,
            desc='CD increment with landing gear down',
        )

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials('*', '*', dependent=False)
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials('CD_base', ['*'], method='cs')
        self.declare_partials(
            'CD_base',
            [Dynamic.Mission.ALTITUDE, 'CL', 'cf', 'SA5', 'SA6', 'SA7'],
            rows=ar,
            cols=ar,
            method='cs',
        )
        # self.declare_partials(
        #     "CD_base", [Mission.Design.GROSS_MASS, "dCD_flaps_model", "wing_area"], val=0
        # )

        self.declare_partials('dCD_flaps_full', ['dCD_flaps_model'], val=1)

        self.declare_partials(
            'dCD_gear_full',
            [Mission.Design.GROSS_MASS, Aircraft.Wing.AREA, 'flap_defl'],
            method='cs',
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

        outputs['CD_base'] = cd0 + cdi + dcd_ground
        outputs['dCD_flaps_full'] = dCD_flaps_model  # same as inputs['dCD_flaps_model']
        outputs['dCD_gear_full'] = dcd_gear


class DragCoefClean(om.ExplicitComponent):
    """Clean drag coefficient for high-speed flight."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # mission inputs
        add_aviary_input(self, Dynamic.Atmosphere.MACH, shape=nn, units='unitless')
        self.add_input('CL', val=1.0, units='unitless', shape=nn, desc='Lift coefficient')

        # user inputs
        add_aviary_input(self, Aircraft.Design.DRAG_DIVERGENCE_SHIFT, units='unitless')
        add_aviary_input(self, Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, units='unitless')

        # from aero setup
        self.add_input(
            'cf',
            units='unitless',
            shape=nn,
            desc='CFIN: Skin friction coefficient at Re=1e7',
        )
        self.add_input('SA1', units='unitless', shape=nn, desc='SA1: Drag param')
        self.add_input('SA2', units='unitless', shape=nn, desc='SA2: Drag param')
        self.add_input('SA5', units='unitless', shape=nn, desc='SA5: Drag param')
        self.add_input('SA6', units='unitless', shape=nn, desc='SA6: Drag param')
        self.add_input('SA7', units='unitless', shape=nn, desc='SA7: Drag param')

        self.add_output('CD', units='unitless', shape=nn, desc='Drag coefficient')

    def setup_partials(self):
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(
            'CD',
            [Dynamic.Atmosphere.MACH, 'CL', 'cf', 'SA1', 'SA2', 'SA5', 'SA6', 'SA7'],
            rows=ar,
            cols=ar,
            method='cs',
        )
        self.declare_partials('CD', [Aircraft.Design.DRAG_DIVERGENCE_SHIFT], method='cs')

    def compute(self, inputs, outputs):
        (
            mach,
            CL,
            div_drag_supercrit,
            subsonic_factor,
            supersonic_factor,
            lift_factor,
            zero_lift_factor,
            cf,
            SA1,
            SA2,
            SA5,
            SA6,
            SA7,
        ) = inputs.values()

        mach_div = SA1 + SA2 * CL + div_drag_supercrit

        sig = sigmoidX(mach, mach_div, mu=0.005)
        delcdm = sig * (10 * (mach - mach_div) ** 3)

        # delcdm = np.zeros_like(mach)
        # mask = np.bitwise_and(mach >= mach_div, mach > 0.6)
        # delcdm[mask] = 10 * (mach[mask] - mach_div[mask]) ** 3

        # profile drag
        cd0 = SA5 + SA6 * cf
        # induced drag
        cdi = SA7 * CL**2

        CD = cd0 * zero_lift_factor + cdi * lift_factor + delcdm

        # scale drag
        idx_sup = np.where(mach >= 1.0)
        CD_scaled = CD * subsonic_factor
        CD_scaled[idx_sup] = CD[idx_sup] * supersonic_factor

        outputs['CD'] = CD_scaled


class GroundEffect(om.ExplicitComponent):
    """Factor of CL due to ground effect."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # mission inputs
        add_aviary_input(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=nn, units='deg')
        add_aviary_input(self, Dynamic.Mission.ALTITUDE, shape=nn, units='ft')
        self.add_input(
            'lift_curve_slope', units='unitless', shape=nn, desc='CLAW: Lift-curve slope'
        )

        # user inputs
        add_aviary_input(self, Aircraft.Wing.ZERO_LIFT_ANGLE, units='deg')
        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.HEIGHT, units='ft')
        self.add_input('airport_alt', val=0.0, units='ft', desc='HPORT: Airport altitude')
        self.add_input('flap_defl', val=10.0, units='deg', desc='Full flap deflection')
        add_aviary_input(self, Aircraft.Wing.FLAP_CHORD_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')

        # from flaps
        self.add_input(
            'dCL_flaps_model',
            val=0.0,
            units='unitless',
            desc='Delta CL from flaps model',
        )

        # from sizing
        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, units='ft')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        self.add_output(
            'kclge', units='unitless', shape=nn, desc='KCLGE: factor of CL due to ground effect'
        )

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials('*', '*', dependent=False)
        ar = np.arange(self.options['num_nodes'])

        dynvars = [
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            Dynamic.Mission.ALTITUDE,
            'lift_curve_slope',
        ]

        self.declare_partials('kclge', ['*'], method='cs')
        self.declare_partials('kclge', dynvars, rows=ar, cols=ar, method='cs')

    def compute(self, inputs, outputs):
        (
            alpha,
            alt,
            lift_curve_slope,
            alpha0,
            sweep_c4,
            AR,
            wing_height,
            airport_alt,
            flap_defl,
            flap_chord_ratio,
            taper_ratio,
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
        # make sure kclge >= 1.0
        kclge = np.clip(kclge, 1.0, None)

        hob = hac / wingspan
        for i_node in range(self.options['num_nodes']):
            if hob[i_node] >= 10.0:
                kclge[i_node] = 1.0

        outputs['kclge'] = kclge


class LiftCoeff(om.ExplicitComponent):
    """GASP lift coefficient calculation for low-speed near-ground flight."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=nn, units='deg')
        self.add_input(
            'lift_curve_slope', units='unitless', shape=nn, desc='CLAW: Lift-curve slope'
        )
        self.add_input('lift_ratio', units='unitless', shape=nn, desc='BARL: Lift ratio')

        add_aviary_input(self, Aircraft.Wing.ZERO_LIFT_ANGLE, units='deg')

        # from flaps
        self.add_input(
            'CL_max_flaps',
            units='unitless',
            desc='CLMWTO | CLMWLD: Max lift coefficient from flaps model (takeoff | landing)',
        )
        self.add_input(
            'dCL_flaps_model',
            val=0.0,
            units='unitless',
            desc='Delta CL from flaps model',
        )

        self.add_input(
            'kclge', units='unitless', shape=nn, desc='factor of CL due to ground effect'
        )
        self.add_input(
            'flap_factor', shape=nn, units='unitless', desc='factor of CL due to flap deployment'
        )

        self.add_output('CL_base', units='unitless', shape=nn, desc='Base lift coefficient')
        self.add_output(
            'dCL_flaps_full',
            units='unitless',
            shape=nn,
            desc='CL increment with full flap deflection',
        )
        self.add_output('alpha_stall', units='deg', shape=nn, desc='Stall angle of attack')
        self.add_output('CL_max', units='unitless', shape=nn, desc='Max lift coefficient')

        self.add_output(
            'CL_full_flaps',
            units='unitless',
            shape=nn,
            desc='lift coefficient with full flaps deployed',
        )
        self.add_output('CL', units='unitless', shape=nn, desc='lift coefficient')

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials('*', '*', dependent=False)
        ar = np.arange(self.options['num_nodes'])

        dynvars = [
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            'lift_curve_slope',
            'lift_ratio',
            'kclge',
        ]

        self.declare_partials('CL_base', ['*'])
        self.declare_partials('CL_base', dynvars, rows=ar, cols=ar)

        self.declare_partials('dCL_flaps_full', ['dCL_flaps_model'])
        self.declare_partials('dCL_flaps_full', ['lift_ratio'], rows=ar, cols=ar)

        self.declare_partials('alpha_stall', ['*'])
        self.declare_partials('alpha_stall', dynvars, rows=ar, cols=ar)

        self.declare_partials('CL_max', ['CL_max_flaps'])
        self.declare_partials('CL_max', ['lift_ratio'], rows=ar, cols=ar)

        self.declare_partials('CL_full_flaps', ['*'])
        self.declare_partials('CL_full_flaps', dynvars, rows=ar, cols=ar)
        self.declare_partials('CL_full_flaps', ['dCL_flaps_model'])
        self.declare_partials('CL_full_flaps', ['lift_ratio'], rows=ar, cols=ar)

        self.declare_partials('CL', ['*'])
        self.declare_partials('CL', dynvars, rows=ar, cols=ar)
        self.declare_partials('CL', ['dCL_flaps_model'])
        self.declare_partials('CL', ['lift_ratio', 'flap_factor'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        (
            alpha,
            lift_curve_slope,
            lift_ratio,
            alpha0,
            CL_max_flaps,
            dCL_flaps_model,
            kclge,
            flap_factor,
        ) = inputs.values()

        # clw_base = kclge * lift_curve_slope * deg2rad(alpha - alpha0)
        # clw = clw_base + dCL_flaps_model

        outputs['CL_base'] = kclge * lift_curve_slope * deg2rad(alpha - alpha0) * (1 + lift_ratio)
        outputs['dCL_flaps_full'] = dCL_flaps_model * (1 + lift_ratio)
        outputs['alpha_stall'] = (
            rad2deg((CL_max_flaps - dCL_flaps_model) / (kclge * lift_curve_slope)) + alpha0
        )
        outputs['CL_max'] = CL_max_flaps * (1 + lift_ratio)

        outputs['CL_full_flaps'] = outputs['CL_base'] + outputs['dCL_flaps_full']
        outputs['CL'] = outputs['CL_base'] + flap_factor * outputs['dCL_flaps_full']

    def compute_partials(self, inputs, J):
        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        lift_curve_slope = inputs['lift_curve_slope']
        lift_ratio = inputs['lift_ratio']
        alpha0 = inputs[Aircraft.Wing.ZERO_LIFT_ANGLE]
        CL_max_flaps = inputs['CL_max_flaps']

        dCL_flaps_model = inputs['dCL_flaps_model']
        kclge = inputs['kclge']
        flap_factor = inputs['flap_factor']

        J['CL_base', 'kclge'] = lift_curve_slope * deg2rad(alpha - alpha0) * (1 + lift_ratio)
        J['CL_base', 'lift_curve_slope'] = kclge * deg2rad(alpha - alpha0) * (1 + lift_ratio)
        J['CL_base', Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
            kclge * lift_curve_slope * np.pi / 180.0 * (1 + lift_ratio)
        )
        J['CL_base', Aircraft.Wing.ZERO_LIFT_ANGLE] = (
            -kclge * lift_curve_slope * np.pi / 180.0 * (1 + lift_ratio)
        )
        J['CL_base', 'lift_ratio'] = kclge * lift_curve_slope * deg2rad(alpha - alpha0)

        J['dCL_flaps_full', 'dCL_flaps_model'] = 1 + lift_ratio
        J['dCL_flaps_full', 'lift_ratio'] = dCL_flaps_model

        J['alpha_stall', 'CL_max_flaps'] = 180.0 / np.pi / (kclge * lift_curve_slope)
        J['alpha_stall', 'dCL_flaps_model'] = -180.0 / np.pi / (kclge * lift_curve_slope)
        J['alpha_stall', 'kclge'] = -rad2deg(
            (CL_max_flaps - dCL_flaps_model) / (kclge**2 * lift_curve_slope)
        )
        J['alpha_stall', 'lift_curve_slope'] = -rad2deg(
            (CL_max_flaps - dCL_flaps_model) / (kclge * lift_curve_slope**2)
        )
        J['alpha_stall', Aircraft.Wing.ZERO_LIFT_ANGLE] = 1

        J['CL_max', 'CL_max_flaps'] = 1 + lift_ratio
        J['CL_max', 'lift_ratio'] = CL_max_flaps

        J['CL_full_flaps', 'kclge'] = J['CL_base', 'kclge']
        J['CL_full_flaps', 'lift_curve_slope'] = J['CL_base', 'lift_curve_slope']
        J['CL_full_flaps', Dynamic.Vehicle.ANGLE_OF_ATTACK] = J[
            'CL_base', Dynamic.Vehicle.ANGLE_OF_ATTACK
        ]
        J['CL_full_flaps', Aircraft.Wing.ZERO_LIFT_ANGLE] = J[
            'CL_base', Aircraft.Wing.ZERO_LIFT_ANGLE
        ]
        J['CL_full_flaps', 'lift_ratio'] = (
            J['CL_base', 'lift_ratio'] + J['dCL_flaps_full', 'lift_ratio']
        )
        J['CL_full_flaps', 'dCL_flaps_model'] = J['dCL_flaps_full', 'dCL_flaps_model']

        J['CL', 'kclge'] = J['CL_base', 'kclge']
        J['CL', 'lift_curve_slope'] = J['CL_base', 'lift_curve_slope']
        J['CL', Dynamic.Vehicle.ANGLE_OF_ATTACK] = J['CL_base', Dynamic.Vehicle.ANGLE_OF_ATTACK]
        J['CL', Aircraft.Wing.ZERO_LIFT_ANGLE] = J['CL_base', Aircraft.Wing.ZERO_LIFT_ANGLE]

        J['CL', 'flap_factor'] = dCL_flaps_model * (1 + lift_ratio)
        J['CL', 'lift_ratio'] = (
            J['CL_base', 'lift_ratio'] + flap_factor * J['dCL_flaps_full', 'lift_ratio']
        )
        J['CL', 'dCL_flaps_model'] = flap_factor * (1 + lift_ratio)


class BWBLiftCoeff(om.ExplicitComponent):
    """GASP lift coefficient calculation for low-speed near-ground flight for BWB."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=nn, units='deg')
        self.add_input(
            'lift_curve_slope', units='unitless', shape=nn, desc='CLAW: Lift-curve slope'
        )

        add_aviary_input(self, Aircraft.Wing.ZERO_LIFT_ANGLE, units='deg')

        # from flaps
        self.add_input(
            'CL_max_flaps',
            units='unitless',
            desc='CLMWTO | CLMWLD: Max lift coefficient from flaps model (takeoff | landing)',
        )
        self.add_input(
            'dCL_flaps_model',
            val=0.0,
            units='unitless',
            desc='Delta CL from flaps model',
        )

        self.add_input(
            'kclge', units='unitless', shape=nn, desc='factor of CL due to ground effect'
        )
        self.add_input(
            'flap_factor', shape=nn, units='unitless', desc='factor of CL due to flap deployment'
        )

        self.add_input(
            'body_lift_curve_slope',
            units='unitless',
            shape=nn,
            desc='CLALPH_B: Lift-curve slope of fuselage for the given Mach',
        )
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2', desc='SREF')
        add_aviary_input(self, Aircraft.Wing.EXPOSED_AREA, units='ft**2', desc='SW_EXP')
        add_aviary_input(self, Aircraft.Fuselage.PLANFORM_AREA, units='ft**2', desc='SPF_BODY')

        self.add_output('CL_base', units='unitless', shape=nn, desc='Base lift coefficient')
        self.add_output(
            'dCL_flaps_full',
            units='unitless',
            shape=nn,
            desc='CL increment with full flap deflection',
        )
        self.add_output('alpha_stall', units='deg', shape=nn, desc='Stall angle of attack')
        self.add_output('CL_max', units='unitless', shape=nn, desc='Max lift coefficient')

        self.add_output(
            'CL_full_flaps',
            units='unitless',
            shape=nn,
            desc='lift coefficient with full flaps deployed',
        )
        self.add_output('CL', units='unitless', shape=nn, desc='lift coefficient')

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(
            'CL_base',
            [
                'kclge',
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Aircraft.Wing.ZERO_LIFT_ANGLE,
            ],
        )
        self.declare_partials(
            'CL_base',
            [
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                'lift_curve_slope',
                'kclge',
            ],
            rows=ar,
            cols=ar,
        )

        self.declare_partials('dCL_flaps_full', ['dCL_flaps_model'])

        self.declare_partials('alpha_stall', ['lift_curve_slope', 'kclge'], rows=ar, cols=ar)
        self.declare_partials(
            'alpha_stall',
            [
                'CL_max_flaps',
                'dCL_flaps_model',
                Aircraft.Wing.ZERO_LIFT_ANGLE,
            ],
        )

        self.declare_partials('CL_max', ['CL_max_flaps'])

        self.declare_partials(
            'CL_full_flaps',
            [
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                'lift_curve_slope',
                'kclge',
                'body_lift_curve_slope',
            ],
            rows=ar,
            cols=ar,
        )
        self.declare_partials(
            'CL_full_flaps',
            [
                'dCL_flaps_model',
                Aircraft.Wing.ZERO_LIFT_ANGLE,
                Aircraft.Wing.AREA,
                Aircraft.Wing.EXPOSED_AREA,
                Aircraft.Fuselage.PLANFORM_AREA,
            ],
        )

        # self.declare_partials('CL', ['*'])
        self.declare_partials(
            'CL',
            [
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                'lift_curve_slope',
                'kclge',
                'flap_factor',
                'body_lift_curve_slope',
            ],
            rows=ar,
            cols=ar,
        )
        self.declare_partials(
            'CL',
            [
                'dCL_flaps_model',
                Aircraft.Wing.ZERO_LIFT_ANGLE,
                Aircraft.Wing.AREA,
                Aircraft.Wing.EXPOSED_AREA,
                Aircraft.Fuselage.PLANFORM_AREA,
            ],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        lift_curve_slope = inputs['lift_curve_slope']
        alpha0 = inputs[Aircraft.Wing.ZERO_LIFT_ANGLE]
        CL_max_flaps = inputs['CL_max_flaps']
        dCL_flaps_model = inputs['dCL_flaps_model']
        kclge = inputs['kclge']
        flap_factor = inputs['flap_factor']
        body_lift_curve_slope = inputs['body_lift_curve_slope']
        wing_area = inputs[Aircraft.Wing.AREA]
        exp_wing_area = inputs[Aircraft.Wing.EXPOSED_AREA]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]

        # clw_base = kclge * lift_curve_slope * deg2rad(alpha - alpha0)
        # clw = clw_base + dCL_flaps_model

        # This is actually CLw_base
        outputs['CL_base'] = kclge * lift_curve_slope * deg2rad(alpha - alpha0)
        outputs['dCL_flaps_full'] = dCL_flaps_model
        alpha_stall = (
            rad2deg((CL_max_flaps - dCL_flaps_model) / (kclge * lift_curve_slope)) + alpha0
        )
        outputs['alpha_stall'] = alpha_stall
        if any(x > 0.0 for x in alpha.real - alpha_stall.real):
            if verbosity > Verbosity.BRIEF:
                warnings.warn(
                    f'Some angle of attack {alpha} might be greater than alpha stall {alpha_stall}.'
                )
        outputs['CL_max'] = CL_max_flaps

        CLw_full_flaps = outputs['CL_base'] + outputs['dCL_flaps_full']
        # CL when flaps is partially deployed
        CLw_partial_flaps = outputs['CL_base'] + flap_factor * outputs['dCL_flaps_full']

        CL_body = planform / wing_area * kclge * body_lift_curve_slope * deg2rad(alpha - alpha0)

        outputs['CL_full_flaps'] = exp_wing_area / wing_area * CLw_full_flaps + CL_body
        # CL when flaps is partially deployed
        outputs['CL'] = exp_wing_area / wing_area * CLw_partial_flaps + CL_body

    def compute_partials(self, inputs, J):
        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        lift_curve_slope = inputs['lift_curve_slope']
        alpha0 = inputs[Aircraft.Wing.ZERO_LIFT_ANGLE]
        CL_max_flaps = inputs['CL_max_flaps']
        dCL_flaps_model = inputs['dCL_flaps_model']
        kclge = inputs['kclge']
        flap_factor = inputs['flap_factor']

        body_lift_curve_slope = inputs['body_lift_curve_slope']
        wing_area = inputs[Aircraft.Wing.AREA]
        exp_wing_area = inputs[Aircraft.Wing.EXPOSED_AREA]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]

        J['CL_base', 'kclge'] = lift_curve_slope * deg2rad(alpha - alpha0)
        J['CL_base', 'lift_curve_slope'] = kclge * deg2rad(alpha - alpha0)
        J['CL_base', Dynamic.Vehicle.ANGLE_OF_ATTACK] = kclge * lift_curve_slope * np.pi / 180.0
        J['CL_base', Aircraft.Wing.ZERO_LIFT_ANGLE] = -kclge * lift_curve_slope * np.pi / 180.0

        J['dCL_flaps_full', 'dCL_flaps_model'] = 1

        J['alpha_stall', 'CL_max_flaps'] = 180.0 / np.pi / (kclge * lift_curve_slope)
        J['alpha_stall', 'dCL_flaps_model'] = -180.0 / np.pi / (kclge * lift_curve_slope)
        J['alpha_stall', 'kclge'] = -rad2deg(
            (CL_max_flaps - dCL_flaps_model) / (kclge**2 * lift_curve_slope)
        )
        J['alpha_stall', 'lift_curve_slope'] = -rad2deg(
            (CL_max_flaps - dCL_flaps_model) / (kclge * lift_curve_slope**2)
        )
        J['alpha_stall', Aircraft.Wing.ZERO_LIFT_ANGLE] = 1

        J['CL_max', 'CL_max_flaps'] = 1

        # with planform/wing_area factor
        # CL_body = planform / wing_area * kclge * body_lift_curve_slope * deg2rad(alpha - alpha0)

        dCL_body_dplanform = 1 / wing_area * kclge * body_lift_curve_slope * deg2rad(alpha - alpha0)
        dCL_body_dwing_area = (
            -planform / wing_area**2 * kclge * body_lift_curve_slope * deg2rad(alpha - alpha0)
        )
        dCL_body_dkclge = planform / wing_area * body_lift_curve_slope * deg2rad(alpha - alpha0)
        dCL_body_dbody_lift_curve_slope = planform / wing_area * kclge * deg2rad(alpha - alpha0)
        dCL_body_dalpha = planform / wing_area * kclge * body_lift_curve_slope * np.pi / 180.0
        dCL_body_dalpha0 = -planform / wing_area * kclge * body_lift_curve_slope * np.pi / 180.0

        # with exp_wing_area/wing_area factor
        # CLw_full_flaps = exp_wing_area / wing_area * (kclge * lift_curve_slope * deg2rad(alpha - alpha0)
        #                  + dCL_flaps_model)

        dCLw_full_flaps_dkclge = (
            exp_wing_area / wing_area * lift_curve_slope * deg2rad(alpha - alpha0)
        )
        dCLw_full_flaps_dlift_curve_slope = (
            exp_wing_area / wing_area * kclge * deg2rad(alpha - alpha0)
        )
        dCLw_full_flaps_dalpha = (
            exp_wing_area / wing_area * kclge * lift_curve_slope * np.pi / 180.0
        )
        dCLw_full_flaps_dalpha0 = (
            -exp_wing_area / wing_area * kclge * lift_curve_slope * np.pi / 180.0
        )
        dCLw_full_flaps_ddCL_flaps_model = exp_wing_area / wing_area

        CL_base = kclge * lift_curve_slope * deg2rad(alpha - alpha0)
        dCLw_full_flaps = dCL_flaps_model
        CLw_full_flaps = CL_base + dCLw_full_flaps

        J['CL_full_flaps', 'kclge'] = dCLw_full_flaps_dkclge + dCL_body_dkclge
        J['CL_full_flaps', 'lift_curve_slope'] = dCLw_full_flaps_dlift_curve_slope
        J['CL_full_flaps', Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
            dCLw_full_flaps_dalpha + dCL_body_dalpha
        )
        J['CL_full_flaps', Aircraft.Wing.ZERO_LIFT_ANGLE] = (
            dCLw_full_flaps_dalpha0 + dCL_body_dalpha0
        )
        J['CL_full_flaps', 'dCL_flaps_model'] = dCLw_full_flaps_ddCL_flaps_model

        J['CL_full_flaps', Aircraft.Fuselage.PLANFORM_AREA] = dCL_body_dplanform
        J['CL_full_flaps', Aircraft.Wing.AREA] = (
            -exp_wing_area / wing_area**2 * CLw_full_flaps + dCL_body_dwing_area
        )
        J['CL_full_flaps', Aircraft.Wing.EXPOSED_AREA] = 1 / wing_area * CLw_full_flaps
        J['CL_full_flaps', Aircraft.Fuselage.PLANFORM_AREA] = dCL_body_dplanform
        J['CL_full_flaps', 'body_lift_curve_slope'] = dCL_body_dbody_lift_curve_slope

        # CL when flaps is partially deployed
        # with exp_wing_area/wing_area factor
        # CLw_partial_flaps = exp_wing_area / wing_area * (kclge * lift_curve_slope * deg2rad(alpha - alpha0)
        #                   + flap_factor * dCL_flaps_model)

        dCLw_partial_flaps_dkclge = (
            exp_wing_area / wing_area * lift_curve_slope * deg2rad(alpha - alpha0)
        )
        dCLw_partial_flaps_dlift_curve_slope = (
            exp_wing_area / wing_area * kclge * deg2rad(alpha - alpha0)
        )
        dCLw_partial_flaps_dalpha = (
            exp_wing_area / wing_area * kclge * lift_curve_slope * np.pi / 180.0
        )
        dCLw_partial_flaps_dalpha0 = (
            -exp_wing_area / wing_area * kclge * lift_curve_slope * np.pi / 180.0
        )
        dCLw_partial_flaps_ddCL_flaps_model = exp_wing_area / wing_area * flap_factor
        dCLw_partial_flaps_dflap_factor = exp_wing_area / wing_area * dCL_flaps_model

        dCLw_partial_flaps = flap_factor * dCL_flaps_model
        CLw_partial_flaps = CL_base + dCLw_partial_flaps

        J['CL', 'kclge'] = dCLw_partial_flaps_dkclge + dCL_body_dkclge
        J['CL', 'lift_curve_slope'] = dCLw_partial_flaps_dlift_curve_slope
        J['CL', Dynamic.Vehicle.ANGLE_OF_ATTACK] = dCLw_partial_flaps_dalpha + dCL_body_dalpha
        J['CL', Aircraft.Wing.ZERO_LIFT_ANGLE] = dCLw_partial_flaps_dalpha0 + dCL_body_dalpha0
        J['CL', 'dCL_flaps_model'] = dCLw_partial_flaps_ddCL_flaps_model

        J['CL', Aircraft.Fuselage.PLANFORM_AREA] = dCL_body_dplanform
        J['CL', Aircraft.Wing.AREA] = (
            -exp_wing_area / wing_area**2 * CLw_partial_flaps + dCL_body_dwing_area
        )
        J['CL', Aircraft.Wing.EXPOSED_AREA] = 1 / wing_area * CLw_partial_flaps
        J['CL', Aircraft.Fuselage.PLANFORM_AREA] = dCL_body_dplanform
        J['CL', 'body_lift_curve_slope'] = dCL_body_dbody_lift_curve_slope

        J['CL', 'flap_factor'] = dCLw_partial_flaps_dflap_factor


class LiftCoeffClean(om.ExplicitComponent):
    """Clean wing lift coefficient for high-speed flight."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare(
            'output_alpha',
            default=False,
            types=bool,
            desc='If True, output alpha for a given input CL',
        )

    def setup(self):
        nn = self.options['num_nodes']

        if self.options['output_alpha']:
            add_aviary_output(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=nn, units='deg')
            self.add_input('CL', val=1.0, units='unitless', shape=nn, desc='Lift coefficient')
        else:
            add_aviary_input(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=nn, units='deg')
            self.add_output('CL', val=1.0, units='unitless', shape=nn, desc='Lift coefficient')

        self.add_input(
            'lift_curve_slope', units='unitless', shape=nn, desc='CLAW: Lift-curve slope'
        )
        self.add_input('lift_ratio', units='unitless', shape=nn, desc='BARL: Lift ratio')

        add_aviary_input(self, Aircraft.Wing.ZERO_LIFT_ANGLE, units='deg', desc='ALPHL0')

        add_aviary_input(self, Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP, units='unitless')

        self.add_output('alpha_stall', shape=nn, units='deg', desc='Stall angle of attack')
        self.add_output('CL_max', units='unitless', shape=nn, desc='Max lift coefficient')

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials('*', '*', dependent=False)
        ar = np.arange(self.options['num_nodes'])

        if self.options['output_alpha']:
            self.declare_partials(
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                ['CL', 'lift_ratio', 'lift_curve_slope'],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                [Aircraft.Wing.ZERO_LIFT_ANGLE],
            )
        else:
            self.declare_partials(
                'CL',
                ['lift_curve_slope', Dynamic.Vehicle.ANGLE_OF_ATTACK, 'lift_ratio'],
                rows=ar,
                cols=ar,
            )
            self.declare_partials('CL', [Aircraft.Wing.ZERO_LIFT_ANGLE])

        self.declare_partials('alpha_stall', ['lift_curve_slope'], rows=ar, cols=ar)
        self.declare_partials(
            'alpha_stall',
            [
                Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP,
                Aircraft.Wing.ZERO_LIFT_ANGLE,
            ],
        )

        self.declare_partials('CL_max', ['lift_ratio'], rows=ar, cols=ar)
        self.declare_partials('CL_max', [Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP])

    def compute(self, inputs, outputs):
        _, lift_curve_slope, lift_ratio, alpha0, CL_max_flaps = inputs.values()
        if self.options['output_alpha']:
            CL = inputs['CL']
            clw = CL / (1 + lift_ratio)
            outputs[Dynamic.Vehicle.ANGLE_OF_ATTACK] = rad2deg(clw / lift_curve_slope) + alpha0
        else:
            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
            outputs['CL'] = lift_curve_slope * deg2rad(alpha - alpha0) * (1 + lift_ratio)

        outputs['alpha_stall'] = rad2deg(CL_max_flaps / lift_curve_slope) + alpha0
        outputs['CL_max'] = CL_max_flaps * (1 + lift_ratio)

    def compute_partials(self, inputs, J):
        lift_curve_slope = inputs['lift_curve_slope']
        lift_ratio = inputs['lift_ratio']
        alpha0 = inputs[Aircraft.Wing.ZERO_LIFT_ANGLE]
        CL_max_flaps = inputs[Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP]

        J['alpha_stall', Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP] = (
            180.0 / np.pi / lift_curve_slope
        )
        J['alpha_stall', 'lift_curve_slope'] = -180.0 / np.pi * CL_max_flaps / lift_curve_slope**2
        J['alpha_stall', Aircraft.Wing.ZERO_LIFT_ANGLE] = 1
        J['CL_max', Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP] = 1 + lift_ratio
        J['CL_max', 'lift_ratio'] = CL_max_flaps

        if self.options['output_alpha']:
            CL = inputs['CL']
            clw = CL / (1 + lift_ratio)
            # outputs[Dynamic.Vehicle.ANGLE_OF_ATTACK] = rad2deg(clw / lift_curve_slope) + alpha0
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, Aircraft.Wing.ZERO_LIFT_ANGLE] = 1
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, 'lift_curve_slope'] = (
                -180.0 / np.pi * clw / lift_curve_slope**2
            )
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, 'CL'] = (
                180.0 / np.pi / (1 + lift_ratio) / lift_curve_slope
            )
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, 'lift_ratio'] = (
                -180.0 / np.pi * CL / (1 + lift_ratio) ** 2 / lift_curve_slope
            )
        else:
            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
            J['CL', Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
                lift_curve_slope * np.pi / 180.0 * (1 + lift_ratio)
            )
            J['CL', 'lift_curve_slope'] = deg2rad(alpha - alpha0) * (1 + lift_ratio)
            J['CL', 'lift_ratio'] = lift_curve_slope * deg2rad(alpha - alpha0)
            J['CL', Aircraft.Wing.ZERO_LIFT_ANGLE] = (
                -lift_curve_slope * np.pi / 180.0 * (1 + lift_ratio)
            )


class BWBLiftCoeffClean(om.ExplicitComponent):
    """Clean wing lift coefficient for high-speed flight of BWB."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare(
            'output_alpha',
            default=False,
            types=bool,
            desc='If True, output alpha for a given input CL',
        )
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        nn = self.options['num_nodes']

        if self.options['output_alpha']:
            self.add_input('CL', val=1.0, units='unitless', shape=nn, desc='Lift coefficient')
        else:
            add_aviary_input(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=nn, units='deg')

        self.add_input(
            'lift_curve_slope', units='unitless', shape=nn, desc='CLAW: Lift-curve slope'
        )
        self.add_input(
            'body_lift_curve_slope',
            units='unitless',
            shape=nn,
            desc='CLALPH_B: Lift-curve slope of fuselage for the given Mach',
        )
        add_aviary_input(self, Aircraft.Wing.ZERO_LIFT_ANGLE, units='deg', desc='ALPHL0')
        add_aviary_input(self, Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP, units='unitless')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2', desc='SREF')
        add_aviary_input(self, Aircraft.Wing.EXPOSED_AREA, units='ft**2', desc='SW_EXP')
        add_aviary_input(self, Aircraft.Fuselage.PLANFORM_AREA, units='ft**2', desc='SPF_BODY')

        if self.options['output_alpha']:
            self.add_output('mod_lift_curve_slope', units='unitless', shape=nn)
            add_aviary_output(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=nn, units='deg')
        else:
            self.add_output('CL', val=1.0, units='unitless', shape=nn, desc='Lift coefficient')
        self.add_output('alpha_stall', shape=nn, desc='Stall angle of attack')
        self.add_output('CL_max', units='unitless', shape=nn, desc='Max lift coefficient')

    def setup_partials(self):
        # self.declare_coloring(method="cs", show_summary=False)
        self.declare_partials('*', '*', dependent=False)
        ar = np.arange(self.options['num_nodes'])

        if self.options['output_alpha']:
            self.declare_partials(
                'mod_lift_curve_slope',
                [
                    Aircraft.Wing.AREA,
                    Aircraft.Wing.EXPOSED_AREA,
                    Aircraft.Fuselage.PLANFORM_AREA,
                ],
            )
            self.declare_partials(
                'mod_lift_curve_slope',
                [
                    'lift_curve_slope',
                    'body_lift_curve_slope',
                ],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                [
                    'CL',
                    'lift_curve_slope',
                    'body_lift_curve_slope',
                ],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                [
                    Aircraft.Wing.AREA,
                    Aircraft.Wing.EXPOSED_AREA,
                    Aircraft.Fuselage.PLANFORM_AREA,
                    Aircraft.Wing.ZERO_LIFT_ANGLE,
                ],
            )
        else:
            self.declare_partials(
                'CL',
                [
                    'lift_curve_slope',
                    'body_lift_curve_slope',
                    Dynamic.Vehicle.ANGLE_OF_ATTACK,
                ],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                'CL',
                [
                    Aircraft.Wing.AREA,
                    Aircraft.Wing.EXPOSED_AREA,
                    Aircraft.Fuselage.PLANFORM_AREA,
                    Aircraft.Wing.ZERO_LIFT_ANGLE,
                ],
            )

        self.declare_partials('alpha_stall', ['lift_curve_slope'], rows=ar, cols=ar)
        self.declare_partials(
            'alpha_stall',
            [
                Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP,
                Aircraft.Wing.ZERO_LIFT_ANGLE,
                Aircraft.Wing.AREA,
                Aircraft.Wing.EXPOSED_AREA,
            ],
        )

        self.declare_partials('CL_max', [Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP])

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        lift_curve_slope = inputs['lift_curve_slope']
        body_lift_curve_slope = inputs['body_lift_curve_slope']
        alpha0 = inputs[Aircraft.Wing.ZERO_LIFT_ANGLE]
        CL_max_flaps = inputs[Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP]
        wing_area = inputs[Aircraft.Wing.AREA]
        exp_wing_area = inputs[Aircraft.Wing.EXPOSED_AREA]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]

        if self.options['output_alpha']:
            CL = inputs['CL']
            # clw = wing_area / exp_wing_area * CL
            mod_lift_curve_slope = (
                exp_wing_area / wing_area * lift_curve_slope
                + planform / wing_area * body_lift_curve_slope
            )
            outputs['mod_lift_curve_slope'] = mod_lift_curve_slope
            alpha = rad2deg(CL / mod_lift_curve_slope) + alpha0
            outputs[Dynamic.Vehicle.ANGLE_OF_ATTACK] = alpha
        else:
            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
            clw = lift_curve_slope * np.pi / 180.0 * (alpha - alpha0)
            outputs['CL'] = (
                exp_wing_area / wing_area * clw
                + planform / wing_area * deg2rad(alpha - alpha0) * body_lift_curve_slope
            )

        alpha_stall = rad2deg(CL_max_flaps / lift_curve_slope) + alpha0
        outputs['alpha_stall'] = alpha_stall
        if any(x > 0.0 for x in alpha.real - alpha_stall.real):
            if verbosity >= Verbosity.BRIEF:
                warnings.warn(
                    f'Some angle of attack {alpha} might be greater than alpha stall {alpha_stall}.'
                )

        outputs['CL_max'] = CL_max_flaps

    def compute_partials(self, inputs, J):
        lift_curve_slope = inputs['lift_curve_slope']
        body_lift_curve_slope = inputs['body_lift_curve_slope']
        alpha0 = inputs[Aircraft.Wing.ZERO_LIFT_ANGLE]
        CL_max_flaps = inputs[Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP]
        wing_area = inputs[Aircraft.Wing.AREA]
        exp_wing_area = inputs[Aircraft.Wing.EXPOSED_AREA]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]

        if self.options['output_alpha']:
            CL = inputs['CL']
            mod_lift_curve_slope = (
                exp_wing_area / wing_area * lift_curve_slope
                + planform / wing_area * body_lift_curve_slope
            )

            # partials of mod_lift_curve_slope
            dmod_dexp_wing_area = 1.0 / wing_area * lift_curve_slope
            dmod_dwing_area = (
                -exp_wing_area / wing_area**2 * lift_curve_slope
                - planform / wing_area**2 * body_lift_curve_slope
            )
            dmod_dplanform = 1.0 / wing_area * body_lift_curve_slope
            dmod_dlift_curve_slope = exp_wing_area / wing_area
            dmod_dbody_lift_curve_slope = planform / wing_area

            J['mod_lift_curve_slope', Aircraft.Wing.EXPOSED_AREA] = dmod_dexp_wing_area
            J['mod_lift_curve_slope', Aircraft.Wing.AREA] = dmod_dwing_area
            J['mod_lift_curve_slope', Aircraft.Fuselage.PLANFORM_AREA] = dmod_dplanform
            J['mod_lift_curve_slope', 'lift_curve_slope'] = dmod_dlift_curve_slope
            J['mod_lift_curve_slope', 'body_lift_curve_slope'] = dmod_dbody_lift_curve_slope

            # alpha = rad2deg(CL / mod_lift_curve_slope) + alpha0

            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, Aircraft.Wing.AREA] = (
                -180.0 / np.pi * CL / mod_lift_curve_slope**2 * dmod_dwing_area
            )
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, Aircraft.Wing.EXPOSED_AREA] = (
                -180.0 / np.pi * CL / mod_lift_curve_slope**2 * dmod_dexp_wing_area
            )
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, Aircraft.Fuselage.PLANFORM_AREA] = (
                -180.0 / np.pi * CL / mod_lift_curve_slope**2 * dmod_dplanform
            )
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, 'lift_curve_slope'] = (
                -180.0 / np.pi * CL / mod_lift_curve_slope**2 * dmod_dlift_curve_slope
            )
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, 'body_lift_curve_slope'] = (
                -180.0 / np.pi * CL / mod_lift_curve_slope**2 * dmod_dbody_lift_curve_slope
            )

            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, 'CL'] = 180.0 / np.pi / mod_lift_curve_slope
            J[Dynamic.Vehicle.ANGLE_OF_ATTACK, Aircraft.Wing.ZERO_LIFT_ANGLE] = 1
        else:
            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
            # CL_1 = exp_wing_area / wing_area * lift_curve_slope * np.pi / 180.0 * (alpha - alpha0)
            # CL_2 = planform / wing_area * np.pi / 180.0 * (alpha - alpha0) * body_lift_curve_slope
            # CL = CL_1 + CL_2

            J['CL', Aircraft.Wing.EXPOSED_AREA] = (
                1.0 / wing_area * lift_curve_slope * np.pi / 180.0 * (alpha - alpha0)
            )
            J['CL', Aircraft.Fuselage.PLANFORM_AREA] = (
                1.0 / wing_area * deg2rad(alpha - alpha0) * body_lift_curve_slope
            )
            J['CL', Aircraft.Wing.AREA] = (
                -exp_wing_area / wing_area**2 * lift_curve_slope * np.pi / 180.0 * (alpha - alpha0)
                - planform / wing_area**2 * deg2rad(alpha - alpha0) * body_lift_curve_slope
            )
            J['CL', Aircraft.Wing.ZERO_LIFT_ANGLE] = (
                -exp_wing_area / wing_area * lift_curve_slope * np.pi / 180.0
                - planform / wing_area * body_lift_curve_slope * np.pi / 180.0
            )
            J['CL', Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
                exp_wing_area / wing_area * lift_curve_slope * np.pi / 180.0
                + planform / wing_area * body_lift_curve_slope * np.pi / 180.0
            )
            J['CL', 'lift_curve_slope'] = (
                exp_wing_area / wing_area * np.pi / 180.0 * (alpha - alpha0)
            )
            J['CL', 'body_lift_curve_slope'] = (
                planform / wing_area * np.pi / 180.0 * (alpha - alpha0)
            )

        J['alpha_stall', Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP] = (
            180.0 / np.pi / lift_curve_slope
        )
        J['alpha_stall', 'lift_curve_slope'] = -180.0 / np.pi * CL_max_flaps / lift_curve_slope**2
        J['alpha_stall', Aircraft.Wing.ZERO_LIFT_ANGLE] = 1
        J['CL_max', Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP] = 1


class CruiseAero(om.Group):
    """Top-level aerodynamics group for cruise (no flaps, no landing gear)."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        add_aviary_option(self, Aircraft.Design.TYPE)

        self.options.declare(
            'output_alpha',
            default=False,
            types=bool,
            desc='If True, output alpha for a given input CL',
        )
        self.options.declare(
            'input_atmos',
            default=False,
            types=bool,
            desc='Directly input speed of sound and kinematic viscosity instead of '
            'computing them with an atmospherics component. For testing.',
        )

    def setup(self):
        design_type = self.options[Aircraft.Design.TYPE]
        nn = self.options['num_nodes']
        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'aero_setup',
                BWBAeroSetup(
                    num_nodes=nn,
                    input_atmos=self.options['input_atmos'],
                ),
                promotes=['*'],
            )
        else:
            self.add_subsystem(
                'aero_setup',
                AeroSetup(
                    num_nodes=nn,
                    input_atmos=self.options['input_atmos'],
                ),
                promotes=['*'],
            )
        if self.options['output_alpha']:
            # lift_req -> CL
            self.add_subsystem('lift2cl', CLFromLift(num_nodes=nn), promotes=['*'])
        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'body_lift_curve',
                BWBBodyLiftCurveSlope(num_nodes=nn),
                promotes=['*'],
            )
            self.add_subsystem(
                'lift_coef',
                BWBLiftCoeffClean(output_alpha=self.options['output_alpha'], num_nodes=nn),
                promotes=['*'],
            )
        else:
            self.add_subsystem(
                'lift_coef',
                LiftCoeffClean(output_alpha=self.options['output_alpha'], num_nodes=nn),
                promotes=['*'],
            )
        self.add_subsystem('drag_coef', DragCoefClean(num_nodes=nn), promotes=['*'])
        self.add_subsystem('forces', AeroForces(num_nodes=nn), promotes=['*'])


class LowSpeedAero(om.Group):
    """Top-level aerodynamics group for near-ground flight."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        add_aviary_option(self, Aircraft.Design.TYPE)
        self.options.declare(
            'retract_gear',
            default=True,
            types=bool,
            desc='True to start with gear landing gear down, False for reverse',
        )
        self.options.declare(
            'retract_flaps',
            default=True,
            types=bool,
            desc='True to start with flaps applied, False for reverse',
        )
        self.options.declare(
            'lift_required',
            default=False,
            types=bool,
            desc='If True, compute CL from lift required (mass). If False, compute lift '
            'from current flight conditions including angle of attack.',
        )
        self.options.declare(
            'input_atmos',
            default=False,
            types=bool,
            desc='Directly input speed of sound and kinematic viscosity instead of '
            'computing them with an atmospherics component. For testing.',
        )

    def setup(self):
        nn = self.options['num_nodes']
        design_type = self.options[Aircraft.Design.TYPE]
        lift_required = self.options['lift_required']
        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'aero_setup',
                BWBAeroSetup(
                    num_nodes=nn,
                    input_atmos=self.options['input_atmos'],
                ),
                promotes=['*'],
            )
        else:
            self.add_subsystem(
                'aero_setup',
                AeroSetup(
                    num_nodes=nn,
                    input_atmos=self.options['input_atmos'],
                ),
                promotes=['*'],
            )

        aero_ramps = TanhRampComp(time_units='s', num_nodes=nn)
        aero_ramps.add_ramp(
            'flap_factor',
            output_units='unitless',
            initial_val=1.0 if self.options['retract_flaps'] else 0.0,
            final_val=0.0 if self.options['retract_flaps'] else 1.0,
        )
        aero_ramps.add_ramp(
            'gear_factor',
            output_units='unitless',
            initial_val=1.0 if self.options['retract_gear'] else 0.0,
            final_val=0.0 if self.options['retract_gear'] else 1.0,
        )

        self.add_subsystem(
            'aero_ramps',
            aero_ramps,
            promotes_inputs=[
                ('time', 't_curr'),
                ('flap_factor:t_init', 't_init_flaps'),
                ('flap_factor:t_duration', 'dt_flaps'),
                ('gear_factor:t_init', 't_init_gear'),
                ('gear_factor:t_duration', 'dt_gear'),
            ],
            promotes_outputs=['flap_factor', 'gear_factor'],
        )

        if lift_required:
            # lift_req -> CL
            self.add_subsystem(
                'lift2cl',
                CLFromLift(num_nodes=nn),
                promotes_inputs=['*'],
                # little bit of a hack here - input CL bypasses CL increment ramp
                # so ensure this is what's passed to DragCoef
                promotes_outputs=[('CL', 'CL_full_flaps')],
            )
            # NOTE Alpha is NOT an output from LowSpeedAero.
        else:
            self.add_subsystem(
                'kclge',
                GroundEffect(num_nodes=nn),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
            if design_type is AircraftTypes.BLENDED_WING_BODY:
                self.add_subsystem(
                    'body_lift_curve',
                    BWBBodyLiftCurveSlope(num_nodes=nn),
                    promotes=['*'],
                )
                self.add_subsystem(
                    'lift_coef',
                    BWBLiftCoeff(num_nodes=nn),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            else:
                self.add_subsystem(
                    'lift_coef',
                    LiftCoeff(num_nodes=nn),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )

        interp = om.MetaModelStructuredComp(method='slinear')
        interp.add_input('flap_defl', 10.0, units='deg', training_data=adelfd)
        interp.add_output('dCL_flaps_coef', 0.0, units='unitless', training_data=asigma)
        interp.add_output('CDI_factor', 1.0, units='unitless', training_data=adel6)
        self.add_subsystem('cdi_flap_interp', interp, promotes=['*'])

        self.add_subsystem(
            'drag_coef', DragCoef(num_nodes=nn), promotes=['*', ('CL', 'CL_full_flaps')]
        )

        self.add_subsystem(
            'total_cd',
            om.ExecComp(
                'CD = CD_base + flap_factor * dCD_flaps_full + gear_factor * dCD_gear_full',
                # "CD = CD_base + dCD_flaps + dCD_gear",
                CD=dict(shape=nn, units='unitless'),
                CD_base=dict(shape=nn, units='unitless'),
                # dCD_flaps=dict(shape=nn, units='unitless'),
                # dCD_gear=dict(shape=nn, units='unitless'),
                flap_factor=dict(shape=nn, units='unitless'),
                gear_factor=dict(shape=nn, units='unitless'),
                dCD_gear_full=dict(shape=nn, units='unitless'),
                dCD_flaps_full=dict(shape=nn, units='unitless'),
                has_diag_partials=True,
            ),
            promotes=['*'],
        )

        self.add_subsystem('forces', AeroForces(num_nodes=nn), promotes=['*'])

        self.set_input_defaults(Dynamic.Mission.ALTITUDE, np.zeros(nn))

        if self.options['retract_gear']:
            # takeoff defaults
            self.set_input_defaults('dt_gear', 7)
        # gear not dynamically extended during landing

        if self.options['retract_flaps']:
            # takeoff defaults
            self.set_input_defaults('flap_defl', 10)
            self.set_input_defaults('dt_flaps', 3)
        else:
            # landing defaults
            self.set_input_defaults('flap_defl', 40)
            # flaps not dynamically extended during landing
