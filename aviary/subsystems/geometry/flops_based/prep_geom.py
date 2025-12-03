"""
Define utilities to prepare derived values of aircraft geometry for
aerodynamics analysis.

TODO: blended-wing-body support
TODO: multiple engine model support
"""

import openmdao.api as om
from numpy import pi

from aviary.subsystems.geometry.flops_based.canard import Canard
from aviary.subsystems.geometry.flops_based.characteristic_lengths import (
    BWBWingCharacteristicLength,
    OtherCharacteristicLengths,
    WingCharacteristicLength,
)
from aviary.subsystems.geometry.flops_based.fuselage import (
    BWBDetailedCabinLayout,
    BWBFuselagePrelim,
    BWBSimpleCabinLayout,
    DetailedCabinLayout,
    FuselagePrelim,
    SimpleCabinLayout,
)
from aviary.subsystems.geometry.flops_based.nacelle import Nacelles
from aviary.subsystems.geometry.flops_based.utils import (
    Names,
    calc_fuselage_adjustment,
    calc_lifting_surface_scaler,
    d_calc_fuselage_adjustment,
    thickness_to_chord_scaler,
)
from aviary.subsystems.geometry.flops_based.wetted_area_total import TotalWettedArea
from aviary.subsystems.geometry.flops_based.wing import WingPrelim
from aviary.subsystems.geometry.flops_based.bwb_wing_detailed import (
    BWBUpdateDetailedWingDist,
    BWBComputeDetailedWingDist,
    BWBWingPrelim,
)
from aviary.variable_info.enums import AircraftTypes, Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings


class PrepGeom(om.Group):
    """Prepare derived values of aircraft geometry for aerodynamics analysis."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuselage.SIMPLE_LAYOUT)
        add_aviary_option(self, Aircraft.Design.TYPE)
        add_aviary_option(self, Aircraft.BWB.DETAILED_WING_PROVIDED)

    def setup(self):
        is_simple_layout = self.options[Aircraft.Fuselage.SIMPLE_LAYOUT]
        design_type = self.options[Aircraft.Design.TYPE]

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            if is_simple_layout:
                self.add_subsystem(
                    'fuselage_layout',
                    BWBSimpleCabinLayout(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            else:
                self.add_subsystem(
                    'fuselage_layout',
                    BWBDetailedCabinLayout(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            if self.options[Aircraft.BWB.DETAILED_WING_PROVIDED]:
                self.add_subsystem(
                    'detailed_wing',
                    BWBUpdateDetailedWingDist(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            else:
                self.add_subsystem(
                    'detailed_wing',
                    BWBComputeDetailedWingDist(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
        elif design_type is AircraftTypes.TRANSPORT:
            if is_simple_layout:
                self.add_subsystem(
                    'fuselage_layout',
                    SimpleCabinLayout(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            else:
                self.add_subsystem(
                    'fuselage_layout',
                    DetailedCabinLayout(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'fuselage_prelim',
                BWBFuselagePrelim(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'fuselage_prelim', FuselagePrelim(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'wing_prelim', BWBWingPrelim(), promotes_inputs=['*'], promotes_outputs=['*']
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'wing_prelim', WingPrelim(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        self.add_subsystem(
            'prelim',
            _Prelim(),
            promotes_inputs=['*'],
        )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem('wing', _BWBWing(), promotes_inputs=['*'], promotes_outputs=['*'])
        else:
            self.add_subsystem(
                'wing', _Wing(), promotes_inputs=['aircraft*'], promotes_outputs=['*']
            )

        if design_type is AircraftTypes.TRANSPORT:
            self.connect(f'prelim.{Names.CROOT}', f'wing.{Names.CROOT}')
            self.connect(f'prelim.{Names.CROOTB}', f'wing.{Names.CROOTB}')
            self.connect(f'prelim.{Names.XDX}', f'wing.{Names.XDX}')
            self.connect(f'prelim.{Names.XMULT}', f'wing.{Names.XMULT}')

        self.add_subsystem('tail', _Tail(), promotes_inputs=['aircraft*'], promotes_outputs=['*'])

        self.connect(f'prelim.{Names.XMULTH}', f'tail.{Names.XMULTH}')
        self.connect(f'prelim.{Names.XMULTV}', f'tail.{Names.XMULTV}')

        self.add_subsystem(
            'fus_ratios', _FuselageRatios(), promotes_inputs=['aircraft*'], promotes_outputs=['*']
        )
        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem('fuselage', _BWBFuselage(), promotes_outputs=['*'])
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'fuselage', _Fuselage(), promotes_inputs=['aircraft*'], promotes_outputs=['*']
            )

        if design_type is AircraftTypes.TRANSPORT:
            self.connect(f'prelim.{Names.CROOTB}', f'fuselage.{Names.CROOTB}')
            self.connect(f'prelim.{Names.CROTVT}', f'fuselage.{Names.CROTVT}')
            self.connect(f'prelim.{Names.CRTHTB}', f'fuselage.{Names.CRTHTB}')

        self.add_subsystem(
            'nacelles', Nacelles(), promotes_inputs=['aircraft*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'canard', Canard(), promotes_inputs=['aircraft*'], promotes_outputs=['*']
        )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'wing_characteristic_lengths',
                BWBWingCharacteristicLength(),
                promotes_inputs=['aircraft*'],
                promotes_outputs=['*'],
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'wing_characteristic_lengths',
                WingCharacteristicLength(),
                promotes_inputs=['aircraft*'],
                promotes_outputs=['*'],
            )
        self.add_subsystem(
            'other_characteristic_lengths',
            OtherCharacteristicLengths(),
            promotes_inputs=['aircraft*'],
            promotes_outputs=['*'],
        )

        self.connect(f'prelim.{Names.CROOT}', f'other_characteristic_lengths.{Names.CROOT}')

        self.add_subsystem(
            'total_wetted_area', TotalWettedArea(), promotes_inputs=['*'], promotes_outputs=['*']
        )


class _Prelim(om.ExplicitComponent):
    """Calculate internal derived values of aircraft geometry for FLOPS-based aerodynamics analysis."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')

        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_input(self, Aircraft.VerticalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.VerticalTail.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.GLOVE_AND_BAT, units='ft**2')
        # NOTE: FLOPS/aviary1 calculate span locally
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')

        self.add_output(Names.CROOT, 1.0, units='unitless')
        self.add_output(Names.CROOTB, 1.0, units='unitless')
        self.add_output(Names.CROTM, 1.0, units='unitless')
        self.add_output(Names.CROTVT, 1.0, units='unitless')
        self.add_output(Names.CRTHTB, 1.0, units='unitless')
        self.add_output(Names.SPANHT, 1.0, units='unitless')
        self.add_output(Names.SPANVT, 1.0, units='unitless')
        self.add_output(Names.XDX, 1.0, units='unitless')
        self.add_output(Names.XMULT, 1.0, units='unitless')
        self.add_output(Names.XMULTH, 1.0, units='unitless')
        self.add_output(Names.XMULTV, 1.0, units='unitless')

    def setup_partials(self):
        fuselage_var = self.fuselage_var

        self.declare_partials(Names.XDX, fuselage_var, val=1.0)

        self.declare_partials(
            Names.XMULT, Aircraft.Wing.THICKNESS_TO_CHORD, val=thickness_to_chord_scaler
        )

        self.declare_partials(
            Names.XMULTH, Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=thickness_to_chord_scaler
        )

        self.declare_partials(
            Names.XMULTV, Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=thickness_to_chord_scaler
        )

        self.declare_partials(
            Names.SPANHT,
            [
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.ASPECT_RATIO,
            ],
        )

        self.declare_partials(
            Names.CRTHTB,
            [
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.ASPECT_RATIO,
                Aircraft.HorizontalTail.TAPER_RATIO,
                fuselage_var,
            ],
        )

        self.declare_partials(
            Names.CROOT,
            [
                Aircraft.Wing.AREA,
                Aircraft.Wing.GLOVE_AND_BAT,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
            ],
        )

        self.declare_partials(
            Names.CROTM,
            [
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                fuselage_var,
            ],
        )

        self.declare_partials(
            Names.CROOTB,
            [
                Aircraft.Wing.AREA,
                Aircraft.Wing.GLOVE_AND_BAT,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                fuselage_var,
            ],
        )

        self.declare_partials(
            Names.SPANVT,
            [
                Aircraft.VerticalTail.AREA,
                Aircraft.VerticalTail.ASPECT_RATIO,
            ],
        )

        self.declare_partials(
            Names.CROTVT,
            [
                Aircraft.VerticalTail.AREA,
                Aircraft.VerticalTail.ASPECT_RATIO,
                Aircraft.VerticalTail.TAPER_RATIO,
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        w_tc = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        h_tc = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]
        v_tc = inputs[Aircraft.VerticalTail.THICKNESS_TO_CHORD]

        outputs[Names.XMULT] = calc_lifting_surface_scaler(w_tc)
        outputs[Names.XMULTH] = calc_lifting_surface_scaler(h_tc)
        outputs[Names.XMULTV] = calc_lifting_surface_scaler(v_tc)

        fuselage_var = self.fuselage_var

        XDX = outputs[Names.XDX] = inputs[fuselage_var]
        aspect_ratio = inputs[Aircraft.HorizontalTail.ASPECT_RATIO]
        area = inputs[Aircraft.HorizontalTail.AREA]

        span = outputs[Names.SPANHT] = (aspect_ratio * area) ** 0.5

        CRTHTB = 0.0

        if 0.0 < span:
            taper_ratio = inputs[Aircraft.HorizontalTail.TAPER_RATIO]

            CRTHTB = (
                2.0 * area / (span * (1.0 + taper_ratio))
                + ((span / 2.0 - XDX / 4.0) / (span / 2.0)) * (1.0 - taper_ratio)
                + taper_ratio
            )

        outputs[Names.CRTHTB] = CRTHTB

        area = inputs[Aircraft.Wing.AREA]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]
        span = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]

        CROOT = outputs[Names.CROOT] = ((area - glove_and_bat) * 2.0) / ((1.0 + taper_ratio) * span)

        CROTM = outputs[Names.CROTM] = ((span / 2.0 - XDX / 2.0) / (span / 2.0)) * (
            1.0 - taper_ratio
        ) + taper_ratio

        outputs[Names.CROOTB] = CROOT * CROTM

        area = inputs[Aircraft.VerticalTail.AREA]
        aspect_ratio = inputs[Aircraft.VerticalTail.ASPECT_RATIO]

        span = outputs[Names.SPANVT] = (area * aspect_ratio) ** 0.5

        CROTVT = 0.0

        if 0.0 < span:
            taper_ratio = inputs[Aircraft.VerticalTail.TAPER_RATIO]

            CROTVT = 2.0 * area / (span * (1.0 + taper_ratio))

        outputs[Names.CROTVT] = CROTVT

    def compute_partials(self, inputs, J, discrete_inputs=None):
        fuselage_var = self.fuselage_var

        XDX = inputs[fuselage_var]
        area = inputs[Aircraft.HorizontalTail.AREA]
        aspect_ratio = inputs[Aircraft.HorizontalTail.ASPECT_RATIO]

        span2 = area * aspect_ratio
        span = span2**0.5
        f = 0.5 / span

        J[Names.SPANHT, Aircraft.HorizontalTail.AREA] = f * aspect_ratio
        J[Names.SPANHT, Aircraft.HorizontalTail.ASPECT_RATIO] = f * area

        da = dr = dt = dx = 0.0

        if 0.0 < span:
            # b = (a * ar)**0.5
            #
            #        2 * a       b / 2 - x / 4
            # c = ____________ + _____________ * (1 - tr) + tr
            #     b * (1 + tr)       b / 2
            #
            #              2 * a               x * (1 - tr)
            #   = ________________________ - _________________ + 1
            #     (a * ar)**0.5 * (1 + tr)   2 * (a * ar)**0.5
            taper_ratio = inputs[Aircraft.HorizontalTail.TAPER_RATIO]

            _1p_tr = 1.0 + taper_ratio
            _1m_tr = 1.0 - taper_ratio

            # da = d(f0 / g0) + d(f1 / g1) + 0
            #      df0 * g0 - f0 * dg0   df1 * g1 - f1 * dg1
            #    = ___________________ + ___________________
            #             g0**2                 g1**2
            dspan_darea = 0.5 * (aspect_ratio / area) ** 0.5

            da = (
                2.0 / _1p_tr * (1.0 - area * dspan_darea / span) / span
                + _1m_tr * 0.5 * XDX * dspan_darea / span**2
            )

            # dr = d(k0 * a / (a * ar)**0.5) - d(k1 / (a * ar)**0.5) + 0
            #    = d((k0 * a - k1) / (a * ar)**0.5)
            #    = -0.5 * (k0 * a - k1) / (a * ar)**1.5 * a
            k0 = 2.0 / _1p_tr
            k1 = XDX * _1m_tr / 2.0
            dr = -0.5 * area * (k0 * area - k1) / span**3.0

            # dt = d(k0 / (1 + tr)) - d(k1 * (1 - tr)) + 0
            #    = -k0 / (1 + tr)**2 + k1
            k0 = 2.0 * area / span
            k1 = XDX / (2.0 * span)
            dt = k1 - k0 / _1p_tr**2.0

            # dx = 0 - d(x * k) + 0
            #    = -k
            dx = -_1m_tr / (2.0 * span)

        J[Names.CRTHTB, Aircraft.HorizontalTail.AREA] = da
        J[Names.CRTHTB, Aircraft.HorizontalTail.ASPECT_RATIO] = dr
        J[Names.CRTHTB, Aircraft.HorizontalTail.TAPER_RATIO] = dt
        J[Names.CRTHTB, fuselage_var] = dx

        area = inputs[Aircraft.Wing.AREA]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]
        span = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]

        a_g = area - glove_and_bat
        _1p_tr = 1.0 + taper_ratio
        span_1p_tr = span * _1p_tr

        #     (a - g) * 2
        # c = ____________
        #     (1 + tr) * b
        CROOT = 2.0 * a_g / span_1p_tr

        dc = J[Names.CROOT, Aircraft.Wing.AREA] = 2.0 / span_1p_tr
        J[Names.CROOT, Aircraft.Wing.GLOVE_AND_BAT] = -dc
        J[Names.CROOT, Aircraft.Wing.SPAN] = -CROOT / span
        J[Names.CROOT, Aircraft.Wing.TAPER_RATIO] = -CROOT / _1p_tr

        #
        #     (b / 2) - (x / 2)
        # c = _________________ * (1 - tr) + tr
        #          (b / 2)

        # db = d(f / g) + 0
        #    = ((df * g) - (f * dg)) / g**2
        _1m_tr = 1.0 - taper_ratio
        g = span / 2.0
        f = (g - (XDX / 2.0)) * _1m_tr
        df = _1m_tr / 2.0
        dg = 0.5
        J[Names.CROTM, Aircraft.Wing.SPAN] = (df * g - f * dg) / g**2.0

        # dt = d(k * (1 - tr) + tr)
        k = (g - (XDX / 2.0)) / g
        J[Names.CROTM, Aircraft.Wing.TAPER_RATIO] = -k + 1
        # dx = 0 - d(k * x) + 0 = -k
        J[Names.CROTM, fuselage_var] = -_1m_tr / span

        # dc = d(f * g) = df * g + f * dg
        f = CROOT

        g = (1.0 - taper_ratio) * ((span / 2.0) - (XDX / 2.0)) / (span / 2.0) + taper_ratio

        df = J[Names.CROOT, Aircraft.Wing.AREA]
        dg = 0.0
        J[Names.CROOTB, Aircraft.Wing.AREA] = df * g

        df = J[Names.CROOT, Aircraft.Wing.GLOVE_AND_BAT]
        dg = 0.0
        J[Names.CROOTB, Aircraft.Wing.GLOVE_AND_BAT] = df * g

        df = J[Names.CROOT, Aircraft.Wing.SPAN]
        dg = J[Names.CROTM, Aircraft.Wing.SPAN]
        J[Names.CROOTB, Aircraft.Wing.SPAN] = df * g + f * dg

        df = J[Names.CROOT, Aircraft.Wing.TAPER_RATIO]
        dg = J[Names.CROTM, Aircraft.Wing.TAPER_RATIO]
        J[Names.CROOTB, Aircraft.Wing.TAPER_RATIO] = df * g + f * dg

        df = 0.0
        dg = J[Names.CROTM, fuselage_var]
        J[Names.CROOTB, fuselage_var] = f * dg

        area = inputs[Aircraft.VerticalTail.AREA]
        aspect_ratio = inputs[Aircraft.VerticalTail.ASPECT_RATIO]

        span = (area * aspect_ratio) ** 0.5

        J[Names.SPANVT, Aircraft.VerticalTail.AREA] = 0.5 * aspect_ratio / span

        J[Names.SPANVT, Aircraft.VerticalTail.ASPECT_RATIO] = 0.5 * area / span

        da = dr = dt = 0.0

        if 0.0 < span:
            taper_ratio = inputs[Aircraft.VerticalTail.TAPER_RATIO]

            _1p_tr = 1.0 + taper_ratio

            f = 2.0 * area / _1p_tr
            g = span
            df = 2.0 / _1p_tr
            dg = J[Names.SPANVT, Aircraft.VerticalTail.AREA]
            da = (df * g - f * dg) / g**2

            # dr = d(k / (a * ar)**0.5)
            #    = -0.5 * k / (a * ar)**1.5 * a
            dr = -(area**2.0) / (_1p_tr * span**3.0)

            # dt = d(k / (1 + tr)) = -k / (1 + tr)**2
            dt = -2.0 * area / (span * _1p_tr**2.0)

        J[Names.CROTVT, Aircraft.VerticalTail.AREA] = da
        J[Names.CROTVT, Aircraft.VerticalTail.ASPECT_RATIO] = dr
        J[Names.CROTVT, Aircraft.VerticalTail.TAPER_RATIO] = dt

    @property
    def fuselage_var(self):
        """Define the variable name associated with XDX."""
        value = Aircraft.Fuselage.AVG_DIAMETER

        if self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]:
            value = Aircraft.Fuselage.MAX_WIDTH

        return value


class _Wing(om.ExplicitComponent):
    """Calculate wing wetted area of aircraft geometry for FLOPS-based aerodynamics analysis."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuselage.NUM_FUSELAGES)

    def setup(self):
        self.add_input(Names.CROOT, 0.0, units='unitless')
        self.add_input(Names.CROOTB, 0.0, units='unitless')
        self.add_input(Names.XDX, 0.0, units='unitless')
        self.add_input(Names.XMULT, 0.0, units='unitless')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.WETTED_AREA_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Wing.WETTED_AREA, units='ft**2')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Wing.WETTED_AREA,
            [
                Names.CROOT,
                Names.CROOTB,
                Names.XDX,
                Names.XMULT,
                Aircraft.Wing.AREA,
                Aircraft.Wing.WETTED_AREA_SCALER,
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_fuselage = self.options[Aircraft.Fuselage.NUM_FUSELAGES]

        area = inputs[Aircraft.Wing.AREA]
        CROOT = inputs[Names.CROOT]
        CROOTB = inputs[Names.CROOTB]
        scaler = inputs[Aircraft.Wing.WETTED_AREA_SCALER]
        XDX = inputs[Names.XDX]
        XMULT = inputs[Names.XMULT]

        wetted_area = scaler * XMULT * (area - (num_fuselage * XDX / 2.0) * (CROOT + CROOTB))

        outputs[Aircraft.Wing.WETTED_AREA] = wetted_area

    def compute_partials(self, inputs, J, discrete_inputs=None):
        num_fuselage = self.options[Aircraft.Fuselage.NUM_FUSELAGES]

        area = inputs[Aircraft.Wing.AREA]
        CROOT = inputs[Names.CROOT]
        CROOTB = inputs[Names.CROOTB]
        scaler = inputs[Aircraft.Wing.WETTED_AREA_SCALER]
        XDX = inputs[Names.XDX]
        XMULT = inputs[Names.XMULT]

        J[Aircraft.Wing.WETTED_AREA, Aircraft.Wing.AREA] = scaler * XMULT

        J[Aircraft.Wing.WETTED_AREA, Aircraft.Wing.WETTED_AREA_SCALER] = XMULT * (
            area - (num_fuselage * XDX / 2.0) * (CROOT + CROOTB)
        )

        J[Aircraft.Wing.WETTED_AREA, Names.CROOT] = -0.5 * scaler * XMULT * (num_fuselage * XDX)

        J[Aircraft.Wing.WETTED_AREA, Names.CROOTB] = J[Aircraft.Wing.WETTED_AREA, Names.CROOT]

        J[Aircraft.Wing.WETTED_AREA, Names.XDX] = (
            -0.5 * scaler * XMULT * num_fuselage * (CROOT + CROOTB)
        )

        J[Aircraft.Wing.WETTED_AREA, Names.XMULT] = scaler * (
            area - 0.5 * (num_fuselage * XDX) * (CROOT + CROOTB)
        )


class _BWBWing(om.ExplicitComponent):
    """Calculate wing wetted area of BWB aircraft geometry for FLOPS-based aerodynamics analysis."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.NUM_INTEGRATION_STATIONS)

    def setup(self):
        num_stations = self.options[Aircraft.Wing.NUM_INTEGRATION_STATIONS]

        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.GLOVE_AND_BAT, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        self.add_input('BWB_INPUT_STATION_DIST', shape=num_stations, units='unitless')
        self.add_input('BWB_CHORD_PER_SEMISPAN_DIST', shape=num_stations, units='unitless')
        self.add_input('BWB_THICKNESS_TO_CHORD_DIST', shape=num_stations, units='unitless')

        add_aviary_output(self, Aircraft.Wing.WETTED_AREA, units='ft**2')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        input_station_dist = inputs['BWB_INPUT_STATION_DIST']
        num_stations = len(inputs['BWB_INPUT_STATION_DIST'])

        span = inputs[Aircraft.Wing.SPAN]

        ssmw = 0.0
        bwb_chord_per_semispan_dist = inputs['BWB_CHORD_PER_SEMISPAN_DIST']
        bwb_thickness_to_chord_dist = inputs['BWB_THICKNESS_TO_CHORD_DIST']

        if bwb_chord_per_semispan_dist[0] <= 5.0:
            C1 = bwb_chord_per_semispan_dist[0] * span / 2.0
        else:
            C1 = bwb_chord_per_semispan_dist[0]
        if input_station_dist[0] <= 1.1:
            Y1 = input_station_dist[0] * span / 2.0
        else:
            Y1 = input_station_dist[0]
        for n in range(1, num_stations):
            avg_toc = (bwb_thickness_to_chord_dist[n - 1] + bwb_thickness_to_chord_dist[n]) / 2.0
            ckt = 2.0 + 0.387 * avg_toc
            if bwb_chord_per_semispan_dist[n] <= 5.0:
                C2 = bwb_chord_per_semispan_dist[n] * span / 2.0
            else:
                C2 = bwb_chord_per_semispan_dist[n]
            if input_station_dist[n] <= 1.1:
                Y2 = input_station_dist[n] * span / 2.0
            else:
                Y2 = input_station_dist[n]
            axp = (Y2 - Y1) * (C1 + C2)
            C1 = C2
            Y1 = Y2
            ssmw = ssmw + axp * ckt

        outputs[Aircraft.Wing.WETTED_AREA] = ssmw


class _Tail(om.ExplicitComponent):
    """
    Calculate horizontal wing and vertical wing wetted areas of aircraft geometry
    for FLOPS-based aerodynamics analysis.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES)
        add_aviary_option(self, Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION)

    def setup(self):
        self.add_input(Names.XMULTH, 0.0, units='unitless')
        self.add_input(Names.XMULTV, 0.0, units='unitless')

        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')

        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, units='unitless')

        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA_SCALER, units='unitless')

        add_aviary_input(self, Aircraft.VerticalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.HorizontalTail.WETTED_AREA, units='ft**2')
        add_aviary_output(self, Aircraft.VerticalTail.WETTED_AREA, units='ft**2')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.HorizontalTail.WETTED_AREA,
            [
                Names.XMULTH,
                Aircraft.HorizontalTail.AREA,
                Aircraft.HorizontalTail.WETTED_AREA_SCALER,
            ],
        )

        self.declare_partials(
            Aircraft.VerticalTail.WETTED_AREA,
            [Names.XMULTV, Aircraft.VerticalTail.AREA, Aircraft.VerticalTail.WETTED_AREA_SCALER],
        )

        redux = self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]

        if not redux:
            self.declare_partials(
                Aircraft.HorizontalTail.WETTED_AREA,
                [Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION],
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # horizontal tail
        XMULTH = inputs[Names.XMULTH]
        area = inputs[Aircraft.HorizontalTail.AREA]
        scaler = inputs[Aircraft.HorizontalTail.WETTED_AREA_SCALER]

        wetted_area = scaler * XMULTH * area

        redux = self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]

        if not redux:
            num_fuselage_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES]

            vertical_tail_fraction = inputs[Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION]

            wetted_area *= 1.0 - (0.185 + num_fuselage_engines * 0.063) * (
                1.0 - vertical_tail_fraction
            )

        outputs[Aircraft.HorizontalTail.WETTED_AREA] = wetted_area

        # vertical tail
        XMULTV = inputs[Names.XMULTV]
        area = inputs[Aircraft.VerticalTail.AREA]
        scaler = inputs[Aircraft.VerticalTail.WETTED_AREA_SCALER]

        wetted_area = scaler * XMULTV * area

        outputs[Aircraft.VerticalTail.WETTED_AREA] = wetted_area

    def compute_partials(self, inputs, J, discrete_inputs=None):
        redux = self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]

        # horizontal tail
        XMULTH = inputs[Names.XMULTH]
        area = inputs[Aircraft.HorizontalTail.AREA]
        scaler = inputs[Aircraft.HorizontalTail.WETTED_AREA_SCALER]

        if not redux:
            num_fuselage_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES]

            vertical_tail_fraction = inputs[Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION]

            fengines = 0.185 + num_fuselage_engines * 0.063
            fact = 1.0 - fengines * (1.0 - vertical_tail_fraction)

            J[Aircraft.HorizontalTail.WETTED_AREA, Names.XMULTH] = scaler * area * fact

            J[Aircraft.HorizontalTail.WETTED_AREA, Aircraft.HorizontalTail.AREA] = (
                scaler * XMULTH * fact
            )

            deriv = scaler * XMULTH * area * fengines

            J[
                Aircraft.HorizontalTail.WETTED_AREA, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION
            ] = deriv

            J[
                Aircraft.HorizontalTail.WETTED_AREA,
                Aircraft.HorizontalTail.WETTED_AREA_SCALER,
            ] = XMULTH * area * fact

        else:
            J[Aircraft.HorizontalTail.WETTED_AREA, Names.XMULTH] = scaler * area

            J[Aircraft.HorizontalTail.WETTED_AREA, Aircraft.HorizontalTail.AREA] = scaler * XMULTH

            J[
                Aircraft.HorizontalTail.WETTED_AREA,
                Aircraft.HorizontalTail.WETTED_AREA_SCALER,
            ] = XMULTH * area

        # vertical tail
        XMULTV = inputs[Names.XMULTV]
        area = inputs[Aircraft.VerticalTail.AREA]
        scaler = inputs[Aircraft.VerticalTail.WETTED_AREA_SCALER]

        J[Aircraft.VerticalTail.WETTED_AREA, Names.XMULTV] = scaler * area

        J[Aircraft.VerticalTail.WETTED_AREA, Aircraft.VerticalTail.AREA] = scaler * XMULTV

        J[Aircraft.VerticalTail.WETTED_AREA, Aircraft.VerticalTail.WETTED_AREA_SCALER] = (
            XMULTV * area
        )


class _BWBFuselage(om.ExplicitComponent):
    """
    Set BWB fuselage cross sectional area, and fuselage wetted area to zero
    for FLOPS-based aerodynamics analysis when BWB has detailed wings.
    """

    def setup(self):
        add_aviary_output(self, Aircraft.Fuselage.CROSS_SECTION, units='ft**2')
        add_aviary_output(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')

    def compute(self, inputs, outputs):
        outputs[Aircraft.Fuselage.CROSS_SECTION] = 0.0
        outputs[Aircraft.Fuselage.WETTED_AREA] = 0.0


class _Fuselage(om.ExplicitComponent):
    """
    Calculate fuselage cross sectional area, and fuselage wetted area of aircraft geometry
    for FLOPS-based aerodynamics analysis.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuselage.NUM_FUSELAGES)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        self.add_input(Names.CROOTB, 0.0, units='unitless')
        self.add_input(Names.CROTVT, 0.0, units='unitless')
        self.add_input(Names.CRTHTB, 0.0, units='unitless')

        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA_SCALER, units='unitless')

        add_aviary_input(self, Aircraft.HorizontalTail.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_output(self, Aircraft.Fuselage.CROSS_SECTION, units='ft**2')
        add_aviary_output(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')

    def setup_partials(self):
        self.declare_partials(Aircraft.Fuselage.CROSS_SECTION, Aircraft.Fuselage.AVG_DIAMETER)

        self.declare_partials(
            Aircraft.Fuselage.WETTED_AREA,
            [
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.WETTED_AREA_SCALER,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.VerticalTail.THICKNESS_TO_CHORD,
                Aircraft.Wing.THICKNESS_TO_CHORD,
                Names.CROOTB,
                Names.CROTVT,
                Names.CRTHTB,
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        verbosity = self.options[Settings.VERBOSITY]
        num_fuselages = self.options[Aircraft.Fuselage.NUM_FUSELAGES]
        if num_fuselages < 1:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Fuselage.NUM_FUSELAGES must be positive.')

        avg_diam = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        if avg_diam <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Fuselage.AVG_DIAMETER must be positive.')

        cross_section = pi * (avg_diam / 2.0) ** 2.0
        outputs[Aircraft.Fuselage.CROSS_SECTION] = cross_section

        if (0 < num_fuselages) and (0.0 < avg_diam):
            CROOTB = inputs[Names.CROOTB]
            thickness_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]

            CRTHTB = inputs[Names.CRTHTB]

            scaler = inputs[Aircraft.Fuselage.WETTED_AREA_SCALER]
            ht_thickness_chord = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]

            vertical_tail_fraction = inputs[Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION]

            CROTVT = inputs[Names.CROTVT]

            length = inputs[Aircraft.Fuselage.LENGTH]
            vt_thickness_chord = inputs[Aircraft.VerticalTail.THICKNESS_TO_CHORD]

            cfa = calc_fuselage_adjustment(CROOTB, thickness_chord)
            cfah = calc_fuselage_adjustment(CRTHTB, ht_thickness_chord)
            cfav = calc_fuselage_adjustment(CROTVT, vt_thickness_chord)

            wetted_area = scaler * (
                pi * avg_diam**2.0 * (length / avg_diam - 1.7)
                - 2.0 * cfa
                - 2.0 * cfah * (1.0 - vertical_tail_fraction)
                - cfav
            )
        else:
            wetted_area = 0.0

        outputs[Aircraft.Fuselage.WETTED_AREA] = wetted_area

    def compute_partials(self, inputs, J, discrete_inputs=None):
        num_fuselages = self.options[Aircraft.Fuselage.NUM_FUSELAGES]

        avg_diam = inputs[Aircraft.Fuselage.AVG_DIAMETER]

        J[Aircraft.Fuselage.CROSS_SECTION, Aircraft.Fuselage.AVG_DIAMETER] = 0.5 * pi * avg_diam

        if (0 < num_fuselages) and (0.0 < avg_diam):
            CROOTB = inputs[Names.CROOTB]
            CRTHTB = inputs[Names.CRTHTB]
            CROTVT = inputs[Names.CROTVT]

            scaler = inputs[Aircraft.Fuselage.WETTED_AREA_SCALER]

            thickness_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]

            ht_thickness_chord = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]

            vt_thickness_chord = inputs[Aircraft.VerticalTail.THICKNESS_TO_CHORD]

            length = inputs[Aircraft.Fuselage.LENGTH]
            vertical_tail_fraction = inputs[Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION]

            cfa = calc_fuselage_adjustment(CROOTB, thickness_chord)
            cfah = calc_fuselage_adjustment(CRTHTB, ht_thickness_chord)
            cfav = calc_fuselage_adjustment(CROTVT, vt_thickness_chord)

            dcfa = d_calc_fuselage_adjustment(CROOTB, thickness_chord)
            dcfah = d_calc_fuselage_adjustment(CRTHTB, ht_thickness_chord)
            dcfav = d_calc_fuselage_adjustment(CROTVT, vt_thickness_chord)

            J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.AVG_DIAMETER] = (
                scaler * pi * (length - 3.4 * avg_diam)
            )

            J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.LENGTH] = scaler * pi * avg_diam

            J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.WETTED_AREA_SCALER] = (
                pi * avg_diam**2.0 * (length / avg_diam - 1.7)
                - 2.0 * cfa
                - 2.0 * cfah * (1.0 - vertical_tail_fraction)
                - cfav
            )

            J[Aircraft.Fuselage.WETTED_AREA, Names.CROOTB] = scaler * -2.0 * dcfa[0]

            J[Aircraft.Fuselage.WETTED_AREA, Names.CRTHTB] = (
                scaler * -2.0 * dcfah[0] * (1.0 - vertical_tail_fraction)
            )

            J[Aircraft.Fuselage.WETTED_AREA, Names.CROTVT] = scaler * -dcfav[0]

            J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Wing.THICKNESS_TO_CHORD] = (
                scaler * -2.0 * dcfa[1]
            )

            J[Aircraft.Fuselage.WETTED_AREA, Aircraft.HorizontalTail.THICKNESS_TO_CHORD] = (
                scaler * -2.0 * dcfah[1] * (1.0 - vertical_tail_fraction)
            )

            J[Aircraft.Fuselage.WETTED_AREA, Aircraft.VerticalTail.THICKNESS_TO_CHORD] = (
                scaler * -dcfav[1]
            )

            J[Aircraft.Fuselage.WETTED_AREA, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION] = (
                scaler * 2.0 * cfah
            )
        else:
            J[Aircraft.Fuselage.WETTED_AREA, Names.CROOTB] = J[
                Aircraft.Fuselage.WETTED_AREA, Names.CRTHTB
            ] = J[Aircraft.Fuselage.WETTED_AREA, Names.CROTVT] = J[
                Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.AVG_DIAMETER
            ] = J[Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.LENGTH] = J[
                Aircraft.Fuselage.WETTED_AREA, Aircraft.Fuselage.WETTED_AREA_SCALER
            ] = J[Aircraft.Fuselage.WETTED_AREA, Aircraft.HorizontalTail.THICKNESS_TO_CHORD] = J[
                Aircraft.Fuselage.WETTED_AREA, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION
            ] = J[Aircraft.Fuselage.WETTED_AREA, Aircraft.VerticalTail.THICKNESS_TO_CHORD] = J[
                Aircraft.Fuselage.WETTED_AREA, Aircraft.Wing.THICKNESS_TO_CHORD
            ] = 0.0


class _FuselageRatios(om.ExplicitComponent):
    """
    Calculate fuselage diameter to wing span ratio and fuselage length to diameter ratio
    of aircraft geometry for FLOPS-based aerodynamics analysis.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.GLOVE_AND_BAT, units='ft**2')

        add_aviary_output(self, Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, units='unitless')
        add_aviary_output(self, Aircraft.Fuselage.LENGTH_TO_DIAMETER, units='unitless')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
            [
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.GLOVE_AND_BAT,
            ],
        )

        self.declare_partials(
            Aircraft.Fuselage.LENGTH_TO_DIAMETER,
            [
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.AVG_DIAMETER,
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area = inputs[Aircraft.Wing.AREA]
        aspect_ratio = inputs[Aircraft.Wing.ASPECT_RATIO]
        avg_diam = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]

        diam_to_wing_span = avg_diam / (aspect_ratio * (area - glove_and_bat)) ** 0.5

        outputs[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN] = diam_to_wing_span

        if 0.0 < avg_diam:
            length = inputs[Aircraft.Fuselage.LENGTH]

            length_to_diam = length / avg_diam
        else:
            length_to_diam = 100.0  # FLOPS default value

        outputs[Aircraft.Fuselage.LENGTH_TO_DIAMETER] = length_to_diam

    def compute_partials(self, inputs, J, discrete_inputs=None):
        area = inputs[Aircraft.Wing.AREA]
        aspect_ratio = inputs[Aircraft.Wing.ASPECT_RATIO]
        avg_diam = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]

        fact = aspect_ratio * (area - glove_and_bat)
        fact2 = 1.0 / fact**1.5

        J[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Fuselage.AVG_DIAMETER] = 1.0 / fact**0.5

        J[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Wing.ASPECT_RATIO] = (
            -0.5 * avg_diam * (area - glove_and_bat) * fact2
        )

        J[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Wing.AREA] = (
            -0.5 * avg_diam * aspect_ratio * fact2
        )

        J[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Wing.GLOVE_AND_BAT] = (
            0.5 * avg_diam * aspect_ratio * fact2
        )

        if 0.0 < avg_diam:
            length = inputs[Aircraft.Fuselage.LENGTH]

            J[Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Fuselage.AVG_DIAMETER] = (
                -length / avg_diam**2
            )

            J[Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Fuselage.LENGTH] = 1.0 / avg_diam

        else:
            J[Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Fuselage.AVG_DIAMETER] = J[
                Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Fuselage.LENGTH
            ] = 0.0
