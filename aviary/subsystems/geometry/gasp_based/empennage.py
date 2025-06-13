import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.variable_info.variables import Aircraft


class TailVolCoef(om.ExplicitComponent):
    """GASP tail volume coefficient fallback calculation.

    This component can be used to compute a volume coefficient for either a horizontal
    or vertical tail. The volume coefficient is based on an empirical relationship
    using gross aircraft parameters such as fuselage length, wing area, etc. For a
    horizontal tail, the wing chord is input. For a vertical tail, the wing span is
    input.
    """

    def initialize(self):
        self.options.declare(
            'orientation',
            values=['horizontal', 'vertical'],
            desc='Tail orientation, can be horizontal or vertical.',
        )

    def setup(self):
        veritcal = self.options['orientation'] == 'vertical'
        if veritcal:
            self.io_names = {
                'vol_coef': Aircraft.VerticalTail.VOLUME_COEFFICIENT,
                'wing_ref': Aircraft.Wing.SPAN,
            }
        else:
            self.io_names = {
                'vol_coef': Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
                'wing_ref': Aircraft.Wing.AVERAGE_CHORD,
            }

        # coefficients used in the empirical equation
        if veritcal:
            self.k = [0.07, 0.0434, 0.336]
        else:
            self.k = [0.43, 0.38, 0.85]

        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, units='unitless')

        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')

        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')

        add_aviary_input(self, self.io_names['wing_ref'], val=12.612, units='ft')

        add_aviary_output(self, self.io_names['vol_coef'], units='unitless')

    def setup_partials(self):
        self.declare_partials(self.io_names['vol_coef'], '*')

    def compute(self, inputs, outputs):
        htail_loc, fus_len, cab_w, wing_area, wing_ref = inputs.values()
        k1, k2, k3 = self.k
        ch1 = k1 - k2 * htail_loc
        outputs[self.io_names['vol_coef']] = k3 * fus_len * cab_w**2 / (wing_area * wing_ref) + ch1

    def compute_partials(self, inputs, J):
        str_vol_coef = self.io_names['vol_coef']
        str_wing_ref = self.io_names['wing_ref']

        htail_loc, fus_len, cab_w, wing_area, wing_ref = inputs.values()
        k1, k2, k3 = self.k
        J[str_vol_coef, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION] = -k2
        J[str_vol_coef, Aircraft.Fuselage.LENGTH] = k3 * cab_w**2 / (wing_area * wing_ref)
        J[str_vol_coef, Aircraft.Fuselage.AVG_DIAMETER] = (
            2 * k3 * fus_len * cab_w / (wing_area * wing_ref)
        )
        J[str_vol_coef, Aircraft.Wing.AREA] = -k3 * fus_len * cab_w**2 / (wing_area**2 * wing_ref)
        J[str_vol_coef, str_wing_ref] = -k3 * fus_len * cab_w**2 / (wing_area * wing_ref**2)


class TailSize(om.ExplicitComponent):
    """GASP tail geometry calculations.

    This component can be used for either a horizontal tail or vertical tail. For a
    horizontal tail, the ratio of wing chord to tail moment arm and the wing chord are
    input for tail moment arm calculation. For a vertical tail, the ratio of wing span
    to tail moment arm and the wing span are input.
    """

    def initialize(self):
        self.options.declare(
            'orientation',
            values=['horizontal', 'vertical'],
            desc='Tail orientation, can be horizontal or vertical.',
        )

    def setup(self):
        orientation = self.options['orientation']

        if orientation == 'horizontal':
            self.io_names = {
                'vol_coef': Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
                'r_arm': Aircraft.HorizontalTail.MOMENT_RATIO,
                'wing_ref': Aircraft.Wing.AVERAGE_CHORD,
                'ar': Aircraft.HorizontalTail.ASPECT_RATIO,
                'tr': Aircraft.HorizontalTail.TAPER_RATIO,
                'area': Aircraft.HorizontalTail.AREA,
                'span': Aircraft.HorizontalTail.SPAN,
                'rchord': Aircraft.HorizontalTail.ROOT_CHORD,
                'chord': Aircraft.HorizontalTail.AVERAGE_CHORD,
                'arm': Aircraft.HorizontalTail.MOMENT_ARM,
            }
        else:
            self.io_names = {
                'vol_coef': Aircraft.VerticalTail.VOLUME_COEFFICIENT,
                'r_arm': Aircraft.VerticalTail.MOMENT_RATIO,
                'wing_ref': Aircraft.Wing.SPAN,
                'ar': Aircraft.VerticalTail.ASPECT_RATIO,
                'tr': Aircraft.VerticalTail.TAPER_RATIO,
                'area': Aircraft.VerticalTail.AREA,
                'span': Aircraft.VerticalTail.SPAN,
                'rchord': Aircraft.VerticalTail.ROOT_CHORD,
                'chord': Aircraft.VerticalTail.AVERAGE_CHORD,
                'arm': Aircraft.VerticalTail.MOMENT_ARM,
            }

        vol_coef = self.io_names['vol_coef']
        r_arm = self.io_names['r_arm']
        wing_ref = self.io_names['wing_ref']
        ar = self.io_names['ar']
        tr = self.io_names['tr']
        area = self.io_names['area']
        span = self.io_names['span']
        rchord = self.io_names['rchord']
        chord = self.io_names['chord']
        arm = self.io_names['arm']

        add_aviary_input(self, vol_coef, units='unitless')

        add_aviary_input(self, Aircraft.Wing.AREA)

        add_aviary_input(self, r_arm, units='unitless')
        add_aviary_input(self, wing_ref, units='ft')
        add_aviary_input(self, ar, units='unitless')
        add_aviary_input(self, tr, units='unitless')

        add_aviary_output(self, area, units='ft**2')
        add_aviary_output(self, span, units='ft')
        add_aviary_output(self, rchord, units='ft')
        add_aviary_output(self, chord, units='ft')
        add_aviary_output(self, arm, units='ft')

    def setup_partials(self):
        vol_coef = self.io_names['vol_coef']
        r_arm = self.io_names['r_arm']
        wing_ref = self.io_names['wing_ref']
        ar = self.io_names['ar']
        tr = self.io_names['tr']
        area = self.io_names['area']
        span = self.io_names['span']
        rchord = self.io_names['rchord']
        chord = self.io_names['chord']
        arm = self.io_names['arm']

        self.declare_partials(area, [vol_coef, Aircraft.Wing.AREA, r_arm])
        self.declare_partials(span, [vol_coef, Aircraft.Wing.AREA, r_arm, ar])
        self.declare_partials(rchord, [vol_coef, Aircraft.Wing.AREA, r_arm, ar, tr])
        self.declare_partials(chord, [vol_coef, Aircraft.Wing.AREA, r_arm, ar, tr])
        self.declare_partials(arm, [r_arm, wing_ref])

    def compute(self, inputs, outputs):
        str_area = self.io_names['area']
        str_span = self.io_names['span']
        str_rchord = self.io_names['rchord']
        str_chord = self.io_names['chord']
        str_arm = self.io_names['arm']

        vol_coef, wing_area, r_arm, wing_ref, ar, tr = inputs.values()

        area = vol_coef * wing_area * r_arm
        span = np.sqrt(area * ar)
        rchord = 2 * area / span / (1 + tr)
        chord = 2 / 3.0 * rchord * ((1 + tr) - (tr / (1 + tr)))
        arm = wing_ref / r_arm

        outputs[str_area] = area
        outputs[str_span] = span
        outputs[str_rchord] = rchord
        outputs[str_chord] = chord
        outputs[str_arm] = arm

    def compute_partials(self, inputs, J):
        str_vol_coef = self.io_names['vol_coef']
        str_r_arm = self.io_names['r_arm']
        str_wing_ref = self.io_names['wing_ref']
        str_ar = self.io_names['ar']
        str_tr = self.io_names['tr']
        str_area = self.io_names['area']
        str_span = self.io_names['span']
        str_rchord = self.io_names['rchord']
        str_chord = self.io_names['chord']
        str_arm = self.io_names['arm']

        vol_coef, wing_area, r_arm, wing_ref, ar, tr = inputs.values()

        J[str_area, str_vol_coef] = wing_area * r_arm
        J[str_area, Aircraft.Wing.AREA] = vol_coef * r_arm
        J[str_area, str_r_arm] = vol_coef * wing_area

        cse1 = np.sqrt(vol_coef * wing_area * r_arm * ar)
        J[str_span, str_vol_coef] = cse1 / (2 * vol_coef)
        J[str_span, Aircraft.Wing.AREA] = cse1 / (2 * wing_area)
        J[str_span, str_r_arm] = cse1 / (2 * r_arm)
        J[str_span, str_ar] = cse1 / (2 * ar)

        cse2 = cse1 * (tr + 1)
        J[str_rchord, str_vol_coef] = wing_area * r_arm / cse2
        J[str_rchord, Aircraft.Wing.AREA] = vol_coef * r_arm / cse2
        J[str_rchord, str_r_arm] = wing_area * vol_coef / cse2
        J[str_rchord, str_ar] = -vol_coef * wing_area * r_arm / (ar * cse2)
        J[str_rchord, str_tr] = -2 * vol_coef * wing_area * r_arm / (cse2 * (tr + 1))

        cse3 = tr - (tr / (tr + 1)) + 1
        J[str_chord, str_vol_coef] = 2 / 3.0 * wing_area * r_arm * cse3 / cse2
        J[str_chord, Aircraft.Wing.AREA] = 2 / 3.0 * vol_coef * r_arm * cse3 / cse2
        J[str_chord, str_r_arm] = 2 / 3.0 * vol_coef * wing_area * cse3 / cse2
        J[str_chord, str_ar] = -2 / 3.0 * vol_coef * wing_area * r_arm * cse3 / (ar * cse2)
        J[str_chord, str_tr] = 4 / 3.0 * cse1 * (tr - 1) / (ar * (tr + 1) ** 3)

        J[str_arm, str_r_arm] = -wing_ref / r_arm**2
        J[str_arm, str_wing_ref] = 1.0 / r_arm


class EmpennageSize(om.Group):
    """GASP geometry calculations for both horizontal and vertical tails.

    Volume coefficients for the tails may be either specified directly (default) or
    computed via empirical relationships to general airplane parameters.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF)
        add_aviary_option(self, Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF)

    def setup(self):
        # TODO: For cruciform/T-tail configurations, GASP checks to make sure the V tail
        # chord at the H tail location is greater than the H tail root chord. If not, it
        # overrides the H tail taper ratio so they match. If that leads to a H tail root
        # chord greater than the H tail tip chord, it sets the taper ratio to 1 and
        # overrides the H tail aspect ratio. H tail taper ratio is used in landing gear
        # mass calculation.

        if self.options[Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF]:
            self.add_subsystem(
                'htail_vc',
                TailVolCoef(orientation='horizontal'),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )
        if self.options[Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF]:
            self.add_subsystem(
                'vtail_vc',
                TailVolCoef(orientation='vertical'),
                promotes_inputs=['aircraft:*'],
                promotes_outputs=['aircraft:*'],
            )

        self.add_subsystem(
            'htail',
            TailSize(orientation='horizontal'),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )

        self.add_subsystem(
            'vtail',
            TailSize(orientation='vertical'),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )
