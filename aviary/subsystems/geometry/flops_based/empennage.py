"""
Size the horizontal and vertical tails of a FLOPS-based model from tail volume
coefficients.

FLOPS treats horizontal and vertical tail areas (WTIN.SHT and WTIN.SVT) as user inputs.
That is convenient when modeling an existing airframe, but it holds the tails at a fixed
size while the rest of the aircraft is resized, which produces inconsistent geometry when
a new configuration is being designed or optimized. The components here let a FLOPS-based
model derive those areas from tail volume coefficients instead, so the tails scale with
the wing.
"""

import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TailSize(om.ExplicitComponent):
    """
    Compute a tail area from its volume coefficient for FLOPS-based geometry.

    A tail volume coefficient is the non-dimensional ratio
    V = (tail area * tail moment arm) / (wing area * reference length), so the tail area
    follows from area = V * wing area * reference length / moment arm. The reference
    length is the wing mean aerodynamic chord for a horizontal tail and the wing span for
    a vertical tail.

    The mean aerodynamic chord of the trapezoidal wing is computed here from wing area,
    span, and taper ratio rather than taken as an input, because FLOPS-based geometry does
    not provide Aircraft.Wing.AVERAGE_CHORD; that variable is only computed by GASP-based
    geometry.
    """

    def initialize(self):
        self.options.declare(
            'orientation',
            values=['horizontal', 'vertical'],
            desc='Tail orientation, can be horizontal or vertical.',
        )

    def setup(self):
        horizontal = self.options['orientation'] == 'horizontal'

        if horizontal:
            self.io_names = {
                'vol_coef': Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
                'arm': Aircraft.HorizontalTail.MOMENT_ARM,
                'area': Aircraft.HorizontalTail.AREA,
            }
        else:
            self.io_names = {
                'vol_coef': Aircraft.VerticalTail.VOLUME_COEFFICIENT,
                'arm': Aircraft.VerticalTail.MOMENT_ARM,
                'area': Aircraft.VerticalTail.AREA,
            }

        add_aviary_input(self, self.io_names['vol_coef'], units='unitless')
        add_aviary_input(self, self.io_names['arm'], units='ft')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        if horizontal:
            add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')

        add_aviary_output(self, self.io_names['area'], units='ft**2')

    def setup_partials(self):
        wrt = [
            self.io_names['vol_coef'],
            self.io_names['arm'],
            Aircraft.Wing.AREA,
            Aircraft.Wing.SPAN,
        ]

        if self.options['orientation'] == 'horizontal':
            wrt.append(Aircraft.Wing.TAPER_RATIO)

        self.declare_partials(self.io_names['area'], wrt)

    def _check_inputs(self, inputs):
        """Guard the divisions below with messages that name the offending variable."""
        if inputs[Aircraft.Wing.SPAN] <= 0.0:
            raise ValueError(f'{Aircraft.Wing.SPAN} must be positive.')

        arm_name = self.io_names['arm']
        if inputs[arm_name] <= 0.0:
            raise ValueError(
                f'{arm_name} must be positive when the tail area is computed from its '
                'volume coefficient.'
            )

        if self.options['orientation'] == 'horizontal':
            if inputs[Aircraft.Wing.TAPER_RATIO] <= -1.0:
                raise ValueError(f'{Aircraft.Wing.TAPER_RATIO} must be greater than -1.')

    def compute(self, inputs, outputs):
        self._check_inputs(inputs)

        vol_coef = inputs[self.io_names['vol_coef']]
        arm = inputs[self.io_names['arm']]
        wing_area = inputs[Aircraft.Wing.AREA]
        wing_span = inputs[Aircraft.Wing.SPAN]

        if self.options['orientation'] == 'horizontal':
            # Mean aerodynamic chord of a trapezoidal wing, written in terms of area and
            # span: MAC = 4 * area / (3 * span) * (1 + tr + tr**2) / (1 + tr)**2
            taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
            ref_length = (
                4.0
                * wing_area
                / (3.0 * wing_span)
                * (1.0 + taper_ratio + taper_ratio**2)
                / (1.0 + taper_ratio) ** 2
            )
        else:
            ref_length = wing_span

        outputs[self.io_names['area']] = vol_coef * wing_area * ref_length / arm

    def compute_partials(self, inputs, J):
        self._check_inputs(inputs)

        str_vol_coef = self.io_names['vol_coef']
        str_arm = self.io_names['arm']
        str_area = self.io_names['area']

        vol_coef = inputs[str_vol_coef]
        arm = inputs[str_arm]
        wing_area = inputs[Aircraft.Wing.AREA]
        wing_span = inputs[Aircraft.Wing.SPAN]

        if self.options['orientation'] == 'horizontal':
            taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]

            # With f = (1 + tr + tr**2) / (1 + tr)**2, the area is
            # 4 / 3 * vol_coef * wing_area**2 * f / (wing_span * arm).
            f = (1.0 + taper_ratio + taper_ratio**2) / (1.0 + taper_ratio) ** 2
            # df/d(taper_ratio) simplifies to (tr - 1) / (1 + tr)**3
            df = (taper_ratio - 1.0) / (1.0 + taper_ratio) ** 3

            coef = 4.0 / 3.0 / (wing_span * arm)

            J[str_area, str_vol_coef] = coef * wing_area**2 * f
            J[str_area, Aircraft.Wing.AREA] = 2.0 * coef * vol_coef * wing_area * f
            J[str_area, Aircraft.Wing.SPAN] = -coef * vol_coef * wing_area**2 * f / wing_span
            J[str_area, str_arm] = -coef * vol_coef * wing_area**2 * f / arm
            J[str_area, Aircraft.Wing.TAPER_RATIO] = coef * vol_coef * wing_area**2 * df

        else:
            # area = vol_coef * wing_area * wing_span / arm
            J[str_area, str_vol_coef] = wing_area * wing_span / arm
            J[str_area, Aircraft.Wing.AREA] = vol_coef * wing_span / arm
            J[str_area, Aircraft.Wing.SPAN] = vol_coef * wing_area / arm
            J[str_area, str_arm] = -vol_coef * wing_area * wing_span / arm**2


class EmpennageSize(om.Group):
    """
    Size whichever tails a FLOPS-based model has asked to have computed.

    Each tail is sized only when its corresponding option is set, so a model that supplies
    tail areas directly is unaffected.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.COMPUTE_HTAIL_AREA)
        add_aviary_option(self, Aircraft.Design.COMPUTE_VTAIL_AREA)

    def setup(self):
        if self.options[Aircraft.Design.COMPUTE_HTAIL_AREA]:
            self.add_subsystem(
                'htail',
                TailSize(orientation='horizontal'),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        if self.options[Aircraft.Design.COMPUTE_VTAIL_AREA]:
            self.add_subsystem(
                'vtail',
                TailSize(orientation='vertical'),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
