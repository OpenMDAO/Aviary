"""
Define utilities for calculating the effect the ground has on lift-to-drag ratio for
aircraft flying in close proximity to the ground.

References
----------
.. [1] DeYoung, John. "Advanced Supersonic Technology Concept Study Reference
    Characteristics," NASA Contractor Report 132374.
"""

import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic


class GroundEffect(om.ExplicitComponent):
    """
    Define a component for adjusting lift and drag to accommodate ground effect.

    Note
    ----
    It is an error (no diagnostic) to try to calculate ground effect for an
    aircraft flying "underground" - current `altitude` must ALWAYS be greater than or
    equal to `ground_altitude`.
    """

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'ground_altitude',
            default=0.0,
            types=float,
            desc='true altitude of the ground from mean sea level (m)',
        )

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.zeros(nn), units='rad')

        add_aviary_input(self, Dynamic.Mission.ALTITUDE, shape=nn, units='m')

        add_aviary_input(self, Dynamic.Mission.FLIGHT_PATH_ANGLE, shape=(nn), units='rad')

        self.add_input(
            'minimum_drag_coefficient', 0.0, desc='coefficient of minimum drag', units='unitless'
        )

        self.add_input(
            'base_lift_coefficient',
            val=np.ones(nn),
            desc='coefficient of lift without ground effect',
            units='unitless',
        )

        self.add_input(
            'base_drag_coefficient',
            val=np.ones(nn),
            desc='coefficient of drag without ground effect',
            units='unitless',
        )

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.HEIGHT, units='m')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='m')

        self.add_output(
            'lift_coefficient',
            val=np.ones(nn),
            desc='coefficient of lift with ground effect',
            units='unitless',
        )

        self.add_output(
            'drag_coefficient',
            val=np.ones(nn),
            desc='coefficient of drag with ground effect',
            units='unitless',
        )

    def setup_partials(self):
        rows_cols = np.arange(self.options['num_nodes'])

        self.declare_partials(
            'lift_coefficient',
            [Aircraft.Wing.ASPECT_RATIO, Aircraft.Wing.HEIGHT, Aircraft.Wing.SPAN],
        )

        self.declare_partials(
            'lift_coefficient',
            [Dynamic.Mission.ALTITUDE, 'base_lift_coefficient'],
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials(
            'lift_coefficient',
            [
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                'minimum_drag_coefficient',
                'base_drag_coefficient',
            ],
            dependent=False,
        )

        self.declare_partials(
            'drag_coefficient',
            [Aircraft.Wing.ASPECT_RATIO, Aircraft.Wing.HEIGHT, Aircraft.Wing.SPAN],
        )

        self.declare_partials(
            'drag_coefficient',
            [
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Mission.ALTITUDE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                'base_drag_coefficient',
                'base_lift_coefficient',
            ],
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials(
            'drag_coefficient',
            'minimum_drag_coefficient',
            rows=rows_cols,
            cols=np.zeros(self.options['num_nodes']),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        options = self.options

        ground_altitude = options['ground_altitude']

        angle_of_attack = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        altitude = inputs[Dynamic.Mission.ALTITUDE]
        flight_path_angle = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        minimum_drag_coefficient = inputs['minimum_drag_coefficient']
        base_lift_coefficient = inputs['base_lift_coefficient']
        base_drag_coefficient = inputs['base_drag_coefficient']
        aspect_ratio = inputs[Aircraft.Wing.ASPECT_RATIO]
        height = inputs[Aircraft.Wing.HEIGHT]
        span = inputs[Aircraft.Wing.SPAN]

        ground_effect_state = ((altitude - ground_altitude) + height) / span
        height_factor = np.ones_like(ground_effect_state)

        idx = np.where(ground_effect_state > 1.0)
        height_factor[idx] = 10.0 * (1.1 - ground_effect_state[idx])

        aspect_ratio_term = (6.0 + aspect_ratio) ** 2 / (36.0 + aspect_ratio)
        ground_effect_term0 = 32.0 * (ground_effect_state * aspect_ratio_term) ** 2 + 1.0

        lift_coeff_factor_denom = (
            ground_effect_term0
            - 0.5
            + 4.0 * ground_effect_state * aspect_ratio_term * np.sqrt(ground_effect_term0)
        )

        lift_coeff_factor = 1.0 + height_factor / lift_coeff_factor_denom

        lift_coefficient = base_lift_coefficient * lift_coeff_factor

        combined_angle = angle_of_attack + flight_path_angle
        ground_effect_term1 = 1.0 + 32.0 * ground_effect_state**2

        drag_coeff_factor_denom = (
            4.0 * ground_effect_state * np.sqrt(ground_effect_term1) + ground_effect_term1
        )

        drag_coeff_factor = 1.0 - height_factor / drag_coeff_factor_denom

        drag_coefficient = (
            minimum_drag_coefficient
            + lift_coeff_factor
            * drag_coeff_factor
            * (base_drag_coefficient - minimum_drag_coefficient)
            + combined_angle * base_lift_coefficient * (lift_coeff_factor - 1.0)
        )

        # Check for out of ground effect.
        idx = np.where(ground_effect_state > 1.1)
        drag_coefficient[idx] = base_drag_coefficient[idx]
        lift_coefficient[idx] = base_lift_coefficient[idx]

        outputs['lift_coefficient'] = lift_coefficient
        outputs['drag_coefficient'] = drag_coefficient

    def compute_partials(self, inputs, J, discrete_inputs=None):
        options = self.options

        ground_altitude = options['ground_altitude']

        angle_of_attack = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        altitude = inputs[Dynamic.Mission.ALTITUDE]
        flight_path_angle = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        minimum_drag_coefficient = inputs['minimum_drag_coefficient']
        base_lift_coefficient = inputs['base_lift_coefficient']
        base_drag_coefficient = inputs['base_drag_coefficient']
        aspect_ratio = inputs[Aircraft.Wing.ASPECT_RATIO]
        height = inputs[Aircraft.Wing.HEIGHT]
        span = inputs[Aircraft.Wing.SPAN]

        # region lift_coefficient wrt [altitude, base_lift_coefficient]
        ground_effect_state = ((altitude - ground_altitude) + height) / span

        d_ges_alt = 1.0 / span

        height_factor = np.ones_like(ground_effect_state)
        d_hf_alt = np.zeros_like(ground_effect_state)

        idx = np.where(ground_effect_state > 1.0)
        height_factor[idx] = 10.0 * (1.1 - ground_effect_state[idx])
        d_hf_alt[idx] = -10.0 / span

        aspect_ratio_term = (6.0 + aspect_ratio) ** 2 / (36.0 + aspect_ratio)
        ground_effect_term0 = 32.0 * (ground_effect_state * aspect_ratio_term) ** 2 + 1.0

        d_get0_alt = (64.0 * ground_effect_state * aspect_ratio_term**2) * d_ges_alt

        sqrt_get0 = np.sqrt(ground_effect_term0)

        lift_coeff_factor_denom = (
            ground_effect_term0 - 0.5 + 4.0 * ground_effect_state * aspect_ratio_term * sqrt_get0
        )

        d_lcfd_alt = d_get0_alt + 4.0 * aspect_ratio_term * (
            d_ges_alt * sqrt_get0 + ground_effect_state * (0.5 / sqrt_get0) * d_get0_alt
        )

        lift_coeff_factor = 1.0 + height_factor / lift_coeff_factor_denom

        d_lcf_alt = (
            (d_hf_alt * lift_coeff_factor_denom) - (height_factor * d_lcfd_alt)
        ) / lift_coeff_factor_denom**2

        J['lift_coefficient', Dynamic.Mission.ALTITUDE] = base_lift_coefficient * d_lcf_alt

        J['lift_coefficient', 'base_lift_coefficient'] = lift_coeff_factor
        # endregion lift_coefficient wrt [altitude, base_lift_coefficient]

        # region lift_coefficient wrt Aircraft.Wing.ASPECT_RATIO
        f = (6.0 + aspect_ratio) ** 2
        d_f = 2.0 * (6.0 + aspect_ratio)
        g = 36.0 + aspect_ratio
        d_g = 1.0

        d_art_ar = ((d_f * g) - (f * d_g)) / g**2

        d_get0_ar = 64.0 * (ground_effect_state**2 * aspect_ratio_term) * d_art_ar

        d_lcfd_ar = d_get0_ar + 4.0 * ground_effect_state * (
            d_art_ar * sqrt_get0 + aspect_ratio_term * 0.5 / sqrt_get0 * d_get0_ar
        )

        f = height_factor
        d_f = 0.0
        g = lift_coeff_factor_denom
        d_g = d_lcfd_ar

        d_lcf_ar = -f * d_g / g**2

        J['lift_coefficient', Aircraft.Wing.ASPECT_RATIO] = base_lift_coefficient * d_lcf_ar
        # endregion lift_coefficient wrt Aircraft.Wing.ASPECT_RATIO

        J['lift_coefficient', Aircraft.Wing.HEIGHT] = base_lift_coefficient * d_lcf_alt

        # region lift_coefficient wrt Aircraft.Wing.SPAN
        d_ges_b = -((altitude - ground_altitude) + height) / span**2

        d_hf_b = np.zeros_like(ground_effect_state)
        d_hf_b[idx] = -10.0 * d_ges_b[idx]

        d_get0_b = 64.0 * (ground_effect_state * aspect_ratio_term**2) * d_ges_b

        d_lcfd_b = d_get0_b + 4.0 * aspect_ratio_term * (
            (d_ges_b * sqrt_get0) + (ground_effect_state * 0.5 / sqrt_get0 * d_get0_b)
        )

        f = height_factor
        d_f = d_hf_b
        g = lift_coeff_factor_denom
        d_g = d_lcfd_b

        d_lcf_b = ((d_f * g) - (f * d_g)) / g**2

        J['lift_coefficient', Aircraft.Wing.SPAN] = base_lift_coefficient * d_lcf_b
        # endregion lift_coefficient wrt Aircraft.Wing.SPAN

        # region drag_coefficient wrt angle_of_attack
        combined_angle = angle_of_attack + flight_path_angle

        d_ca_aoa = 1.0

        ground_effect_term1 = 1.0 + 32.0 * ground_effect_state**2

        sqrt_get1 = np.sqrt(ground_effect_term1)

        drag_coeff_factor_denom = 4.0 * ground_effect_state * sqrt_get1 + ground_effect_term1

        drag_coeff_factor = 1.0 - height_factor / drag_coeff_factor_denom

        d_dc_aoa = base_lift_coefficient * (lift_coeff_factor - 1.0) * d_ca_aoa

        J['drag_coefficient', Dynamic.Vehicle.ANGLE_OF_ATTACK] = d_dc_aoa
        # endregion drag_coefficient wrt angle_of_attack

        # region drag_coefficient wrt flight_path_angle
        d_ca_fpa = 1.0

        d_dc_fpa = base_lift_coefficient * (lift_coeff_factor - 1.0) * d_ca_fpa

        J['drag_coefficient', Dynamic.Mission.FLIGHT_PATH_ANGLE] = d_dc_fpa
        # endregion drag_coefficient wrt flight_path_angle

        # region drag_coefficient wrt altitude
        d_get1_alt = 64.0 * ground_effect_state * d_ges_alt

        f = ground_effect_state
        d_f = d_ges_alt
        g = sqrt_get1
        d_g = 0.5 * d_get1_alt / sqrt_get1

        d_dcfd_alt = 4.0 * ((d_f * g) + (f * d_g)) + d_get1_alt

        f = height_factor
        d_f = d_hf_alt
        g = drag_coeff_factor_denom
        d_g = d_dcfd_alt

        d_dcf_alt = -((d_f * g) - (f * d_g)) / g**2

        f = lift_coeff_factor
        d_f = d_lcf_alt
        g = drag_coeff_factor
        d_g = d_dcf_alt

        d_dc_alt = ((d_f * g) + (d_g * f)) * (
            base_drag_coefficient - minimum_drag_coefficient
        ) + combined_angle * base_lift_coefficient * d_lcf_alt

        J['drag_coefficient', Dynamic.Mission.ALTITUDE] = d_dc_alt
        # endregion drag_coefficient wrt altitude

        # region drag_coefficient wrt minimum_drag_coefficient
        d_dc_mdc = 1.0 - lift_coeff_factor * drag_coeff_factor

        J['drag_coefficient', 'minimum_drag_coefficient'] = d_dc_mdc
        # endregion drag_coefficient wrt minimum_drag_coefficient

        # region drag_coefficient wrt base_lift_coefficient
        d_dc_blc = combined_angle * (lift_coeff_factor - 1.0)

        J['drag_coefficient', 'base_lift_coefficient'] = d_dc_blc
        # endregion drag_coefficient wrt base_lift_coefficient

        # region drag_coefficient wrt base_drag_coefficient
        d_dc_bdc = lift_coeff_factor * drag_coeff_factor

        J['drag_coefficient', 'base_drag_coefficient'] = d_dc_bdc
        # endregion drag_coefficient wrt base_drag_coefficient

        # region drag_coefficient wrt Aircraft.Wing.ASPECT_RATIO
        d_dc_ar = (
            drag_coeff_factor * (base_drag_coefficient - minimum_drag_coefficient)
            + combined_angle * base_lift_coefficient
        ) * d_lcf_ar

        J['drag_coefficient', Aircraft.Wing.ASPECT_RATIO] = d_dc_ar
        # endregion drag_coefficient wrt Aircraft.Wing.ASPECT_RATIO

        J['drag_coefficient', Aircraft.Wing.HEIGHT] = d_dc_alt

        # region drag_coefficient wrt Aircraft.Wing.SPAN
        d_get1_b = 64.0 * ground_effect_state * d_ges_b

        f = ground_effect_state
        d_f = d_ges_b
        g = sqrt_get1
        d_g = 0.5 * d_get1_b / sqrt_get1

        d_dcfd_b = 4.0 * ((d_f * g) + (f * d_g)) + d_get1_b

        f = height_factor
        d_f = d_hf_b
        g = drag_coeff_factor_denom
        d_g = d_dcfd_b

        d_dcf_b = -((d_f * g) - (f * d_g)) / g**2

        f = lift_coeff_factor
        d_f = d_lcf_b
        g = drag_coeff_factor
        d_g = d_dcf_b

        d_dc_b = ((d_f * g) + (f * d_g)) * (
            base_drag_coefficient - minimum_drag_coefficient
        ) + combined_angle * base_lift_coefficient * d_lcf_b

        J['drag_coefficient', Aircraft.Wing.SPAN] = d_dc_b
        # endregion drag_coefficient wrt Aircraft.Wing.SPAN

        # Check for out of ground effect.
        idx = np.where(ground_effect_state > 1.1)
        if idx:
            J['drag_coefficient', Dynamic.Mission.ALTITUDE][idx] = 0.0
            J['drag_coefficient', 'minimum_drag_coefficient'][idx] = 0.0
            J['drag_coefficient', 'base_lift_coefficient'][idx] = 0.0
            J['drag_coefficient', 'base_drag_coefficient'][idx] = 1.0
            J['drag_coefficient', Aircraft.Wing.ASPECT_RATIO][idx] = 0.0
            J['drag_coefficient', Aircraft.Wing.HEIGHT][idx] = 0.0
            J['drag_coefficient', Aircraft.Wing.SPAN][idx] = 0.0
            J['drag_coefficient', Dynamic.Vehicle.ANGLE_OF_ATTACK][idx] = 0.0
            J['drag_coefficient', Dynamic.Mission.FLIGHT_PATH_ANGLE][idx] = 0.0

            J['lift_coefficient', Dynamic.Mission.ALTITUDE][idx] = 0.0
            J['lift_coefficient', 'base_lift_coefficient'][idx] = 1.0
            J['lift_coefficient', Aircraft.Wing.ASPECT_RATIO][idx] = 0.0
            J['lift_coefficient', Aircraft.Wing.HEIGHT][idx] = 0.0
            J['lift_coefficient', Aircraft.Wing.SPAN][idx] = 0.0
