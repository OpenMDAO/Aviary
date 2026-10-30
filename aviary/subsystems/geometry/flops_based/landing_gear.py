import numpy as np
import openmdao.api as om

from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_nacelle_diam_factor,
    distributed_nacelle_diam_factor_deriv,
)
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft

DEG2RAD = np.pi / 180.0


class NoseGearLength(om.ExplicitComponent):
    """
    Computation of nose gear oleo strut length from main gear oleo strut length:
    NOSE_GEAR_OLEO_LENGTH = 0.7 * MAIN_GEAR_OLEO_LENGTH.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_output(self, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, units='inch')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
            Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
            val=0.7,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH] = (
            0.7 * inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        )


class MainGearLength(om.ExplicitComponent):
    """
    Computation of main gear length.

    See issue #1183 does not support more than two wing engines, or more than one engine model
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.NUM_WING_ENGINES)

    def setup(self):
        num_engines = np.atleast_1d(self.options[Aircraft.Engine.NUM_ENGINES])
        num_wing_engines = np.atleast_1d(self.options[Aircraft.Engine.NUM_WING_ENGINES])

        num_engine_type = len(num_engines)
        num_wing_engines_total = int(np.sum(num_wing_engines))

        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')

        if num_wing_engines_total > 1:
            add_aviary_input(
                self,
                Aircraft.Engine.WING_LOCATIONS,
                shape=num_wing_engines_total // 2,
                units='unitless',
            )
        else:  # this case is not tested
            add_aviary_input(self, Aircraft.Engine.WING_LOCATIONS, units='unitless')

        add_aviary_input(self, Aircraft.Wing.DIHEDRAL, units='deg')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        add_aviary_output(self, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, units='inch')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_engines = np.atleast_1d(self.options[Aircraft.Engine.NUM_ENGINES])
        num_wing_engines = np.atleast_1d(self.options[Aircraft.Engine.NUM_WING_ENGINES])

        tan_dih = np.tan(inputs[Aircraft.Wing.DIHEDRAL][0] * DEG2RAD)
        fuse_half_width = inputs[Aircraft.Fuselage.MAX_WIDTH][0] * 6.0
        wing_span = inputs[Aircraft.Wing.SPAN][0]

        locations = np.atleast_1d(inputs[Aircraft.Engine.WING_LOCATIONS])
        diameters = np.atleast_1d(inputs[Aircraft.Nacelle.AVG_DIAMETER])

        max_cmlg = 0.0
        loc_idx = 0

        # Track active variables so we can compute correct partials for the limiting engine
        self._fallback_active = False
        self._active_loc_idx = 0
        self._active_eng_idx = 0
        self._active_yee_raw = 0.0
        self._active_n_total = 0

        # Iterate through each engine type
        for eng_idx, n_wing in enumerate(num_wing_engines):
            pairs_of_wing_engines = int(n_wing) // 2

            # Iterate through spanwise locations for this engine type
            for _ in range(pairs_of_wing_engines):
                yee_raw = locations[loc_idx]
                yee = yee_raw

                # This is triggered when the input engine locations are normalized.
                if yee < 1.0:
                    yee *= 6.0 * wing_span

                d_nacelle = diameters[eng_idx]
                n_total = int(num_engines[eng_idx])

                f_nacelle = distributed_nacelle_diam_factor(d_nacelle, n_total)
                cmlg = 12.0 * f_nacelle + (0.26 - tan_dih) * (yee - fuse_half_width)

                # Check if this engine sizes the gear
                if cmlg > max_cmlg:
                    max_cmlg = cmlg

                    # Save the "active" parameters for the derivative calculation
                    self._active_loc_idx = loc_idx
                    self._active_eng_idx = eng_idx
                    self._active_yee_raw = yee_raw
                    self._active_n_total = n_total

                loc_idx += 1

        # Fallback if no wing engines exist or if calculated gear length is too small
        if max_cmlg < 12.0:
            max_cmlg = 0.75 * inputs[Aircraft.Fuselage.LENGTH][0]
            self._fallback_active = True

        outputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH] = max_cmlg

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # Initialize all partials to zero because the "active" limiting engine might change
        partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Fuselage.LENGTH] = 0.0
        partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Fuselage.MAX_WIDTH] = 0.0
        partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Wing.DIHEDRAL] = 0.0
        partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Wing.SPAN] = 0.0

        # Zero out the array partials
        partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Nacelle.AVG_DIAMETER] *= 0.0
        partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Engine.WING_LOCATIONS] *= 0.0

        if self._fallback_active:
            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Fuselage.LENGTH] = 0.75

        else:
            # We only compute gradients with respect to the engine that sized the gear length
            tan_dih = np.tan(inputs[Aircraft.Wing.DIHEDRAL][0] * DEG2RAD)
            dtan_dih = DEG2RAD / np.cos(inputs[Aircraft.Wing.DIHEDRAL][0] * DEG2RAD) ** 2

            fuse_half_width = inputs[Aircraft.Fuselage.MAX_WIDTH][0] * 6.0
            dhw_dfuse_wid = 6.0

            yee_raw = self._active_yee_raw
            yee = yee_raw

            dyee_dwel = 1.0
            dyee_dspan = 0.0

            if yee_raw < 1.0:
                dyee_dwel = 6.0 * inputs[Aircraft.Wing.SPAN][0]
                dyee_dspan = 6.0 * yee_raw
                yee *= 6.0 * inputs[Aircraft.Wing.SPAN][0]

            d_nac = distributed_nacelle_diam_factor_deriv(self._active_n_total)

            dcmlg_dnac = 12.0 * d_nac
            dcmlg_dtan = -(yee - fuse_half_width)
            dcmlg_dyee = 0.26 - tan_dih
            dcmlg_dhw = tan_dih - 0.26

            # Always apply gradients to the fuselage & wing parameters
            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Fuselage.MAX_WIDTH] = (
                dcmlg_dhw * dhw_dfuse_wid
            )
            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Wing.DIHEDRAL] = (
                dcmlg_dtan * dtan_dih
            )
            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Wing.SPAN] = (
                dcmlg_dyee * dyee_dspan
            )

            # Only apply array gradients to the limiting engine type and specific location
            partials[
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Nacelle.AVG_DIAMETER
            ].flat[self._active_eng_idx] = dcmlg_dnac
            partials[
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Engine.WING_LOCATIONS
            ].flat[self._active_loc_idx] = dcmlg_dyee * dyee_dwel
