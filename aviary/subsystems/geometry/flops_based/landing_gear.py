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

    TODO does not support more than two wing engines, or more than one engine model
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.NUM_WING_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        num_wing_engines_total = sum(self.options[Aircraft.Engine.NUM_WING_ENGINES])

        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')
        if num_wing_engines_total > 1:
            add_aviary_input(
                self,
                Aircraft.Engine.WING_LOCATIONS,
                shape=int(num_wing_engines_total / 2),
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
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES][0]

        # TODO temp using first engine, heterogeneous engines not supported
        num_wing_eng = self.options[Aircraft.Engine.NUM_WING_ENGINES][0]

        # TODO: high engine-count configuration.
        y_eng_aft = 0

        if num_wing_eng > 0:
            y_eng_fore = inputs[Aircraft.Engine.WING_LOCATIONS][0]

            tan_dih = np.tan(inputs[Aircraft.Wing.DIHEDRAL] * DEG2RAD)
            fuse_half_width = inputs[Aircraft.Fuselage.MAX_WIDTH] * 6.0

            d_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER][0]
            # f_nacelle = d_nacelle
            # if num_eng > 4:
            #     f_nacelle = 0.5 * d_nacelle * num_eng ** 0.5

            f_nacelle = distributed_nacelle_diam_factor(d_nacelle, num_eng)

            yee = y_eng_fore
            if num_wing_eng > 2 and y_eng_aft > 0.0:
                yee = y_eng_aft

            if yee < 1.0:
                # This is triggered when the input engine locations are normalized.
                yee *= 6.0 * inputs[Aircraft.Wing.SPAN]

            cmlg = 12.0 * f_nacelle + (0.26 - tan_dih) * (yee - fuse_half_width)

        else:
            cmlg = 0.0

        if cmlg < 12.0:
            cmlg = 0.75 * inputs[Aircraft.Fuselage.LENGTH]

        outputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH] = cmlg

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # TODO temp using first engine, heterogeneous engines not supported
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES][0]
        num_wing_eng = self.options[Aircraft.Engine.NUM_WING_ENGINES][0]

        y_eng_aft = 0

        if num_wing_eng > 0:
            y_eng_fore = inputs[Aircraft.Engine.WING_LOCATIONS][0]

            tan_dih = np.tan(inputs[Aircraft.Wing.DIHEDRAL] * DEG2RAD)
            dtan_dih = DEG2RAD / np.cos(inputs[Aircraft.Wing.DIHEDRAL] * DEG2RAD) ** 2

            fuse_half_width = inputs[Aircraft.Fuselage.MAX_WIDTH] * 6.0
            dhw_dfuse_wid = 6.0

            d_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER][0]
            # f_nacelle = d_nacelle
            # d_nac = 1.0
            # if num_eng > 4:
            #     f_nacelle = 0.5 * d_nacelle * num_eng ** 0.5
            #     d_nac = 0.5 * num_eng ** 0.5

            f_nacelle = distributed_nacelle_diam_factor(d_nacelle, num_eng)
            d_nac = distributed_nacelle_diam_factor_deriv(num_eng)

            yee = y_eng_fore
            if num_wing_eng > 2 and y_eng_aft > 0.0:
                yee = y_eng_aft

            dyee_dwel = 1.0
            dyee_dspan = 1.0
            if yee < 1.0:
                dyee_dwel = 6.0 * inputs[Aircraft.Wing.SPAN]
                dyee_dspan = 6.0 * yee

                yee *= 6.0 * inputs[Aircraft.Wing.SPAN]

            cmlg = 12.0 * f_nacelle + (0.26 - tan_dih) * (yee - fuse_half_width)
            dcmlg_dnac = 12.0 * d_nac
            dcmlg_dtan = -(yee - fuse_half_width)
            dcmlg_dyee = 0.26 - tan_dih
            dcmlg_dhw = tan_dih - 0.26

        else:
            cmlg = 0.0

        if cmlg < 12.0:
            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Fuselage.LENGTH] = 0.75

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Fuselage.MAX_WIDTH] = 0.0

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Nacelle.AVG_DIAMETER] = (
                0.0
            )

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Engine.WING_LOCATIONS] = (
                0.0
            )

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Wing.DIHEDRAL] = 0.0

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Wing.SPAN] = 0.0

        else:
            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Fuselage.LENGTH] = 0.0

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Fuselage.MAX_WIDTH] = (
                dcmlg_dhw * dhw_dfuse_wid
            )

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Nacelle.AVG_DIAMETER][
                :
            ] = dcmlg_dnac

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Engine.WING_LOCATIONS] = (
                dcmlg_dyee * dyee_dwel
            )

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Wing.DIHEDRAL] = (
                dcmlg_dtan * dtan_dih
            )

            partials[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, Aircraft.Wing.SPAN] = (
                dcmlg_dyee * dyee_dspan
            )
