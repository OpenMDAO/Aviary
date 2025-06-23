import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_nacelle_diam_factor,
    distributed_nacelle_diam_factor_deriv,
)
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission

DEG2RAD = np.pi / 180.0


class LandingGearMass(om.ExplicitComponent):
    """
    Calculate the mass of the landing gear. The methodology is based on the
    FLOPS weight equations, modified to output mass instead of weight.
    """

    # TODO: add in aircraft type and carrier factors as options and modify
    # equations

    def setup(self):
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Design.TOUCHDOWN_MASS, units='lbm')

        add_aviary_output(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')
        add_aviary_output(self, Aircraft.LandingGear.NOSE_GEAR_MASS, units='lbm')

        # TODO landing weight is not a landing_gear component level variable
        # self.add_input('aircraft:landing_gear:weights:landing_weight', val=0.0, desc='design landing weight', units='lbf')
        # self.add_input('aircraft:landing_gear:dimensions:extend_main_gear_oleo_len', val=0.0, desc='length of extended \
        #     main landing gear oleo', units='inch')
        # self.add_input('aircraft:landing_gear:dimensions:extend_nose_gear_oleo_len', val=0.0, desc='length of extended \
        #     nose landing gear oleo', units='inch')
        # self.add_input('TBD:aircraft:landing_gear:main_landing_gear_weight_multipler', val=1.0, desc='weight multiplier for \
        #     main landing gear weight', units='unitless')
        # self.add_input('TBD:aircraft:landing_gear:nose_landing_gear_weight_multipler', val=1.0, desc='weight multiplier for \
        #     nose landing gear weight', units='unitless')

        # self.add_output('TBD:landing_gear:weights:main_landing_gear_weight', val=0.0, desc='main landing gear weight', units='lbf')
        # self.add_output('TBD:landing_gear:weights:nose_landing_gear_weight', val=0.0, desc='nose landing gear weight', units='lbf')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.MAIN_GEAR_MASS,
            [
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
                Aircraft.Design.TOUCHDOWN_MASS,
            ],
        )
        self.declare_partials(
            Aircraft.LandingGear.NOSE_GEAR_MASS,
            [
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
                Aircraft.Design.TOUCHDOWN_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        main_gear_length = inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        main_gear_scaler = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER]
        nose_gear_length = inputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH]
        nose_gear_scaler = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER]
        landing_weight = inputs[Aircraft.Design.TOUCHDOWN_MASS] * GRAV_ENGLISH_LBM

        outputs[Aircraft.LandingGear.MAIN_GEAR_MASS] = (
            0.0117
            * landing_weight**0.95
            * main_gear_length**0.43
            * main_gear_scaler
            / GRAV_ENGLISH_LBM
        )
        outputs[Aircraft.LandingGear.NOSE_GEAR_MASS] = (
            0.048
            * landing_weight**0.67
            * nose_gear_length**0.43
            * nose_gear_scaler
            / GRAV_ENGLISH_LBM
        )

        # main_gear_weight = (0.0117 - 0.0012 * type_factor) * landing_weight**0.95 * main_gear_length**0.43
        # outputs['TBD:landing_gear:weights:main_landing_gear_weight'] = main_gear_weight * inputs['TBD:aircraft:landing_gear:main_landing_gear_weight_multipler']
        # nose_gear_weight = (0.048 - 0.008 * type_factor) * landing_weight**0.67 * nose_gear_length**0.43 * (1 + 0.8*carrier_factor)
        # outputs['TBD:landing_gear:weights:nose_landing_gear_weight'] = nose_gear_weight * inputs['TBD:aircraft:landing_gear:nose_landing_gear_weight_multipler']

    def compute_partials(self, inputs, J):
        main_gear_length = inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        main_gear_scaler = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER]
        nose_gear_length = inputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH]
        nose_gear_scaler = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER]
        landing_weight = inputs[Aircraft.Design.TOUCHDOWN_MASS] * GRAV_ENGLISH_LBM

        landing_weight_exp1 = landing_weight**0.95
        landing_weight_exp2 = landing_weight**0.67
        main_gear_length_exp = main_gear_length**0.43
        nose_gear_length_exp = nose_gear_length**0.43

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH] = (
            0.005031
            * landing_weight_exp1
            * main_gear_length**-0.57
            * main_gear_scaler
            / GRAV_ENGLISH_LBM
        )
        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER] = (
            0.0117 * landing_weight_exp1 * main_gear_length_exp / GRAV_ENGLISH_LBM
        )
        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.Design.TOUCHDOWN_MASS] = (
            0.011115 * landing_weight**-0.05 * main_gear_length_exp * main_gear_scaler
        )

        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH] = (
            0.02064
            * landing_weight_exp2
            * nose_gear_length**-0.57
            * nose_gear_scaler
            / GRAV_ENGLISH_LBM
        )
        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER] = (
            0.048 * landing_weight_exp2 * nose_gear_length_exp / GRAV_ENGLISH_LBM
        )
        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.Design.TOUCHDOWN_MASS] = (
            0.03216 * landing_weight**-0.33 * nose_gear_length_exp * nose_gear_scaler
        )


class AltLandingGearMass(om.ExplicitComponent):
    """
    Calculate the mass of the landing gear using the alternate method.
    The methodology is based on the FLOPS weight equations, modified
    to output mass instead of weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, units='unitless')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')

        add_aviary_output(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')
        add_aviary_output(self, Aircraft.LandingGear.NOSE_GEAR_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.MAIN_GEAR_MASS,
            [
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
                Mission.Design.GROSS_MASS,
            ],
        )
        self.declare_partials(
            Aircraft.LandingGear.NOSE_GEAR_MASS,
            [
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
                Mission.Design.GROSS_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        main_gear_length = inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        main_gear_scaler = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER]
        nose_gear_length = inputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH]
        nose_gear_scaler = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM

        total_gear_weight = gross_weight * (
            (
                30100.0
                + 0.3876 * main_gear_length * main_gear_length
                + 0.09579 * nose_gear_length * nose_gear_length
            )
            / 1.0e6
        )

        outputs[Aircraft.LandingGear.MAIN_GEAR_MASS] = (
            0.85 * total_gear_weight * main_gear_scaler / GRAV_ENGLISH_LBM
        )
        outputs[Aircraft.LandingGear.NOSE_GEAR_MASS] = (
            0.15 * total_gear_weight * nose_gear_scaler / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        main_gear_length = inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        main_gear_scaler = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER]
        nose_gear_length = inputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH]
        nose_gear_scaler = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM

        total_gear_fact = (
            30100.0
            + 0.3876 * main_gear_length * main_gear_length
            + 0.09579 * nose_gear_length * nose_gear_length
        ) / 1.0e6
        total_gear_weight = gross_weight * total_gear_fact
        total_gear_weight_dmain = gross_weight * 7.752e-7 * main_gear_length
        total_gear_weight_dnose = gross_weight * 1.9158e-7 * nose_gear_length

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH] = (
            0.85 * total_gear_weight_dmain * main_gear_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH] = (
            0.85 * total_gear_weight_dnose * main_gear_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER] = (
            0.85 * total_gear_weight / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Mission.Design.GROSS_MASS] = (
            0.85 * total_gear_fact * main_gear_scaler
        )

        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH] = (
            0.15 * total_gear_weight_dmain * nose_gear_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH] = (
            0.15 * total_gear_weight_dnose * nose_gear_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER] = (
            0.15 * total_gear_weight / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Mission.Design.GROSS_MASS] = (
            0.15 * total_gear_fact * nose_gear_scaler
        )


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
