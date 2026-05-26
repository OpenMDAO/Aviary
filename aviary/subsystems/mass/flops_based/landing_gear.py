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


class LandingGearMass(om.Group):
    def initialize(self):
        add_aviary_option(self, Aircraft.Design.USE_ALT_MASS)

    def setup(self):
        if self.options[Aircraft.Design.USE_ALT_MASS]:
            self.add_subsystem(
                'alternate_mass',
                AltLandingGearMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        else:
            self.add_subsystem(
                'main_gear', MainGearMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )
            self.add_subsystem(
                'nose_gear', NoseGearMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        self.add_subsystem(
            'total_mass', LandingGearTotalMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )


class LandingGearTotalMass(om.ExplicitComponent):
    """Sum the total mass of the landing gear."""

    def setup(self):
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_MASS, units='lbm')

        add_aviary_output(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.LandingGear.TOTAL_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        main_gear_mass = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS]
        nose_gear_mass = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS]

        outputs[Aircraft.LandingGear.TOTAL_MASS] = main_gear_mass + nose_gear_mass


class MainGearMass(om.ExplicitComponent):
    """
    Calculate the mass of the main landing gear. The methodology is based on the FLOPS weight
    equations, modified to output mass instead of weight.
    """

    # add in aircraft type and carrier factors as options and modify equations, see issue #1094.

    def setup(self):
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Design.TOUCHDOWN_MASS_MAX, units='lbm')

        add_aviary_output(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.MAIN_GEAR_MASS,
            [
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
                Aircraft.Design.TOUCHDOWN_MASS_MAX,
            ],
        )

    def compute(self, inputs, outputs):
        main_gear_length = inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        main_gear_scaler = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER]
        landing_weight = inputs[Aircraft.Design.TOUCHDOWN_MASS_MAX] * GRAV_ENGLISH_LBM

        main_gear_mass = (
            0.0117
            * landing_weight**0.95
            * main_gear_length**0.43
            * main_gear_scaler
            / GRAV_ENGLISH_LBM
        )

        outputs[Aircraft.LandingGear.MAIN_GEAR_MASS] = main_gear_mass

    def compute_partials(self, inputs, J):
        main_gear_length = inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        main_gear_scaler = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER]
        landing_weight = inputs[Aircraft.Design.TOUCHDOWN_MASS_MAX] * GRAV_ENGLISH_LBM

        landing_weight_exp = landing_weight**0.95
        main_gear_length_exp = main_gear_length**0.43

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH] = (
            0.005031
            * landing_weight_exp
            * main_gear_length**-0.57
            * main_gear_scaler
            / GRAV_ENGLISH_LBM
        )
        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER] = (
            0.0117 * landing_weight_exp * main_gear_length_exp / GRAV_ENGLISH_LBM
        )
        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.Design.TOUCHDOWN_MASS_MAX] = (
            0.011115 * landing_weight**-0.05 * main_gear_length_exp * main_gear_scaler
        )


class NoseGearMass(om.ExplicitComponent):
    """
    Calculate the mass of the nose landing gear. The methodology is based on the FLOPS weight
    equations, modified to output mass instead of weight.
    """

    # add in aircraft type and carrier factors as options and modify equations, see issue #1094.

    def setup(self):
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Design.TOUCHDOWN_MASS_MAX, units='lbm')

        add_aviary_output(self, Aircraft.LandingGear.NOSE_GEAR_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.NOSE_GEAR_MASS,
            [
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
                Aircraft.Design.TOUCHDOWN_MASS_MAX,
            ],
        )

    def compute(self, inputs, outputs):
        nose_gear_length = inputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH]
        nose_gear_scaler = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER]
        landing_weight = inputs[Aircraft.Design.TOUCHDOWN_MASS_MAX] * GRAV_ENGLISH_LBM

        nose_gear_mass = (
            0.048
            * landing_weight**0.67
            * nose_gear_length**0.43
            * nose_gear_scaler
            / GRAV_ENGLISH_LBM
        )

        outputs[Aircraft.LandingGear.NOSE_GEAR_MASS] = nose_gear_mass

    def compute_partials(self, inputs, J):
        nose_gear_length = inputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH]
        nose_gear_scaler = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER]
        landing_weight = inputs[Aircraft.Design.TOUCHDOWN_MASS_MAX] * GRAV_ENGLISH_LBM

        landing_weight_exp = landing_weight**0.67
        nose_gear_length_exp = nose_gear_length**0.43

        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH] = (
            0.02064
            * landing_weight_exp
            * nose_gear_length**-0.57
            * nose_gear_scaler
            / GRAV_ENGLISH_LBM
        )
        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER] = (
            0.048 * landing_weight_exp * nose_gear_length_exp / GRAV_ENGLISH_LBM
        )
        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.Design.TOUCHDOWN_MASS_MAX] = (
            0.03216 * landing_weight**-0.33 * nose_gear_length_exp * nose_gear_scaler
        )


class AltLandingGearMass(om.ExplicitComponent):
    """
    Calculate the mass of the main and nose landing gears using the alternate method. The
    methodology is based on the alternate FLOPS weight equations, modified to output mass instead of
    weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, units='inch')
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')

        add_aviary_output(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')
        add_aviary_output(self, Aircraft.LandingGear.NOSE_GEAR_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.MAIN_GEAR_MASS,
            [
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
                Aircraft.Design.GROSS_MASS,
            ],
        )
        self.declare_partials(
            Aircraft.LandingGear.NOSE_GEAR_MASS,
            [
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
                Aircraft.Design.GROSS_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        main_gear_length = inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        main_gear_scaler = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER]
        nose_gear_length = inputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH]
        nose_gear_scaler = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER]
        gross_weight = inputs[Aircraft.Design.GROSS_MASS] * GRAV_ENGLISH_LBM

        unscaled_total_gear_mass = gross_weight * (
            (
                30100.0
                + 0.3876 * main_gear_length * main_gear_length
                + 0.09579 * nose_gear_length * nose_gear_length
            )
            / 1.0e6
        )

        main_gear_mass = (0.85 * unscaled_total_gear_mass * main_gear_scaler) / GRAV_ENGLISH_LBM
        nose_gear_mass = (0.15 * unscaled_total_gear_mass * nose_gear_scaler) / GRAV_ENGLISH_LBM

        outputs[Aircraft.LandingGear.MAIN_GEAR_MASS] = main_gear_mass
        outputs[Aircraft.LandingGear.NOSE_GEAR_MASS] = nose_gear_mass

    def compute_partials(self, inputs, J):
        main_gear_length = inputs[Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH]
        main_gear_scaler = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER]
        nose_gear_length = inputs[Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH]
        nose_gear_scaler = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER]
        gross_weight = inputs[Aircraft.Design.GROSS_MASS] * GRAV_ENGLISH_LBM

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

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.Design.GROSS_MASS] = (
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

        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.Design.GROSS_MASS] = (
            0.15 * total_gear_fact * nose_gear_scaler
        )
