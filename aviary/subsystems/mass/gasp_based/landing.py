import numpy as np
import openmdao.api as om
from openmdao.components.ks_comp import KSfunction

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.math import dSigmoidXdx, sigmoidX
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class LandingMass(om.ExplicitComponent):
    """Maximum landing mass is maximum takeoff gross mass times the ratio of landing/takeoff mass."""

    def setup(self):
        add_aviary_input(self, Aircraft.Design.GROSS_MASS)
        add_aviary_input(self, Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO)

        add_aviary_output(self, Aircraft.Design.TOUCHDOWN_MASS_MAX)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        gross_mass = inputs[Aircraft.Design.GROSS_MASS]
        landing_to_takeoff_mass_ratio = inputs[Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO]

        outputs[Aircraft.Design.TOUCHDOWN_MASS_MAX] = gross_mass * landing_to_takeoff_mass_ratio

    def compute_partials(self, inputs, J):
        gross_mass = inputs[Aircraft.Design.GROSS_MASS]
        landing_to_takeoff_mass_ratio = inputs[Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO]

        J[Aircraft.Design.TOUCHDOWN_MASS_MAX, Aircraft.Design.GROSS_MASS] = (
            landing_to_takeoff_mass_ratio
        )

        J[Aircraft.Design.TOUCHDOWN_MASS_MAX, Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO] = (
            gross_mass
        )


class TotalLandingGearMass(om.ExplicitComponent):
    """Computation of total mass of landing gear."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Wing.VERTICAL_MOUNT_LOCATION, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Design.TOUCHDOWN_MASS_MAX, units='lbm')
        add_aviary_input(
            self,
            Aircraft.Nacelle.CLEARANCE_RATIO,
            shape=num_engine_type,
            units='unitless',
        )
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS_SCALER)

        add_aviary_output(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.TOTAL_MASS,
            [
                Aircraft.LandingGear.MASS_COEFFICIENT,
                Aircraft.Design.TOUCHDOWN_MASS_MAX,
                Aircraft.Nacelle.CLEARANCE_RATIO,
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.LandingGear.TOTAL_MASS_SCALER,
            ],
        )

    def compute(self, inputs, outputs):
        wing_loc = inputs[Aircraft.Wing.VERTICAL_MOUNT_LOCATION]
        c_gear_mass = inputs[Aircraft.LandingGear.MASS_COEFFICIENT]
        landing_wt = inputs[Aircraft.Design.TOUCHDOWN_MASS_MAX] * GRAV_ENGLISH_LBM
        clearance_ratio = inputs[Aircraft.Nacelle.CLEARANCE_RATIO]
        nacelle_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        CK12 = inputs[Aircraft.LandingGear.TOTAL_MASS_SCALER]

        # When there are multiple engine types, use the largest required clearance
        # TODO this does not match variable description (e.g. clearance ratio of 1.0 is
        #      actually two nacelle diameters above ground)
        # Note: KSFunction for smooth derivatives.
        gear_height_temp = KSfunction.compute((1.0 + clearance_ratio) * nacelle_diam, 50.0)

        # A minimum gear height of 6 feet is enforced here using a smoothing function to
        # prevent discontinuities in the function and it's derivatives.
        gear_height_temp = gear_height_temp[0]
        gear_height = gear_height_temp * sigmoidX(gear_height_temp, 6, 1 / 320.0) + 6 * sigmoidX(
            gear_height_temp, 6, -1 / 320.0
        )

        # Low wing aircraft (defined as having the wing at the lowest position on the
        # fuselage) have a separate equation for calculating gear mass. A smoothing
        # function centered at a wing height of .5% smooths the equations between 0 and
        # 1%. The equations should produce no noticeable difference between the stepwise
        # versions at 0% and at or above 1%.
        c_gear_mass_modified = (c_gear_mass * 0.85 * (1.0 + 0.1765 * gear_height / 6.0)) * sigmoidX(
            wing_loc, 0.005, -0.01 / 320
        ) + c_gear_mass * sigmoidX(wing_loc, 0.005, 0.01 / 320)

        landing_gear_wt = c_gear_mass_modified * landing_wt

        outputs[Aircraft.LandingGear.TOTAL_MASS] = CK12 * landing_gear_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        c_gear_mass = inputs[Aircraft.LandingGear.MASS_COEFFICIENT]
        landing_wt = inputs[Aircraft.Design.TOUCHDOWN_MASS_MAX] * GRAV_ENGLISH_LBM
        wing_loc = inputs[Aircraft.Wing.VERTICAL_MOUNT_LOCATION]
        clearance_ratio = inputs[Aircraft.Nacelle.CLEARANCE_RATIO]
        nacelle_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]

        val = (1.0 + clearance_ratio) * nacelle_diam
        gear_height_temp = KSfunction.compute(val, 50.0)
        dKS, _ = KSfunction.derivatives(val, 50.0)

        gear_height_temp = gear_height_temp[0]
        gear_height = gear_height_temp * sigmoidX(gear_height_temp, 6, 1 / 320.0) + 6 * sigmoidX(
            gear_height_temp, 6, -1 / 320.0
        )

        dLGW_dCGW = (
            (0.85 * (1.0 + 0.1765 * gear_height / 6.0)) * sigmoidX(wing_loc, 0.005, -0.01 / 320.0)
            + sigmoidX(wing_loc, 0.005, 0.01 / 320.0)
        ) * landing_wt

        dGH_dCR = (
            (
                sigmoidX(gear_height_temp, 6, 1 / 320.0) * nacelle_diam
                + gear_height_temp * dSigmoidXdx(gear_height_temp, 6, 1 / 320.0) * nacelle_diam
            )
            + (6 * dSigmoidXdx(gear_height_temp, 6, 1 / 320.0) * -nacelle_diam)
        ) * dKS

        dGH_dND = (
            sigmoidX(gear_height_temp, 6, 1 / 320.0) * (1 + clearance_ratio)
            + gear_height_temp * dSigmoidXdx(gear_height_temp, 6, 1 / 320.0) * (1 + clearance_ratio)
        ) * dKS + (6 * dSigmoidXdx(gear_height_temp, 6, 1 / 320.0) * (1 + clearance_ratio))

        c_gear_mass_modified = (c_gear_mass * 0.85 * (1.0 + 0.1765 * gear_height / 6.0)) * sigmoidX(
            wing_loc, 0.005, -0.01 / 320.0
        ) + c_gear_mass * sigmoidX(wing_loc, 0.005, 0.01 / 320.0)

        dLGW_dCR = max(
            (c_gear_mass * 0.85 * 0.1765 / 6 * dGH_dCR * landing_wt)
            * sigmoidX(wing_loc, 0.005, -0.01 / 320.0)
        )

        dLGW_dND = max(
            (c_gear_mass * 0.85 * 0.1765 / 6 * dGH_dND * landing_wt)
            * sigmoidX(wing_loc, 0.005, -0.01 / 320.0)
        )

        J[Aircraft.LandingGear.TOTAL_MASS, Aircraft.LandingGear.MASS_COEFFICIENT] = (
            dLGW_dCGW / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.TOTAL_MASS, Aircraft.Nacelle.CLEARANCE_RATIO] = (
            dLGW_dCR / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.TOTAL_MASS, Aircraft.Nacelle.AVG_DIAMETER] = (
            dLGW_dND / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.TOTAL_MASS, Aircraft.Design.TOUCHDOWN_MASS_MAX] = (
            c_gear_mass_modified
        )

        J[Aircraft.LandingGear.TOTAL_MASS, Aircraft.LandingGear.TOTAL_MASS_SCALER] = (
            c_gear_mass_modified * landing_wt
        ) / GRAV_ENGLISH_LBM


class LandingGearMass(om.ExplicitComponent):
    """Computation main and nose landing gear mass."""

    def setup(self):
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')
        add_aviary_output(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')
        add_aviary_output(self, Aircraft.LandingGear.NOSE_GEAR_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.MAIN_GEAR_MASS,
            [
                Aircraft.LandingGear.TOTAL_MASS,
                Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION,
            ],
        )
        self.declare_partials(
            Aircraft.LandingGear.NOSE_GEAR_MASS,
            [
                Aircraft.LandingGear.TOTAL_MASS,
                Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION,
            ],
        )

    def compute(self, inputs, outputs):
        c_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION]
        landing_gear_mass = inputs[Aircraft.LandingGear.TOTAL_MASS]

        outputs[Aircraft.LandingGear.MAIN_GEAR_MASS] = c_main_gear * landing_gear_mass
        outputs[Aircraft.LandingGear.NOSE_GEAR_MASS] = (1 - c_main_gear) * landing_gear_mass

    def compute_partials(self, inputs, J):
        c_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION]
        landing_gear_mass = inputs[Aircraft.LandingGear.TOTAL_MASS]

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.TOTAL_MASS] = c_main_gear
        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION] = (
            landing_gear_mass
        )

        J[Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.TOTAL_MASS] = 1 - c_main_gear
        J[
            Aircraft.LandingGear.NOSE_GEAR_MASS, Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION
        ] = -landing_gear_mass


class LandingGearMassGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'landing_mass', LandingMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )
        self.add_subsystem(
            'total_landing_gear',
            TotalLandingGearMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'landing_gear', LandingGearMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )
