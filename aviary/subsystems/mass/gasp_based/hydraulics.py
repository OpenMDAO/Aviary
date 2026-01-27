import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class HydraulicsMass(om.ExplicitComponent):
    """
    Calculates the mass of the hydraulics group using the transport/general aviation
    method. The methodology is based on the GASP weight equations, modified to
    output mass instead of weight.

    ASSUMPTIONS: all engines use hydraulics which follow this equation
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.LandingGear.FIXED_GEAR)

    def setup(self):
        add_aviary_input(
            self, Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, units='unitless'
        )
        add_aviary_input(self, Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Controls.TOTAL_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Hydraulics.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        landing_gear_fixed = self.options[Aircraft.LandingGear.FIXED_GEAR]

        flight_control_mass_coeff = inputs[Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT]
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        landing_gear_wt = inputs[Aircraft.LandingGear.TOTAL_MASS] * GRAV_ENGLISH_LBM
        gear_mass_coeff = inputs[Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT]

        outputs[Aircraft.Hydraulics.MASS] = (
            flight_control_mass_coeff * control_wt
            + gear_mass_coeff * landing_gear_wt * (not landing_gear_fixed)
        )

    def compute_partials(self, inputs, J):
        landing_gear_wt = inputs[Aircraft.LandingGear.TOTAL_MASS] * GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        flight_control_mass_coeff = inputs[Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT]
        gear_mass_coeff = inputs[Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT]

        gear_val = not self.options[Aircraft.LandingGear.FIXED_GEAR]

        J[Aircraft.Hydraulics.MASS, Aircraft.LandingGear.TOTAL_MASS] = gear_mass_coeff * gear_val
        J[Aircraft.Hydraulics.MASS, Aircraft.Controls.TOTAL_MASS] = flight_control_mass_coeff
        J[Aircraft.Hydraulics.MASS, Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT] = (
            control_wt
        )
        J[Aircraft.Hydraulics.MASS, Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT] = (
            landing_gear_wt * gear_val
        )
