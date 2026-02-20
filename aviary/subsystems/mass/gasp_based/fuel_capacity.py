import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft

from aviary.constants import GRAV_ENGLISH_LBM


class TrappedFuelCapacity(om.ExplicitComponent):
    """Compute the maximum fuel that can be carried in the fuselage."""

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless')

        add_aviary_output(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        wing_area = inputs[Aircraft.Wing.AREA]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        unusable_fuel_coeff = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]

        trapped_fuel_wt = unusable_fuel_coeff * (wing_area**0.5) * fuel_vol_frac / 0.430

        if fuel_vol_frac <= 0.075:  # note: this technically creates a discontinuity # won't change
            trapped_fuel_wt = unusable_fuel_coeff * 0.18 * (wing_area**0.5)

        outputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS] = trapped_fuel_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        wing_area = inputs[Aircraft.Wing.AREA]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        unusable_fuel_coeff = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]

        dtrapped_fuel_wt_dmass_coeff_12 = (wing_area**0.5) * fuel_vol_frac / 0.430
        dtrapped_fuel_wt_dwing_area = (
            0.5 * unusable_fuel_coeff * (wing_area**-0.5) * fuel_vol_frac / 0.430
        )
        dtrapped_fuel_wt_dfuel_vol_frac = unusable_fuel_coeff * (wing_area**0.5) / 0.430

        if fuel_vol_frac <= 0.075:  # note: this technically creates a discontinuity # won't change
            dtrapped_fuel_wt_dmass_coeff_12 = 0.18 * (wing_area**0.5)
            dtrapped_fuel_wt_dwing_area = 0.5 * unusable_fuel_coeff * 0.18 * (wing_area**-0.5)
            dtrapped_fuel_wt_dfuel_vol_frac = 0.0

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Wing.AREA] = (
            dtrapped_fuel_wt_dwing_area / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.WING_FUEL_FRACTION] = (
            dtrapped_fuel_wt_dfuel_vol_frac
        ) / GRAV_ENGLISH_LBM
        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] = (
            dtrapped_fuel_wt_dmass_coeff_12
        ) / GRAV_ENGLISH_LBM
