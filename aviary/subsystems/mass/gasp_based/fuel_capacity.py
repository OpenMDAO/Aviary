import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.math import dSigmoidXdx, sigmoidX
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TrappedFuelCapacity(om.ExplicitComponent):
    """Compute the maximum fuel that can be carried in the fuselage."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        self.options.declare('x0', default=0.075, types=float)
        self.options.declare('mu', default=0.01, types=float)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless')

        add_aviary_output(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        mu = self.options['mu']
        x0 = self.options['x0']
        wing_area = inputs[Aircraft.Wing.AREA]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        unusable_fuel_coeff = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]

        if wing_area <= 0.0:
            raise ValueError(
                f'Aircraft.Wing.AREA must be positive, however {wing_area} is provided.'
            )

        trapped_fuel_1_wt = unusable_fuel_coeff * (wing_area**0.5) * fuel_vol_frac / 0.430
        trapped_fuel_2_wt = unusable_fuel_coeff * 0.18 * (wing_area**0.5)

        import pdb

        # pdb.set_trace()
        if smooth:
            smoother = sigmoidX(fuel_vol_frac, x0, mu)
            trapped_fuel_wt = smoother * trapped_fuel_1_wt + (1 - smoother) * trapped_fuel_2_wt
        else:
            if fuel_vol_frac > 0.075:
                trapped_fuel_wt = trapped_fuel_1_wt
            else:
                trapped_fuel_wt = trapped_fuel_2_wt

        outputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS] = trapped_fuel_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        mu = self.options['mu']
        x0 = self.options['x0']
        wing_area = inputs[Aircraft.Wing.AREA]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        unusable_fuel_coeff = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]

        if smooth:
            smoother = sigmoidX(fuel_vol_frac, x0, mu)
            dsmoother_dfuel_vol_frac = dSigmoidXdx(fuel_vol_frac, x0, mu)
            trapped_fuel_1_wt = unusable_fuel_coeff * (wing_area**0.5) * fuel_vol_frac / 0.430
            trapped_fuel_2_wt = unusable_fuel_coeff * 0.18 * (wing_area**0.5)
            dtrapped_fuel_wt_dmass_coeff_1 = (wing_area**0.5) * fuel_vol_frac / 0.430
            dtrapped_fuel_wt_dwing_area_1 = (
                0.5 * unusable_fuel_coeff * (wing_area**-0.5) * fuel_vol_frac / 0.430
            )
            dtrapped_fuel_wt_dfuel_vol_frac_1 = unusable_fuel_coeff * (wing_area**0.5) / 0.430
            dtrapped_fuel_wt_dmass_coeff_2 = 0.18 * (wing_area**0.5)
            dtrapped_fuel_wt_dwing_area_2 = 0.5 * unusable_fuel_coeff * 0.18 * (wing_area**-0.5)
            dtrapped_fuel_wt_dfuel_vol_frac_2 = 0.0
            dtrapped_fuel_wt_dmass_coeff = (
                smoother * dtrapped_fuel_wt_dmass_coeff_1
                + (1 - smoother) * dtrapped_fuel_wt_dmass_coeff_2
            )
            dtrapped_fuel_wt_dwing_area = (
                smoother * dtrapped_fuel_wt_dwing_area_1
                + (1 - smoother) * dtrapped_fuel_wt_dwing_area_2
            )
            dtrapped_fuel_wt_dfuel_vol_frac = (
                dsmoother_dfuel_vol_frac * trapped_fuel_1_wt
                + smoother * dtrapped_fuel_wt_dfuel_vol_frac_1
                - dsmoother_dfuel_vol_frac * trapped_fuel_2_wt
                + (1 - smoother) * dtrapped_fuel_wt_dfuel_vol_frac_2
            )
        else:
            if fuel_vol_frac > 0.075:
                dtrapped_fuel_wt_dmass_coeff = (wing_area**0.5) * fuel_vol_frac / 0.430
                dtrapped_fuel_wt_dwing_area = (
                    0.5 * unusable_fuel_coeff * (wing_area**-0.5) * fuel_vol_frac / 0.430
                )
                dtrapped_fuel_wt_dfuel_vol_frac = unusable_fuel_coeff * (wing_area**0.5) / 0.430
            else:
                dtrapped_fuel_wt_dmass_coeff = 0.18 * (wing_area**0.5)
                dtrapped_fuel_wt_dwing_area = 0.5 * unusable_fuel_coeff * 0.18 * (wing_area**-0.5)
                dtrapped_fuel_wt_dfuel_vol_frac = 0.0

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Wing.AREA] = (
            dtrapped_fuel_wt_dwing_area / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.WING_FUEL_FRACTION] = (
            dtrapped_fuel_wt_dfuel_vol_frac
        ) / GRAV_ENGLISH_LBM
        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] = (
            dtrapped_fuel_wt_dmass_coeff
        ) / GRAV_ENGLISH_LBM
