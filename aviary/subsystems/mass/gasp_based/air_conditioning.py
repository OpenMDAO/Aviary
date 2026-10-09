import openmdao.api as om

from aviary.utils.math_utils import dSigmoidXdx, sigmoidX
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


def common_compute(smooth, mu, x0, gross_mass_initial, fus_len, p_diff_fus, cabin_width, ac_coeff):
    if smooth:
        smoother = sigmoidX(gross_mass_initial, x0, mu)
        # gross_mass_initial > 3500.0:
        air_cond1_mass = ac_coeff * (1.5 + p_diff_fus) * (0.358 * fus_len * cabin_width**2) ** 0.5
        # gross_mass_initial <= 3500.0:
        air_cond2_mass = 5.0
        air_conditioning_mass = smoother * air_cond1_mass + (1 - smoother) * air_cond2_mass
    else:
        if gross_mass_initial > 3500.0:
            air_conditioning_mass = (
                ac_coeff * (1.5 + p_diff_fus) * (0.358 * fus_len * cabin_width**2) ** 0.5
            )
        else:
            air_conditioning_mass = 5.0

    return air_conditioning_mass


def common_conpute_partials(
    smooth, mu, x0, gross_mass_initial, fus_len, p_diff_fus, cabin_width, ac_coeff
):
    if smooth:
        air_cond1_mass = ac_coeff * (1.5 + p_diff_fus) * (0.358 * fus_len * cabin_width**2) ** 0.5
        air_cond2_mass = 5.0

        dac_mass_dgross_mass = (
            dSigmoidXdx(gross_mass_initial, x0, mu) * air_cond1_mass
            - dSigmoidXdx(gross_mass_initial, x0, mu) * air_cond2_mass
        )
    else:
        dac_mass_dgross_mass = 0.0

    # case gross_mass_initial > 3500.0:
    dac_mass_dfus_1_len = (
        0.5
        * ac_coeff
        * (1.5 + p_diff_fus)
        * 0.358
        * cabin_width**2
        * (0.358 * fus_len * cabin_width**2) ** -0.5
    )
    dac_mass_dp_diff_1_fus = ac_coeff * (0.358 * fus_len * cabin_width**2) ** 0.5
    dac_mass_dcabin_1_width = (
        ac_coeff
        * (1.5 + p_diff_fus)
        * 0.358
        * fus_len
        * cabin_width
        * (0.358 * fus_len * cabin_width**2) ** -0.5
    )
    dac_mass_dac_1_coeff = (1.5 + p_diff_fus) * (0.358 * fus_len * cabin_width**2) ** 0.5

    # case gross_mass_initial <= 3500.0:
    dac_mass_dfus_2_len = 0.0
    dac_mass_dp_diff_2_fus = 0.0
    dac_mass_dcabin_2_width = 0.0
    dac_mass_dac_2_coeff = 0.0

    if smooth:
        dac_mass_dfus_len = (
            sigmoidX(gross_mass_initial, x0, mu) * dac_mass_dfus_1_len
            + (1 - sigmoidX(gross_mass_initial, x0, mu)) * dac_mass_dfus_2_len
        )
    else:
        if gross_mass_initial > 3500.0:
            dac_mass_dfus_len = dac_mass_dfus_1_len
        else:
            dac_mass_dfus_len = dac_mass_dfus_2_len

    if smooth:
        dac_mass_dp_diff_fus = (
            sigmoidX(gross_mass_initial, x0, mu) * dac_mass_dp_diff_1_fus
            + (1 - sigmoidX(gross_mass_initial, x0, mu)) * dac_mass_dp_diff_2_fus
        )
    else:
        if gross_mass_initial > 3500.0:
            dac_mass_dp_diff_fus = dac_mass_dp_diff_1_fus
        else:
            dac_mass_dp_diff_fus = dac_mass_dp_diff_2_fus

    if smooth:
        dac_mass_dcabin_width = (
            sigmoidX(gross_mass_initial, x0, mu) * dac_mass_dcabin_1_width
            + (1 - sigmoidX(gross_mass_initial, x0, mu)) * dac_mass_dcabin_2_width
        )
    else:
        if gross_mass_initial > 3500.0:
            dac_mass_dcabin_width = dac_mass_dcabin_1_width
        else:
            dac_mass_dcabin_width = dac_mass_dcabin_2_width

    if smooth:
        dac_mass_dac_coeff = (
            sigmoidX(gross_mass_initial, x0, mu) * dac_mass_dac_1_coeff
            + (1 - sigmoidX(gross_mass_initial, x0, mu)) * dac_mass_dac_2_coeff
        )
    else:
        if gross_mass_initial > 3500.0:
            dac_mass_dac_coeff = dac_mass_dac_1_coeff
        else:
            dac_mass_dac_coeff = dac_mass_dac_2_coeff

    return [
        dac_mass_dgross_mass,
        dac_mass_dfus_len,
        dac_mass_dp_diff_fus,
        dac_mass_dcabin_width,
        dac_mass_dac_coeff,
    ]


class ACMass(om.ExplicitComponent):
    """Computation of air conditioning mass."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        self.options.declare('x0', default=3500.0, types=float)
        self.options.declare('mu', default=0.01, types=float)

    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, units='psi')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')

        add_aviary_output(self, Aircraft.AirConditioning.MASS, units='lbm')

        self.declare_partials(
            Aircraft.AirConditioning.MASS,
            [
                Aircraft.AirConditioning.MASS_COEFFICIENT,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
                Aircraft.Design.GROSS_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        mu = self.options['mu']
        x0 = self.options['x0']
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        ac_coeff = inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]

        air_conditioning_mass = common_compute(
            smooth, mu, x0, gross_mass_initial, fus_len, p_diff_fus, cabin_width, ac_coeff
        )
        outputs[Aircraft.AirConditioning.MASS] = air_conditioning_mass

    def compute_partials(self, inputs, J):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        mu = self.options['mu']
        x0 = self.options['x0']
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        ac_coeff = inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]

        [
            dac_mass_dgross_mass,
            dac_mass_dfus_len,
            dac_mass_dp_diff_fus,
            dac_mass_dcabin_width,
            dac_mass_dac_coeff,
        ] = common_conpute_partials(
            smooth, mu, x0, gross_mass_initial, fus_len, p_diff_fus, cabin_width, ac_coeff
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.Design.GROSS_MASS] = dac_mass_dgross_mass
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.LENGTH] = dac_mass_dfus_len
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL] = (
            dac_mass_dp_diff_fus
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.AVG_DIAMETER] = dac_mass_dcabin_width
        J[Aircraft.AirConditioning.MASS, Aircraft.AirConditioning.MASS_COEFFICIENT] = (
            dac_mass_dac_coeff
        )


class BWBACMass(om.ExplicitComponent):
    """Computation of air conditioning mass for BWB."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        self.options.declare('x0', default=3500.0, types=float)
        self.options.declare('mu', default=0.01, types=float)

    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, units='psi')
        add_aviary_input(self, Aircraft.Fuselage.HYDRAULIC_DIAMETER, units='ft')

        add_aviary_output(self, Aircraft.AirConditioning.MASS, units='lbm')

        self.declare_partials(
            Aircraft.AirConditioning.MASS,
            [
                Aircraft.AirConditioning.MASS_COEFFICIENT,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.HYDRAULIC_DIAMETER,
                Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
                Aircraft.Design.GROSS_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        mu = self.options['mu']
        x0 = self.options['x0']
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        ac_coeff = inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]

        air_conditioning_mass = common_compute(
            smooth, mu, x0, gross_mass_initial, fus_len, p_diff_fus, cabin_width, ac_coeff
        )

        outputs[Aircraft.AirConditioning.MASS] = air_conditioning_mass

    def compute_partials(self, inputs, J):
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        mu = self.options['mu']
        x0 = self.options['x0']
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        ac_coeff = inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]

        [
            dac_mass_dgross_mass,
            dac_mass_dfus_len,
            dac_mass_dp_diff_fus,
            dac_mass_dcabin_width,
            dac_mass_dac_coeff,
        ] = common_conpute_partials(
            smooth, mu, x0, gross_mass_initial, fus_len, p_diff_fus, cabin_width, ac_coeff
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.Design.GROSS_MASS] = dac_mass_dgross_mass
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.LENGTH] = dac_mass_dfus_len
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL] = (
            dac_mass_dp_diff_fus
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.HYDRAULIC_DIAMETER] = (
            dac_mass_dcabin_width
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.AirConditioning.MASS_COEFFICIENT] = (
            dac_mass_dac_coeff
        )
