import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Mission


class ACMass(om.ExplicitComponent):
    """
    Computation of air conditioning mass.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, units='psi')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')

        self.add_output(Aircraft.AirConditioning.MASS, units='lbm')

        self.declare_partials(
            Aircraft.AirConditioning.MASS,
            [
                Aircraft.AirConditioning.MASS_COEFFICIENT,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
                Mission.Design.GROSS_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        ac_coeff = inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]

        # note: this technically creates a discontinuity but we will not smooth it.
        if gross_wt_initial > 3500.0:
            air_conditioning_wt = (
                ac_coeff * (1.5 + p_diff_fus) * (0.358 * fus_len * cabin_width**2) ** 0.5
            )
        else:
            air_conditioning_wt = 5.0

        outputs[Aircraft.AirConditioning.MASS] = air_conditioning_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        ac_coeff = inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]

        dac_wt_dgross_wt = 0.0

        if gross_wt_initial > 3500.0:
            dac_wt_dfus_len = (
                0.5
                * ac_coeff
                * (1.5 + p_diff_fus)
                * 0.358
                * cabin_width**2
                * (0.358 * fus_len * cabin_width**2) ** -0.5
            )
            dac_wt_dp_diff_fus = ac_coeff * (0.358 * fus_len * cabin_width**2) ** 0.5
            dac_wt_dcabin_width = (
                ac_coeff
                * (1.5 + p_diff_fus)
                * 0.358
                * fus_len
                * cabin_width
                * (0.358 * fus_len * cabin_width**2) ** -0.5
            )
            dac_wt_dac_coeff = (1.5 + p_diff_fus) * (0.358 * fus_len * cabin_width**2) ** 0.5
        else:
            dac_wt_dfus_len = 0.0
            dac_wt_dp_diff_fus = 0.0
            dac_wt_dcabin_width = 0.0
            dac_wt_dac_coeff = 0.0

        J[Aircraft.AirConditioning.MASS, Mission.Design.GROSS_MASS] = dac_wt_dgross_wt
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.LENGTH] = (
            dac_wt_dfus_len / GRAV_ENGLISH_LBM
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL] = (
            dac_wt_dp_diff_fus / GRAV_ENGLISH_LBM
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.AVG_DIAMETER] = (
            dac_wt_dcabin_width / GRAV_ENGLISH_LBM
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.AirConditioning.MASS_COEFFICIENT] = (
            dac_wt_dac_coeff / GRAV_ENGLISH_LBM
        )


class BWBACMass(om.ExplicitComponent):
    """
    Computation of air conditioning mass for BWB
    """

    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, units='psi')
        add_aviary_input(self, Aircraft.Fuselage.HYDRAULIC_DIAMETER, units='ft')

        self.add_output(Aircraft.AirConditioning.MASS, units='lbm')

        self.declare_partials(
            Aircraft.AirConditioning.MASS,
            [
                Aircraft.AirConditioning.MASS_COEFFICIENT,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.HYDRAULIC_DIAMETER,
                Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
                Mission.Design.GROSS_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        ac_coeff = inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]

        # note: this technically creates a discontinuity but we will not smooth it.
        if gross_wt_initial > 3500.0:
            air_conditioning_wt = (
                ac_coeff * (1.5 + p_diff_fus) * (0.358 * fus_len * cabin_width**2) ** 0.5
            )
        else:
            air_conditioning_wt = 5.0

        outputs[Aircraft.AirConditioning.MASS] = air_conditioning_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        ac_coeff = inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]

        dac_wt_dgross_wt = 0.0

        if gross_wt_initial > 3500.0:
            dac_wt_dfus_len = (
                0.5
                * ac_coeff
                * (1.5 + p_diff_fus)
                * 0.358
                * cabin_width**2
                * (0.358 * fus_len * cabin_width**2) ** -0.5
            )
            dac_wt_dp_diff_fus = ac_coeff * (0.358 * fus_len * cabin_width**2) ** 0.5
            dac_wt_dcabin_width = (
                ac_coeff
                * (1.5 + p_diff_fus)
                * 0.358
                * fus_len
                * cabin_width
                * (0.358 * fus_len * cabin_width**2) ** -0.5
            )
            dac_wt_dac_coeff = (1.5 + p_diff_fus) * (0.358 * fus_len * cabin_width**2) ** 0.5
        else:
            dac_wt_dfus_len = 0.0
            dac_wt_dp_diff_fus = 0.0
            dac_wt_dcabin_width = 0.0
            dac_wt_dac_coeff = 0.0

        J[Aircraft.AirConditioning.MASS, Mission.Design.GROSS_MASS] = dac_wt_dgross_wt
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.LENGTH] = (
            dac_wt_dfus_len / GRAV_ENGLISH_LBM
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL] = (
            dac_wt_dp_diff_fus / GRAV_ENGLISH_LBM
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.Fuselage.HYDRAULIC_DIAMETER] = (
            dac_wt_dcabin_width / GRAV_ENGLISH_LBM
        )
        J[Aircraft.AirConditioning.MASS, Aircraft.AirConditioning.MASS_COEFFICIENT] = (
            dac_wt_dac_coeff / GRAV_ENGLISH_LBM
        )
