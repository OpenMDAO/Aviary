import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class FuselageMass(om.ExplicitComponent):
    """Computation of the fuselage structure mass."""

    def setup(self):
        self.add_input(
            'fus_mass_full',
            units='lbm',
            desc='WX: mass of fuselage and contents, including empennage',
        )
        add_aviary_input(self, Aircraft.Fuselage.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.TailBoom.LENGTH, units='ft')
        # pylon_len is not computed in Aviary
        self.add_input(
            'pylon_len',
            val=0,
            units='ft',
            desc='ELRW: length of pylon for fuselage mounted engines',
        )
        self.add_input('min_dive_vel', units='kn', desc='VDMIN: dive velocity')
        add_aviary_input(self, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, units='psi')
        add_aviary_input(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')
        # MAT is not computed in Aviary
        self.add_input(
            'MAT', val=0, units='lbm', desc='WAT: Weight of the Fuselage Acoustic Treatment'
        )

        add_aviary_output(self, Aircraft.Fuselage.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.MASS,
            [
                Aircraft.Fuselage.MASS_SCALER,
                Aircraft.Fuselage.MASS_COEFFICIENT,
                'fus_mass_full',
                Aircraft.Fuselage.WETTED_AREA,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.TailBoom.LENGTH,
                'pylon_len',
                'min_dive_vel',
                Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
                Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                'MAT',
            ],
        )

    def compute(self, inputs, outputs):
        CK11 = inputs[Aircraft.Fuselage.MASS_SCALER]
        fus_mass_full = inputs['fus_mass_full']
        c_fuselage = inputs[Aircraft.Fuselage.MASS_COEFFICIENT]
        fus_SA = inputs[Aircraft.Fuselage.WETTED_AREA]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        cabin_len_tailboom = inputs[Aircraft.TailBoom.LENGTH]
        pylon_len = inputs['pylon_len']
        min_dive_vel = inputs['min_dive_vel']
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        WAT = inputs['MAT']

        fus_mass = CK11 * (
            c_fuselage
            * (
                (fus_mass_full / 10000.0) ** 0.7
                * (fus_SA / 1000.0)
                * cabin_width
                * (cabin_len_tailboom + pylon_len) ** 0.5
                * np.log10(min_dive_vel)
                * (p_diff_fus + 1.0) ** 0.2
                * ULF**0.3
            )
            ** 0.508
            + WAT
        )

        outputs[Aircraft.Fuselage.MASS] = fus_mass

    def compute_partials(self, inputs, J):
        CK11 = inputs[Aircraft.Fuselage.MASS_SCALER]
        fus_mass_full = inputs['fus_mass_full']
        c_fuselage = inputs[Aircraft.Fuselage.MASS_COEFFICIENT]
        fus_SA = inputs[Aircraft.Fuselage.WETTED_AREA]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        cabin_len_tailboom = inputs[Aircraft.TailBoom.LENGTH]
        pylon_len = inputs['pylon_len']
        min_dive_vel = inputs['min_dive_vel']
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        WAT = inputs['MAT']

        fus_mass = CK11 * (
            c_fuselage
            * (
                (fus_mass_full / 10000.0) ** 0.7
                * (fus_SA / 1000.0)
                * cabin_width
                * (cabin_len_tailboom + pylon_len) ** 0.5
                * np.log10(min_dive_vel)
                * (p_diff_fus + 1.0) ** 0.2
                * ULF**0.3
            )
            ** 0.508
            + WAT
        )

        x = (
            CK11
            * 0.508
            * (
                (fus_mass_full / 10000.0) ** 0.7
                * (fus_SA / 1000.0)
                * cabin_width
                * (cabin_len_tailboom + pylon_len) ** 0.5
                * np.log10(min_dive_vel)
                * (p_diff_fus + 1.0) ** 0.2
                * ULF**0.3
            )
            ** (0.508 - 1)
        )
        dFusWt_dCFus = (
            CK11
            * (
                (fus_mass_full / 10000.0) ** 0.7
                * (fus_SA / 1000.0)
                * cabin_width
                * (cabin_len_tailboom + pylon_len) ** 0.5
                * np.log10(min_dive_vel)
                * (p_diff_fus + 1.0) ** 0.2
                * ULF**0.3
            )
            ** 0.508
        )
        dFusWt_dFusWtFull = (
            c_fuselage
            * x
            * 0.7
            * (fus_mass_full / 10000.0) ** (0.7 - 1)
            * (1 / 10000)
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dFusSA = (
            c_fuselage
            * x
            * (fus_mass_full / 10000.0) ** 0.7
            * (1 / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dCabWidth = (
            c_fuselage
            * x
            * (fus_mass_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dCabLenBoom = (
            c_fuselage
            * x
            * (fus_mass_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * 0.5
            * (cabin_len_tailboom + pylon_len) ** (-0.5)
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dPylonLen = (
            c_fuselage
            * x
            * (fus_mass_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * 0.5
            * (cabin_len_tailboom + pylon_len) ** (-0.5)
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dMinDiveVel = (
            c_fuselage
            * x
            * (fus_mass_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * 1
            / (min_dive_vel * np.log(10))
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dPdiffFus = (
            c_fuselage
            * x
            * (fus_mass_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * 0.2
            * (p_diff_fus + 1.0) ** (0.2 - 1)
            * ULF**0.3
        )
        dFusWt_dULF = (
            c_fuselage
            * x
            * (fus_mass_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * 0.3
            * ULF ** (0.3 - 1)
        )
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_SCALER] = fus_mass / CK11
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_COEFFICIENT] = dFusWt_dCFus
        J[Aircraft.Fuselage.MASS, 'fus_mass_full'] = dFusWt_dFusWtFull
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.WETTED_AREA] = dFusWt_dFusSA
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.AVG_DIAMETER] = dFusWt_dCabWidth
        J[Aircraft.Fuselage.MASS, Aircraft.TailBoom.LENGTH] = dFusWt_dCabLenBoom
        J[Aircraft.Fuselage.MASS, 'pylon_len'] = dFusWt_dPylonLen
        J[Aircraft.Fuselage.MASS, 'min_dive_vel'] = dFusWt_dMinDiveVel
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL] = dFusWt_dPdiffFus
        J[Aircraft.Fuselage.MASS, Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = dFusWt_dULF
        J[Aircraft.Fuselage.MASS, 'MAT'] = 1


class BWBFuselageMass(om.ExplicitComponent):
    """
    Computation of the fuselage structure mass of BWB type aircraft. The empirical
    equation is completely different from tube and wing type aircraft.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(
            self, Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL, units='unitless'
        )
        add_aviary_input(self, Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA, units='lbm/ft**2')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.MASS_SCALER)

        add_aviary_output(self, Aircraft.Fuselage.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.MASS,
            [
                Aircraft.Fuselage.CABIN_AREA,
                Aircraft.Fuselage.WETTED_AREA,
                Aircraft.Fuselage.MASS_COEFFICIENT,
                Aircraft.Design.GROSS_MASS,
                Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL,
                Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA,
                Aircraft.Fuselage.MASS_SCALER,
            ],
        )

    def compute(self, inputs, outputs):
        c_fuselage = inputs[Aircraft.Fuselage.MASS_COEFFICIENT]
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        area_cabin = inputs[Aircraft.Fuselage.CABIN_AREA]
        area_aft_to_total = inputs[Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL]
        uwt_aft = inputs[Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA]
        fus_SA = inputs[Aircraft.Fuselage.WETTED_AREA]
        CK11 = inputs[Aircraft.Fuselage.MASS_SCALER]

        fus_mass_fb = c_fuselage * 1.8 * (gross_mass_initial**0.167) * (area_cabin**1.06)
        fus_mass_ab = c_fuselage * fus_SA * area_aft_to_total * uwt_aft
        fus_mass = fus_mass_fb + fus_mass_ab

        outputs[Aircraft.Fuselage.MASS] = CK11 * fus_mass

    def compute_partials(self, inputs, J):
        c_fuselage = inputs[Aircraft.Fuselage.MASS_COEFFICIENT]
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        area_cabin = inputs[Aircraft.Fuselage.CABIN_AREA]
        area_aft_to_total = inputs[Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL]
        uwt_aft = inputs[Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA]
        fus_SA = inputs[Aircraft.Fuselage.WETTED_AREA]

        dFusWt_dc_fuselage = (
            1.8 * (gross_mass_initial**0.167) * (area_cabin**1.06)
            + fus_SA * area_aft_to_total * uwt_aft
        )
        dFusWt_dgross_mass_initial = (
            0.167 * c_fuselage * 1.8 * (gross_mass_initial**-0.833) * (area_cabin**1.06)
        )
        dFusWt_darea_cabin = (
            1.06 * c_fuselage * 1.8 * (gross_mass_initial**0.167) * (area_cabin**0.06)
        )
        dFusWt_darea_aft_to_total = c_fuselage * fus_SA * uwt_aft
        dFusWt_duwt_aft = c_fuselage * fus_SA * area_aft_to_total
        dFusWt_dfus_SA = c_fuselage * area_aft_to_total * uwt_aft

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_COEFFICIENT] = dFusWt_dc_fuselage
        J[Aircraft.Fuselage.MASS, Aircraft.Design.GROSS_MASS] = dFusWt_dgross_mass_initial
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.CABIN_AREA] = dFusWt_darea_cabin
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL] = (
            dFusWt_darea_aft_to_total
        )
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA] = dFusWt_duwt_aft
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.WETTED_AREA] = dFusWt_dfus_SA
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_SCALER] = (
            c_fuselage * 1.8 * (gross_mass_initial**0.167) * (area_cabin**1.06)
            + c_fuselage * fus_SA * area_aft_to_total * uwt_aft
        )
