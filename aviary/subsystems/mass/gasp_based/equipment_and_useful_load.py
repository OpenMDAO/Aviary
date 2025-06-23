import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.functions import sigmoidX, dSigmoidXdx, smooth_max, d_smooth_max
from aviary.variable_info.enums import AircraftTypes, GASPEngineType, Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission, Settings


def get_num_of_flight_attendent(num_pax):
    num_flight_attendants = 0
    if num_pax >= 20.0:
        num_flight_attendants = 1
    if num_pax >= 51.0:
        num_flight_attendants = 2
    if num_pax >= 101.0:
        num_flight_attendants = 3
    if num_pax >= 151.0:
        num_flight_attendants = 4
    if num_pax >= 201.0:
        num_flight_attendants = 5
    if num_pax >= 251.0:
        num_flight_attendants = 6

    return num_flight_attendants


def get_num_of_pilots(num_pax, engine_type):
    num_pilots = 1
    if num_pax > 9.0:
        num_pilots = 2
    if engine_type is GASPEngineType.TURBOJET and num_pax > 5.0:
        num_pilots = 2
    if num_pax >= 351.0:
        num_pilots = 3

    return num_pilots


def get_num_of_lavatories(num_pax):
    num_lavatories = 0
    if num_pax > 25.0:
        num_lavatories = 1
    if num_pax >= 51.0:
        num_lavatories = 2
    if num_pax >= 101.0:
        num_lavatories = 3
    if num_pax >= 151.0:
        num_lavatories = 4
    if num_pax >= 201.0:
        num_lavatories = 5
    if num_pax >= 251.0:
        num_lavatories = 6

    return num_lavatories


class EquipMassPartialSum(om.ExplicitComponent):
    """
    Computation of fixed equipment mass and useful load for GASP-based mass.
    AC and furnishing masses are removed. Others will be moved to individual components.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Aircraft.LandingGear.FIXED_GEAR)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)

    def setup(self):
        add_aviary_input(self, Aircraft.AntiIcing.MASS, units='lbm')
        add_aviary_input(self, Aircraft.APU.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Avionics.MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, units='lbm')
        add_aviary_input(self, Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, units='lbm')
        add_aviary_input(
            self, Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, units='unitless'
        )
        add_aviary_input(self, Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Instruments.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(
            self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, units='lbm'
        )
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT)

        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Controls.TOTAL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.VerticalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER, units='lbm')

        self.add_output('equip_mass_part', units='lbm')

        self.declare_partials('equip_mass_part', '*')
        self.declare_partials(
            'equip_mass_part',
            Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS,
            val=1.0 / GRAV_ENGLISH_LBM,
        )

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        wingspan = inputs[Aircraft.Wing.SPAN]

        landing_gear_wt = inputs[Aircraft.LandingGear.TOTAL_MASS] * GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        wing_area = inputs[Aircraft.Wing.AREA]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]
        subsystems_wt = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]
        elec_mass_coeff = inputs[Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER] * GRAV_ENGLISH_LBM

        engine_type = self.options[Aircraft.Engine.TYPE][0]

        if PAX > 35.0:
            APU_wt = 26.2 * PAX**0.944 - 13.6 * PAX
        else:
            APU_wt = 0.0
        # TODO The following if-block should be removed. Aircraft.APU.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.APU.MASS] < 1e-5):
            # note: this technically creates a discontinuity
            APU_wt = inputs[Aircraft.APU.MASS] * GRAV_ENGLISH_LBM

        num_pilots = get_num_of_pilots(PAX, engine_type)

        instrument_wt = (
            inputs[Aircraft.Instruments.MASS_COEFFICIENT]
            * gross_wt_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan**0.696
        )
        hydraulic_wt = inputs[
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT
        ] * control_wt + inputs[Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT] * landing_gear_wt * (
            not self.options[Aircraft.LandingGear.FIXED_GEAR]
        )

        if PAX <= 12:
            electrical_wt = 0.03217 * gross_wt_initial - 20.0
        else:
            if num_engines == 1:
                electrical_wt = 0.00778 * gross_wt_initial + 33.0
            else:
                electrical_wt = elec_mass_coeff * PAX + 170.0

        avionics_wt = 27.0

        # GASP avionics weight model was put together long before modern systems
        # came on-board, and should be updated.
        if PAX < 20:
            if smooth:
                # Exponential regression from four points:
                # (3000, 65), (5500, 113), (7500, 163), (11000, 340)
                # avionics_wt = 36.2 * exp(0.0002024 * gross_wt_initial)
                # Exponential regression from five points:
                # (0, 27), (3000, 65), (5500, 113), (7500, 163), (11000, 340)
                # avionics_wt = 30.03 * exp(0.0002262 * gross_wt_initial)
                # Should we use use 4 sigmoid functions (one for each transition zone) instead?
                avionics_wt = 35.538 * np.exp(0.0002 * gross_wt_initial)
            else:
                if gross_wt_initial >= 3000.0:  # note: this technically creates a discontinuity
                    avionics_wt = 65.0
                if gross_wt_initial >= 5500.0:  # note: this technically creates a discontinuity
                    avionics_wt = 113.0
                if gross_wt_initial >= 7500.0:  # note: this technically creates a discontinuity
                    avionics_wt = 163.0
                if gross_wt_initial >= 11000.0:  # note: this technically creates a discontinuity
                    avionics_wt = 340.0
        if PAX >= 20 and PAX < 30:
            avionics_wt = 400.0
        elif PAX >= 30 and PAX <= 50:
            avionics_wt = 500.0
        elif PAX > 50 and PAX <= 100:
            avionics_wt = 600.0
        if PAX > 100:
            avionics_wt = 2.8 * PAX + 1010.0
        # TODO The following if-block should be removed. Aircraft.Avionics.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.Avionics.MASS] < 1e-5):
            # note: this technically creates a discontinuity !WILL NOT CHANGE
            avionics_wt = inputs[Aircraft.Avionics.MASS] * GRAV_ENGLISH_LBM

        SSUM = wing_area + htail_area + vtail_area
        icing_wt = 22.7 * (SSUM**0.5) - 385.0

        if smooth:
            # This should be implemented:
            # icing_wt = smooth_max(icing_wt, 0.0, mu)
            pass
        else:
            if icing_wt < 0.0:  # note: this technically creates a discontinuity
                icing_wt = 0.0
        # TODO The following if-block should be removed. Aircraft.AntiIcing.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.AntiIcing.MASS] < 1e-5):
            # note: this technically creates a discontinuity !WILL NOT CHANGE
            icing_wt = inputs[Aircraft.AntiIcing.MASS] * GRAV_ENGLISH_LBM

        if PAX < 9:
            if smooth:
                aux_wt = 3 * sigmoidX(gross_wt_initial / 3000, 1.0, 0.01)
            else:
                if gross_wt_initial > 3000.0:  # note: this technically creates a discontinuity
                    aux_wt = 3.0
                else:
                    aux_wt = 0.0
        elif PAX >= 9 and PAX < 20:
            aux_wt = 10.0
        elif PAX >= 20 and PAX < 75:
            aux_wt = 20.0
        else:
            aux_wt = 50.0

        fixed_equip_wt = (
            APU_wt
            + instrument_wt
            + hydraulic_wt
            + electrical_wt
            + avionics_wt
            + icing_wt
            + aux_wt
            + subsystems_wt
        )

        outputs['equip_mass_part'] = fixed_equip_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, partials):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        wingspan = inputs[Aircraft.Wing.SPAN]

        landing_gear_wt = inputs[Aircraft.LandingGear.TOTAL_MASS] * GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        wing_area = inputs[Aircraft.Wing.AREA]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]

        engine_type = self.options[Aircraft.Engine.TYPE][0]

        dAPU_wt_dmass_coeff_0 = 0.0
        # TODO The following if-block should be removed. Aircraft.APU.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.APU.MASS] < 1e-5):
            # note: this technically creates a discontinuity
            dAPU_wt_dmass_coeff_0 = GRAV_ENGLISH_LBM

        num_pilots = get_num_of_pilots(PAX, engine_type)

        dinstrument_wt_dmass_coeff_1 = (
            gross_wt_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan**0.696
        )
        dinstrument_wt_dgross_wt_initial = (
            0.386
            * inputs[Aircraft.Instruments.MASS_COEFFICIENT]
            * gross_wt_initial ** (0.386 - 1)
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan**0.696
        )
        dinstrument_wt_dfus_len = (
            0.05
            * inputs[Aircraft.Instruments.MASS_COEFFICIENT]
            * gross_wt_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len ** (0.05 - 1)
            * wingspan**0.696
        )
        dinstrument_wt_dwingspan = (
            0.696
            * inputs[Aircraft.Instruments.MASS_COEFFICIENT]
            * gross_wt_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan ** (0.696 - 1)
        )

        gear_val = not self.options[Aircraft.LandingGear.FIXED_GEAR]

        dhydraulic_wt_dmass_coeff_2 = control_wt
        dhydraulic_wt_dcontrol_wt = inputs[Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT]
        dhydraulic_wt_dmass_coeff_3 = landing_gear_wt * gear_val
        dhydraulic_wt_dlanding_gear_weight = (
            inputs[Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT] * gear_val
        )

        if PAX <= 12.0:
            delectrical_wt_dgross_wt_initial = 0.03217
            delectrical_wt_delec_mass_coeff = 0.0
        else:
            if num_engines == 1:
                delectrical_wt_dgross_wt_initial = 0.0078
                delectrical_wt_delec_mass_coeff = 0.0
            else:
                delectrical_wt_dgross_wt_initial = 0.0
                delectrical_wt_delec_mass_coeff = PAX * GRAV_ENGLISH_LBM

        davionics_wt_dmass_coeff_4 = 0.0

        if PAX < 20:
            if smooth:
                davionics_wt_dgross_wt_initial = 0.0071076 * np.exp(0.0002 * gross_wt_initial)
            else:
                davionics_wt_dgross_wt_initial = 0.0
        else:
            davionics_wt_dgross_wt_initial = 0.0
        # TODO The following if-block should be removed. Aircraft.Avionics.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.Avionics.MASS] < 1e-5):
            # note: this technically creates a discontinuity !WILL NOT CHANGE
            davionics_wt_dgross_wt_initial = 0.0
            davionics_wt_dmass_coeff_4 = GRAV_ENGLISH_LBM

        SSUM = wing_area + htail_area + vtail_area
        icing_wt = 22.7 * (SSUM**0.5) - 385.0

        dicing_weight_dwing_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dhtail_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dvtail_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dmass_coeff_6 = 0.0

        if smooth:
            # The partials of smooth_max should be implemented here.
            pass
        else:
            if icing_wt < 0.0:  # note: this technically creates a discontinuity
                icing_wt = 0.0
                dicing_weight_dwing_area = 0.0
                dicing_weight_dhtail_area = 0.0
                dicing_weight_dvtail_area = 0.0
                dicing_weight_dmass_coeff_6 = 0.0
        # TODO The following if-block should be removed. Aircraft.AntiIcing.MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.AntiIcing.MASS] < 1e-5):
            # note: this technically creates a discontinuity !WILL NOT CHANGE
            icing_wt = inputs[Aircraft.AntiIcing.MASS] * GRAV_ENGLISH_LBM
            dicing_weight_dwing_area = 0.0
            dicing_weight_dhtail_area = 0.0
            dicing_weight_dvtail_area = 0.0
            dicing_weight_dmass_coeff_6 = GRAV_ENGLISH_LBM

        if PAX < 9:
            if smooth:
                d_aux_wt_dgross_wt_initial = (
                    3 * dSigmoidXdx(gross_wt_initial / 3000, 1, 0.01) * 1 / 3000
                )
            else:
                if gross_wt_initial > 3000.0:  # note: this technically creates a discontinuity
                    d_aux_wt_dgross_wt_initial = 0.0
        else:
            d_aux_wt_dgross_wt_initial = 0.0

        dfixed_equip_mass_dmass_coeff_0 = dAPU_wt_dmass_coeff_0 / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dmass_coeff_1 = dinstrument_wt_dmass_coeff_1 / GRAV_ENGLISH_LBM
        dfixed_equip_wt_dgross_wt_initial = (
            dinstrument_wt_dgross_wt_initial
            + delectrical_wt_dgross_wt_initial
            #    + dfurnishing_wt_dgross_wt_initial
            + davionics_wt_dgross_wt_initial
            + d_aux_wt_dgross_wt_initial
            #    + dfurnishing_wt_dcabin_width
        )
        dfixed_equip_wt_delec_mass_coeff = delectrical_wt_delec_mass_coeff / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dfus_len = (
            dinstrument_wt_dfus_len  # + dair_conditioning_wt_dfus_len
        ) / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dwingspan = dinstrument_wt_dwingspan / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dmass_coeff_2 = dhydraulic_wt_dmass_coeff_2 / GRAV_ENGLISH_LBM
        dfixed_equip_wt_dcontrol_wt = dhydraulic_wt_dcontrol_wt
        dfixed_equip_mass_dmass_coeff_3 = dhydraulic_wt_dmass_coeff_3 / GRAV_ENGLISH_LBM
        dfixed_equip_wt_dlanding_gear_weight = dhydraulic_wt_dlanding_gear_weight
        dfixed_equip_mass_dmass_coeff_4 = davionics_wt_dmass_coeff_4 / GRAV_ENGLISH_LBM

        dfixed_equip_mass_dwing_area = dicing_weight_dwing_area / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dhtail_area = dicing_weight_dhtail_area / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dvtail_area = dicing_weight_dvtail_area / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dmass_coeff_6 = dicing_weight_dmass_coeff_6 / GRAV_ENGLISH_LBM

        partials['equip_mass_part', Aircraft.APU.MASS] = dfixed_equip_mass_dmass_coeff_0
        partials['equip_mass_part', Aircraft.Instruments.MASS_COEFFICIENT] = (
            dfixed_equip_mass_dmass_coeff_1
        )
        partials[
            'equip_mass_part',
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT,
        ] = dfixed_equip_mass_dmass_coeff_2
        partials['equip_mass_part', Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT] = (
            dfixed_equip_mass_dmass_coeff_3
        )
        partials['equip_mass_part', Aircraft.Avionics.MASS] = dfixed_equip_mass_dmass_coeff_4
        partials['equip_mass_part', Aircraft.AntiIcing.MASS] = dfixed_equip_mass_dmass_coeff_6
        partials['equip_mass_part', Mission.Design.GROSS_MASS] = dfixed_equip_wt_dgross_wt_initial
        partials['equip_mass_part', Aircraft.Fuselage.LENGTH] = dfixed_equip_mass_dfus_len
        partials['equip_mass_part', Aircraft.Wing.SPAN] = dfixed_equip_mass_dwingspan
        partials['equip_mass_part', Aircraft.Controls.TOTAL_MASS] = dfixed_equip_wt_dcontrol_wt
        partials['equip_mass_part', Aircraft.LandingGear.TOTAL_MASS] = (
            dfixed_equip_wt_dlanding_gear_weight
        )

        partials['equip_mass_part', Aircraft.Wing.AREA] = dfixed_equip_mass_dwing_area
        partials['equip_mass_part', Aircraft.HorizontalTail.AREA] = dfixed_equip_mass_dhtail_area
        partials['equip_mass_part', Aircraft.VerticalTail.AREA] = dfixed_equip_mass_dvtail_area
        partials['equip_mass_part', Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER] = (
            dfixed_equip_wt_delec_mass_coeff
        )


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


class FurnishingMass(om.ExplicitComponent):
    """
    Computation of furnishing mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Aircraft.Furnishings.USE_EMPIRICAL_EQUATION)
        self.options.declare('mu', default=1.0, types=float)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Furnishings.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2')

        self.add_output(Aircraft.Furnishings.MASS, units='lbm')

        self.declare_partials(
            Aircraft.Furnishings.MASS,
            [
                Aircraft.Fuselage.AVG_DIAMETER,
                Mission.Design.GROSS_MASS,
                Aircraft.Furnishings.MASS_SCALER,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.CABIN_AREA,
            ],
        )

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        empirical = self.options[Aircraft.Furnishings.USE_EMPIRICAL_EQUATION]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        mu = self.options['mu']

        num_pilots = get_num_of_pilots(PAX, engine_type)
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        lavatories = get_num_of_lavatories(PAX)

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        acabin = inputs[Aircraft.Fuselage.CABIN_AREA]

        if gross_wt_initial <= 10000.0:
            # note: this technically creates a discontinuity
            # TODO: Doesn't occur in large single aisle
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
        else:
            if PAX >= 50:
                if empirical:
                    # commonly used empirical furnishing weight equation
                    furnishing_wt_additional = scaler * PAX
                else:
                    # linear regression formula
                    furnishing_wt_additional = 118.4 * PAX - 4190.0
                # baseline furnishings (crew seats, cockpit, lavatories, galleys)
                cabin_len = 0.75 * fus_len
                agalley = 0.50 * PAX
                furnishing_wt = (
                    furnishing_wt_additional
                    + num_pilots * (1.0 + cabin_width / 12.0) * 90.0
                    + num_flight_attendants * 30.0
                    + lavatories * 240.0
                    + agalley * 12.0
                    + 1.5 * cabin_len * cabin_width * np.pi / 2.0
                    + 0.5 * acabin
                )
            else:
                CPX_lin = 28.0 + 10.516 * (cabin_width - 5.667)
                if smooth:
                    CPX = (
                        28 * sigmoidX(CPX_lin / 28, 1, -0.01)
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + 62 * sigmoidX(CPX_lin / 62, 1, 0.01)
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        CPX = 28.0
                    elif cabin_width > 8.90:  # note: this technically creates a discontinuity
                        CPX = 62.0
                furnishing_wt = CPX * PAX + 310.0
        # Theoretically, we should make sure that furnishing_wt >= 0 here

        if smooth:
            furnishing_wt = smooth_max(furnishing_wt, 30.0, mu)
        else:
            furnishing_wt = np.maximum(furnishing_wt, 30.0)

        outputs[Aircraft.Furnishings.MASS] = furnishing_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, partials):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        empirical = self.options[Aircraft.Furnishings.USE_EMPIRICAL_EQUATION]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        mu = self.options['mu']

        num_pilots = get_num_of_pilots(PAX, engine_type)
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        lavatories = get_num_of_lavatories(PAX)

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        acabin = inputs[Aircraft.Fuselage.CABIN_AREA]

        if gross_wt_initial <= 10000.0:
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
            dfurnishing_wt_dgross_wt_initial = 0.065
            dfurnishing_wt_dcabin_width = 0.0
            dfurnishing_wt_dfus_len = 0.0
            dfurnishing_wt_dscaler = 0.0
            dfurnishing_wt_dacabin = 0.0
        else:
            if PAX >= 50:
                if empirical:
                    furnishing_wt_additional = scaler * PAX
                    dfurnishing_wt_additional_dgross_wt_initial = 0.0
                    dfurnishing_wt_additional_dcabin_width = 0.0
                    dfurnishing_wt_additional_dfus_len = 0.0
                    dfurnishing_wt_additional_dscaler = PAX
                    dfurnishing_wt_additional_dacabin = 0.0
                else:
                    furnishing_wt_additional = 118.4 * PAX - 4190.0
                    dfurnishing_wt_additional_dgross_wt_initial = 0.0
                    dfurnishing_wt_additional_dcabin_width = 0.0
                    dfurnishing_wt_additional_dfus_len = 0.0
                    dfurnishing_wt_additional_dscaler = 0.0
                    dfurnishing_wt_additional_dacabin = 0.0
                cabin_len = 0.75 * fus_len
                agalley = 0.50 * PAX
                furnishing_wt = (
                    furnishing_wt_additional
                    + num_pilots * (1.0 + cabin_width / 12.0) * 90.0
                    + num_flight_attendants * 30.0
                    + lavatories * 240.0
                    + agalley * 12.0
                    + 1.5 * cabin_len * cabin_width * np.pi / 2.0
                    + 0.5 * acabin
                )
                dfurnishing_wt_dgross_wt_initial = dfurnishing_wt_additional_dgross_wt_initial + 0.0
                dfurnishing_wt_dcabin_width = (
                    dfurnishing_wt_additional_dcabin_width
                    + num_pilots * (1.0 / 12.0) * 90.0
                    + 1.5 * 0.75 * fus_len * np.pi / 2.0
                )
                dfurnishing_wt_dfus_len = (
                    dfurnishing_wt_additional_dfus_len + 1.5 * 0.75 * cabin_width * np.pi / 2.0
                )
                dfurnishing_wt_dscaler = dfurnishing_wt_additional_dscaler + 0.0
                dfurnishing_wt_dacabin = dfurnishing_wt_additional_dacabin + 0.5
            else:
                CPX_lin = 28.0 + 10.516 * (cabin_width - 5.667)
                if smooth:
                    CPX = (
                        28 * sigmoidX(CPX_lin / 28, 1, -0.01)
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + 62 * sigmoidX(CPX_lin / 62, 1, 0.01)
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        CPX = 28.0
                    elif cabin_width > 8.90:  # note: this technically creates a discontinuity
                        CPX = 62.0
                furnishing_wt = CPX * PAX + 310.0

                dCPX_lin_dcabin_width = 10.516
                if smooth:
                    dCPX_dcabin_width = (
                        1 * dSigmoidXdx(CPX_lin / 28, 1, 0.01) * -dCPX_lin_dcabin_width
                        + dCPX_lin_dcabin_width
                        * CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + CPX_lin
                        * dSigmoidXdx(CPX_lin / 28, 1, 0.01)
                        / 28
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        * dCPX_lin_dcabin_width
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * dSigmoidXdx(CPX_lin / 62, 1, -0.01)
                        / 62
                        * -dCPX_lin_dcabin_width
                        + 1 * dSigmoidXdx(CPX_lin / 62, 1, 0.01) * dCPX_lin_dcabin_width
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        dCPX_dcabin_width = 0.0
                    if cabin_width > 8.90:  # note: this technically creates a discontinuity
                        dCPX_dcabin_width = 0.0

                dfurnishing_wt_dcabin_width = PAX * dCPX_dcabin_width
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dfus_len = 0.0
                dfurnishing_wt_dscaler = 0.0
                dfurnishing_wt_dacabin = 0.0

        if smooth:
            sm_fac = d_smooth_max(furnishing_wt, 30.0, mu)
            dfurnishing_wt_dcabin_width = sm_fac * dfurnishing_wt_dcabin_width
            dfurnishing_wt_dgross_wt_initial = sm_fac * dfurnishing_wt_dgross_wt_initial
            dfurnishing_wt_dfus_len = sm_fac * dfurnishing_wt_dfus_len
            dfurnishing_wt_dscaler = sm_fac * dfurnishing_wt_dscaler
            dfurnishing_wt_dacabin = sm_fac * dfurnishing_wt_dacabin
        else:
            if furnishing_wt < 30.0:  # note: this technically creates a discontinuity
                dfurnishing_wt_dcabin_width = 0.0
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dfus_len = 0.0
                dfurnishing_wt_dscaler = 0.0
                dfurnishing_wt_dacabin = 0.0

        partials[Aircraft.Furnishings.MASS, Mission.Design.GROSS_MASS] = (
            dfurnishing_wt_dgross_wt_initial
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.AVG_DIAMETER] = (
            dfurnishing_wt_dcabin_width / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.LENGTH] = (
            dfurnishing_wt_dfus_len / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Furnishings.MASS_SCALER] = (
            dfurnishing_wt_dscaler / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.CABIN_AREA] = (
            dfurnishing_wt_dacabin / GRAV_ENGLISH_LBM
        )


class EquipMassSum(om.ExplicitComponent):
    def setup(self):
        self.add_input('equip_mass_part', units='lbm')
        add_aviary_input(self, Aircraft.AirConditioning.MASS, units='lbm')
        self.add_input(Aircraft.Furnishings.MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.FIXED_EQUIPMENT_MASS, units='lbm')

        self.declare_partials(
            Aircraft.Design.FIXED_EQUIPMENT_MASS,
            [
                'equip_mass_part',
                Aircraft.AirConditioning.MASS,
                Aircraft.Furnishings.MASS,
            ],
        )

    def compute(self, inputs, outputs):
        equip_mass_part = inputs['equip_mass_part']
        air_conditioning_mass = inputs[Aircraft.AirConditioning.MASS]
        furnishing_mass = inputs[Aircraft.Furnishings.MASS]

        equip_mass_sum = equip_mass_part + air_conditioning_mass + furnishing_mass
        outputs[Aircraft.Design.FIXED_EQUIPMENT_MASS] = equip_mass_sum

    def compute_partials(self, inputs, J):
        J[Aircraft.Design.FIXED_EQUIPMENT_MASS, 'equip_mass_part'] = 1
        J[Aircraft.Design.FIXED_EQUIPMENT_MASS, Aircraft.AirConditioning.MASS] = 1
        J[Aircraft.Design.FIXED_EQUIPMENT_MASS, Aircraft.Furnishings.MASS] = 1


class EquipMassGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'equip_partial',
            EquipMassPartialSum(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['equip_mass_part'],
        )
        self.add_subsystem(
            'ac',
            ACMass(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )
        self.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )
        self.add_subsystem(
            'equip_sum',
            EquipMassSum(),
            promotes_inputs=['equip_mass_part', 'aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )


class UsefulLoadMass(om.ExplicitComponent):
    """
    Computation of fixed equipment mass and useful load for GASP-based mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Settings.VERBOSITY)
        add_aviary_option(self, Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER, units='lbm')

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, units='lbm')
        add_aviary_input(self, Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, units='lbm')
        add_aviary_input(
            self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, units='lbm'
        )
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT)
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless')

        add_aviary_output(self, Aircraft.Design.FIXED_USEFUL_LOAD, units='lbm')

        self.declare_partials(Aircraft.Design.FIXED_USEFUL_LOAD, '*')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]

        wing_area = inputs[Aircraft.Wing.AREA]
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        uld_per_pax = self.options[Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER][0]

        engine_type = self.options[Aircraft.Engine.TYPE][0]
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        num_pilots = get_num_of_pilots(PAX, engine_type)

        # note: the average weight of a pilot was calculated using the following equation:
        # avg_wt = pct_male*avg_wt_male + pct_female*avg_wt_female where
        # pct_male = the percentage of US airline pilots that are male (based on data from
        # the center for aviation in 2018, which listed this percentage as 95.6%, and slightly
        # deflated to account for the upward trend in female pilots, resulting in an estimated
        # percentage of 94%)
        # avg_wt_male is the average weight of males according to the CDC, and is 199.8 lbf
        # pct_female is calculated from the same methods as pct_male, and results in 6%
        # avg_wt_female is the average weight of females according to the CDC, and is 170.8 lbf
        # the resulting value is that the average weight of the US airline pilot is 198 lbf
        pilot_wt = 198 * num_pilots

        # note: the average weight of a flight attendant was calulated using the following equation:
        # avg_wt = pct_male*avg_wt_male + pct_female*avg_wt_female where
        # pct_male = the percentage of US flight attendants that are male (based on data from
        # the women in aerospace international organization in 2020, which listed this percentage as
        # 20.8%)
        # avg_wt_male is the average weight of males according to the CDC, and is 199.8 lbf
        # pct_female is calculated from the same methods as pct_male, and results in 79.2%
        # avg_wt_female is the average weight of females according to the CDC, and is 170.8 lbf
        # the resulting value is that the average weight of the US flight attendant is 177 lbf
        flight_attendant_wt = 177 * num_flight_attendants

        if PAX >= 40.0:
            crew_bag_wt = 20.0 * (num_flight_attendants + num_pilots) + 25.0 * num_pilots
        elif PAX < 20:
            crew_bag_wt = 25.0 * num_pilots
        else:
            crew_bag_wt = 10.0 * (num_pilots + num_flight_attendants) + 25.0

        if engine_type is GASPEngineType.TURBOJET:
            oil_per_eng_wt = 0.0054 * Fn_SLS + 12.0
        elif engine_type is GASPEngineType.TURBOSHAFT or engine_type is GASPEngineType.TURBOPROP:
            oil_per_eng_wt = 0.0214 * Fn_SLS + 14
        # else:
        #     oil_per_eng_wt = 0.062 * (Fn_SLS - 100) + 11
        else:
            # Other engine types are currently not supported in Aviary
            if verbosity > Verbosity.BRIEF:
                print('This engine_type is not curretly supported in Aviary.')
            oil_per_eng_wt = 0

        oil_wt = num_engines * oil_per_eng_wt

        lavatories = get_num_of_lavatories(PAX)

        service_wt = 0.0
        if PAX > 9.0:
            service_wt = (
                inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER]
                * PAX
                * GRAV_ENGLISH_LBM
                + 16.0 * lavatories
            )

        water_wt = 0.0
        if PAX > 19.0:
            water_wt = (
                inputs[Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT]
                * (PAX + num_pilots + num_flight_attendants)
                * GRAV_ENGLISH_LBM
            )

        emergency_wt = 0.0
        if PAX > 5.0:
            emergency_wt = 10.0
        if PAX > 9.0:
            emergency_wt = 15.0
        if PAX >= 35.0:
            emergency_wt = 25.0 * num_flight_attendants + 15.0
        # TODO The following if-block should be removed. Aircraft.Design.EMERGENCY_EQUIPMENT_MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] < 1e-5):
            emergency_wt = inputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] * GRAV_ENGLISH_LBM

        catering_wt = 0.0
        if PAX > 19.0:
            catering_wt = (
                inputs[Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER]
                * PAX
                * GRAV_ENGLISH_LBM
            )

        trapped_fuel_wt = (
            inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]
            * (wing_area**0.5)
            * fuel_vol_frac
            / 0.430
        )
        if fuel_vol_frac <= 0.075:  # note: this technically creates a discontinuity # won't change
            trapped_fuel_wt = (
                inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] * 0.18 * (wing_area**0.5)
            )

        unit_weight_cargo_handling = 165.0
        uld_per_pax = uld_per_pax.real
        cargo_handling_wt = (int(PAX * uld_per_pax) + 1) * unit_weight_cargo_handling

        useful_wt = (
            pilot_wt
            + flight_attendant_wt
            + crew_bag_wt
            + oil_wt
            + service_wt
            + water_wt
            + emergency_wt
            + catering_wt
            + trapped_fuel_wt
            + cargo_handling_wt
        )

        outputs[Aircraft.Design.FIXED_USEFUL_LOAD] = useful_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, partials):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]

        wing_area = inputs[Aircraft.Wing.AREA]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]

        engine_type = self.options[Aircraft.Engine.TYPE][0]

        num_pilots = get_num_of_pilots(PAX, engine_type)

        num_flight_attendants = get_num_of_flight_attendent(PAX)

        if engine_type is GASPEngineType.TURBOJET:
            doil_per_eng_wt_dFn_SLS = 0.0054
        elif engine_type is GASPEngineType.TURBOSHAFT or engine_type is GASPEngineType.TURBOPROP:
            doil_per_eng_wt_dFn_SLS = 0.0124
        # else:
        #     doil_per_eng_wt_dFn_SLS = 0.062
        else:
            # Other engine types are currently not supported in Aviary
            doil_per_eng_wt_dFn_SLS = 0.0

        dservice_wt_dmass_coeff_8 = 0.0
        if PAX > 9.0:
            dservice_wt_dmass_coeff_8 = PAX * GRAV_ENGLISH_LBM

        dwater_wt_dmass_coeff_9 = 0.0
        if PAX > 19.0:
            dwater_wt_dmass_coeff_9 = (PAX + num_pilots + num_flight_attendants) * GRAV_ENGLISH_LBM

        demergency_wt_dmass_coeff_10 = 0.0
        # TODO The following if-block should be removed. Aircraft.Design.EMERGENCY_EQUIPMENT_MASS should be output, not input.
        if not (-1e-5 < inputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] < 1e-5):
            demergency_wt_dmass_coeff_10 = GRAV_ENGLISH_LBM

        dcatering_wt_dmass_coeff_11 = 0.0
        if PAX > 19.0:
            dcatering_wt_dmass_coeff_11 = PAX * GRAV_ENGLISH_LBM

        dtrapped_fuel_wt_dmass_coeff_12 = (wing_area**0.5) * fuel_vol_frac / 0.430
        dtrapped_fuel_wt_dwing_area = (
            0.5
            * inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]
            * (wing_area**-0.5)
            * fuel_vol_frac
            / 0.430
        )
        dtrapped_fuel_wt_dfuel_vol_frac = (
            inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] * (wing_area**0.5) / 0.430
        )

        if fuel_vol_frac <= 0.075:  # note: this technically creates a discontinuity # won't change
            dtrapped_fuel_wt_dmass_coeff_12 = 0.18 * (wing_area**0.5)
            dtrapped_fuel_wt_dwing_area = (
                0.5
                * inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]
                * 0.18
                * (wing_area**-0.5)
            )
            dtrapped_fuel_wt_dfuel_vol_frac = 0.0

        doil_wt_dFnSLS = num_engines * doil_per_eng_wt_dFn_SLS
        duseful_mass_dFn_SLS = doil_wt_dFnSLS / GRAV_ENGLISH_LBM

        duseful_mass_dmass_coeff_8 = dservice_wt_dmass_coeff_8 / GRAV_ENGLISH_LBM
        duseful_mass_dmass_coeff_9 = dwater_wt_dmass_coeff_9 / GRAV_ENGLISH_LBM
        duseful_mass_dmass_coeff_10 = demergency_wt_dmass_coeff_10 / GRAV_ENGLISH_LBM
        duseful_mass_dmass_coeff_11 = dcatering_wt_dmass_coeff_11 / GRAV_ENGLISH_LBM
        duseful_mass_dmass_coeff_12 = dtrapped_fuel_wt_dmass_coeff_12 / GRAV_ENGLISH_LBM
        duseful_mass_dwing_area = dtrapped_fuel_wt_dwing_area / GRAV_ENGLISH_LBM
        duseful_mass_dfuel_vol_frac = dtrapped_fuel_wt_dfuel_vol_frac / GRAV_ENGLISH_LBM
        partials[Aircraft.Design.FIXED_USEFUL_LOAD, Aircraft.Engine.SCALED_SLS_THRUST] = (
            duseful_mass_dFn_SLS
        )
        partials[
            Aircraft.Design.FIXED_USEFUL_LOAD,
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER,
        ] = duseful_mass_dmass_coeff_8
        partials[
            Aircraft.Design.FIXED_USEFUL_LOAD, Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT
        ] = duseful_mass_dmass_coeff_9
        partials[Aircraft.Design.FIXED_USEFUL_LOAD, Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] = (
            duseful_mass_dmass_coeff_10
        )
        partials[
            Aircraft.Design.FIXED_USEFUL_LOAD,
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER,
        ] = duseful_mass_dmass_coeff_11
        partials[
            Aircraft.Design.FIXED_USEFUL_LOAD, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT
        ] = duseful_mass_dmass_coeff_12

        partials[Aircraft.Design.FIXED_USEFUL_LOAD, Aircraft.Wing.AREA] = duseful_mass_dwing_area
        partials[Aircraft.Design.FIXED_USEFUL_LOAD, Aircraft.Fuel.WING_FUEL_FRACTION] = (
            duseful_mass_dfuel_vol_frac
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


class BWBFurnishingMass(om.ExplicitComponent):
    """
    Computation of furnishing mass for BWB
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Aircraft.Furnishings.USE_EMPIRICAL_EQUATION)
        self.options.declare('mu', default=1.0, types=float)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.HYDRAULIC_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Furnishings.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2')

        self.add_output(Aircraft.Furnishings.MASS, units='lbm')

        self.declare_partials(
            Aircraft.Furnishings.MASS,
            [
                Aircraft.Fuselage.HYDRAULIC_DIAMETER,
                Mission.Design.GROSS_MASS,
                Aircraft.Furnishings.MASS_SCALER,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.CABIN_AREA,
            ],
        )

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        empirical = self.options[Aircraft.Furnishings.USE_EMPIRICAL_EQUATION]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        mu = self.options['mu']

        num_pilots = get_num_of_pilots(PAX, engine_type)
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        lavatories = get_num_of_lavatories(PAX)

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        acabin = inputs[Aircraft.Fuselage.CABIN_AREA]

        if gross_wt_initial <= 10000.0:
            # note: this technically creates a discontinuity
            # TODO: Doesn't occur in large single aisle
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
        else:
            if PAX >= 50:
                if empirical:
                    # commonly used empirical furnishing weight equation
                    furnishing_wt_additional = scaler * PAX
                else:
                    # linear regression formula
                    furnishing_wt_additional = 118.4 * PAX - 4190.0
                # baseline furnishings (crew seats, cockpit, lavatories, galleys)
                cabin_len = 0.75 * fus_len
                agalley = 0.50 * PAX
                furnishing_wt = (
                    furnishing_wt_additional
                    + num_pilots * (1.0 + cabin_width / 12.0) * 90.0
                    + num_flight_attendants * 30.0
                    + lavatories * 240.0
                    + agalley * 12.0
                    + 1.5 * cabin_len * cabin_width * np.pi / 2.0
                    + 0.5 * acabin
                )
            else:
                CPX_lin = 28.0 + 10.516 * (cabin_width - 5.667)
                if smooth:
                    CPX = (
                        28 * sigmoidX(CPX_lin / 28, 1, -0.01)
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + 62 * sigmoidX(CPX_lin / 62, 1, 0.01)
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        CPX = 28.0
                    elif cabin_width > 8.90:  # note: this technically creates a discontinuity
                        CPX = 62.0
                furnishing_wt = CPX * PAX + 310.0

        if smooth:
            furnishing_wt = smooth_max(furnishing_wt, 30.0, mu)
        else:
            furnishing_wt = np.maximum(furnishing_wt, 30.0)

        outputs[Aircraft.Furnishings.MASS] = furnishing_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, partials):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        empirical = self.options[Aircraft.Furnishings.USE_EMPIRICAL_EQUATION]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        mu = self.options['mu']

        num_pilots = get_num_of_pilots(PAX, engine_type)
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        lavatories = get_num_of_lavatories(PAX)

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        acabin = inputs[Aircraft.Fuselage.CABIN_AREA]

        if gross_wt_initial <= 10000.0:
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
            dfurnishing_wt_dgross_wt_initial = 0.065
            dfurnishing_wt_dcabin_width = 0.0
            dfurnishing_wt_dfus_len = 0.0
            dfurnishing_wt_dscaler = 0.0
            dfurnishing_wt_dacabin = 0.0
        else:
            if PAX >= 50:
                if empirical:
                    furnishing_wt_additional = scaler * PAX
                    dfurnishing_wt_additional_dgross_wt_initial = 0.0
                    dfurnishing_wt_additional_dcabin_width = 0.0
                    dfurnishing_wt_additional_dfus_len = 0.0
                    dfurnishing_wt_additional_dscaler = PAX
                    dfurnishing_wt_additional_dacabin = 0.0
                else:
                    furnishing_wt_additional = 118.4 * PAX - 4190.0
                    dfurnishing_wt_additional_dgross_wt_initial = 0.0
                    dfurnishing_wt_additional_dcabin_width = 0.0
                    dfurnishing_wt_additional_dfus_len = 0.0
                    dfurnishing_wt_additional_dscaler = 0.0
                    dfurnishing_wt_additional_dacabin = 0.0
                cabin_len = 0.75 * fus_len
                agalley = 0.50 * PAX
                furnishing_wt = (
                    furnishing_wt_additional
                    + num_pilots * (1.0 + cabin_width / 12.0) * 90.0
                    + num_flight_attendants * 30.0
                    + lavatories * 240.0
                    + agalley * 12.0
                    + 1.5 * cabin_len * cabin_width * np.pi / 2.0
                    + 0.5 * acabin
                )
                dfurnishing_wt_dgross_wt_initial = dfurnishing_wt_additional_dgross_wt_initial + 0.0
                dfurnishing_wt_dcabin_width = (
                    dfurnishing_wt_additional_dcabin_width
                    + num_pilots * (1.0 / 12.0) * 90.0
                    + 1.5 * 0.75 * fus_len * np.pi / 2.0
                )
                dfurnishing_wt_dfus_len = (
                    dfurnishing_wt_additional_dfus_len + 1.5 * 0.75 * cabin_width * np.pi / 2.0
                )
                dfurnishing_wt_dscaler = dfurnishing_wt_additional_dscaler + 0.0
                dfurnishing_wt_dacabin = dfurnishing_wt_additional_dacabin + 0.5
            else:
                CPX_lin = 28.0 + 10.516 * (cabin_width - 5.667)
                if smooth:
                    CPX = (
                        28 * sigmoidX(CPX_lin / 28, 1, -0.01)
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + 62 * sigmoidX(CPX_lin / 62, 1, 0.01)
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        CPX = 28.0
                    elif cabin_width > 8.90:  # note: this technically creates a discontinuity
                        CPX = 62.0
                furnishing_wt = CPX * PAX + 310.0

                dCPX_lin_dcabin_width = 10.516
                if smooth:
                    dCPX_dcabin_width = (
                        1 * dSigmoidXdx(CPX_lin / 28, 1, 0.01) * -dCPX_lin_dcabin_width
                        + dCPX_lin_dcabin_width
                        * CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + CPX_lin
                        * dSigmoidXdx(CPX_lin / 28, 1, 0.01)
                        / 28
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        * dCPX_lin_dcabin_width
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * dSigmoidXdx(CPX_lin / 62, 1, -0.01)
                        / 62
                        * -dCPX_lin_dcabin_width
                        + 1 * dSigmoidXdx(CPX_lin / 62, 1, 0.01) * dCPX_lin_dcabin_width
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        dCPX_dcabin_width = 0.0
                    if cabin_width > 8.90:  # note: this technically creates a discontinuity
                        dCPX_dcabin_width = 0.0

                dfurnishing_wt_dcabin_width = PAX * dCPX_dcabin_width
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dfus_len = 0.0
                dfurnishing_wt_dscaler = 0.0
                dfurnishing_wt_dacabin = 0.0

        if smooth:
            sm_fac = d_smooth_max(furnishing_wt, 30.0, mu)
            dfurnishing_wt_dcabin_width = sm_fac * dfurnishing_wt_dcabin_width
            dfurnishing_wt_dgross_wt_initial = sm_fac * dfurnishing_wt_dgross_wt_initial
            dfurnishing_wt_dfus_len = sm_fac * dfurnishing_wt_dfus_len
            dfurnishing_wt_dscaler = sm_fac * dfurnishing_wt_dscaler
            dfurnishing_wt_dacabin = sm_fac * dfurnishing_wt_dacabin
        else:
            if furnishing_wt < 30.0:  # note: this technically creates a discontinuity
                dfurnishing_wt_dcabin_width = 0.0
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dfus_len = 0.0
                dfurnishing_wt_dscaler = 0.0
                dfurnishing_wt_dacabin = 0.0

        partials[Aircraft.Furnishings.MASS, Mission.Design.GROSS_MASS] = (
            dfurnishing_wt_dgross_wt_initial
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.HYDRAULIC_DIAMETER] = (
            dfurnishing_wt_dcabin_width / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.LENGTH] = (
            dfurnishing_wt_dfus_len / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Furnishings.MASS_SCALER] = (
            dfurnishing_wt_dscaler / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.CABIN_AREA] = (
            dfurnishing_wt_dacabin / GRAV_ENGLISH_LBM
        )


class BWBEquipMassGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'equip_partial',
            EquipMassPartialSum(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['equip_mass_part'],
        )
        self.add_subsystem(
            'ac',
            BWBACMass(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )
        self.add_subsystem(
            'furnishing',
            BWBFurnishingMass(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )
        self.add_subsystem(
            'equip_sum',
            EquipMassSum(),
            promotes_inputs=['equip_mass_part', 'aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )


class EquipAndUsefulLoadMassGroup(om.Group):
    def initialize(self):
        add_aviary_option(self, Aircraft.Design.TYPE)

    def setup(self):
        design_type = self.options[Aircraft.Design.TYPE]

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'equip',
                BWBEquipMassGroup(),
                promotes_inputs=['aircraft:*', 'mission:*'],
                promotes_outputs=['aircraft:*'],
            )
        else:
            self.add_subsystem(
                'equip',
                EquipMassGroup(),
                promotes_inputs=['aircraft:*', 'mission:*'],
                promotes_outputs=['aircraft:*'],
            )

        self.add_subsystem(
            'useful',
            UsefulLoadMass(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )
