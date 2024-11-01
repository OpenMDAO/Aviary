import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import GASPEngineType
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


def sig(x):
    return 1 / (1 + np.exp(-100 * x))


def dsig(x):
    return 100 * np.exp(-100 * x) / (np.exp(-100 * x) + 1) ** 2


class EquipAndUsefulLoadMass(om.ExplicitComponent):
    """
    Computation of fixed equipment mass and useful load for GASP-based mass.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        num_engine_type = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))

        add_aviary_input(
            self, Aircraft.AirConditioning.MASS_COEFFICIENT, val=1, units="unitless")
        add_aviary_input(self, Aircraft.AntiIcing.MASS, val=2, units="lbm")
        add_aviary_input(self, Aircraft.APU.MASS, val=3, units="lbm")
        add_aviary_input(self, Aircraft.Avionics.MASS, val=4, units="lbm")
        add_aviary_input(
            self, Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=5, units="lbm")
        add_aviary_input(self, Aircraft.Design.EMERGENCY_EQUIPMENT_MASS,
                         val=6, units="lbm")
        add_aviary_input(self, Aircraft.Furnishings.MASS, val=7, units="lbm")
        add_aviary_input(
            self, Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=8, units="unitless")
        add_aviary_input(
            self, Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=9, units="unitless")
        add_aviary_input(
            self, Aircraft.Instruments.MASS_COEFFICIENT, val=10, units="unitless")
        add_aviary_input(
            self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=11, units="lbm")
        add_aviary_input(
            self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12, units="unitless")
        add_aviary_input(
            self, Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=13, units="lbm")

        add_aviary_input(self, Mission.Design.GROSS_MASS, val=175400)
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, val=128)
        add_aviary_input(self, Aircraft.Wing.SPAN, val=117.8)
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS, val=200)
        add_aviary_input(self, Aircraft.Controls.TOTAL_MASS, val=150)
        add_aviary_input(self, Aircraft.Wing.AREA, val=150)
        add_aviary_input(self, Aircraft.HorizontalTail.AREA, val=150)
        add_aviary_input(self, Aircraft.VerticalTail.AREA, val=150)
        add_aviary_input(self, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5)
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, val=13.1)
        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST,
                         val=np.full(num_engine_type, 4000), units="lbf")
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, val=0.5)
        add_aviary_input(self, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, val=0.)

        add_aviary_output(self, Aircraft.Design.FIXED_USEFUL_LOAD, val=0)
        add_aviary_output(self, Aircraft.Design.FIXED_EQUIPMENT_MASS, val=0)

        self.declare_partials(Aircraft.Design.FIXED_USEFUL_LOAD, '*', val=0.0)
        self.declare_partials(Aircraft.Design.FIXED_EQUIPMENT_MASS, '*', val=0.0)
        self.declare_partials(Aircraft.Design.FIXED_EQUIPMENT_MASS,
                              Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, val=1./GRAV_ENGLISH_LBM)

    def compute(self, inputs, outputs):

        options: AviaryValues = self.options["aviary_options"]
        PAX = options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')
        smooth = options.get_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, units='unitless')

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        num_engines = self.options['aviary_options'].get_val(
            Aircraft.Propulsion.TOTAL_NUM_ENGINES, units='unitless')
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        wingspan = inputs[Aircraft.Wing.SPAN]
        if options.get_val(Aircraft.LandingGear.FIXED_GEAR, units='unitless'):
            gear_type = 1
        else:
            gear_type = 0

        landing_gear_wt = inputs[Aircraft.LandingGear.TOTAL_MASS] * \
            GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        wing_area = inputs[Aircraft.Wing.AREA]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        subsystems_wt = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]

        engine_type = options.get_val(Aircraft.Engine.TYPE, units='unitless')[0]

        APU_wt = 0.0
        if PAX > 35.0:
            APU_wt = 26.2 * PAX**0.944 - 13.6 * PAX
        if ~(
            -1e-5 < inputs[Aircraft.APU.MASS] < 1e-5
        ):  # note: this technically creates a discontinuity
            APU_wt = inputs[Aircraft.APU.MASS] * GRAV_ENGLISH_LBM

        num_pilots = 1.0
        if PAX > 9.0:
            num_pilots = 2.0
        if engine_type is GASPEngineType.TURBOJET and PAX > 5.0:
            num_pilots = 2.0
        if PAX >= 251.0:
            num_pilots = 3.0

        instrument_wt = (
            inputs[Aircraft.Instruments.MASS_COEFFICIENT]
            * gross_wt_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan**0.696
        )
        gear_val = 1 - gear_type
        hydraulic_wt = (
            inputs[Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT] * control_wt +
            inputs[Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT] *
            landing_gear_wt * gear_val
        )

        electrical_wt = 16.0 * PAX + 170.0
        if PAX <= 12.0:
            electrical_wt = 0.03217 * gross_wt_initial - 20.0
        if num_engines == 1.0:
            electrical_wt = 0.00778 * gross_wt_initial + 33.0

        avionics_wt = 27.0

        if smooth:

            avionics_wt = 35.538 * np.exp(0.0002 * gross_wt_initial)

        else:
            if (
                gross_wt_initial >= 3000.0
            ):  # note: this technically creates a discontinuity
                avionics_wt = 65.0
            if (
                gross_wt_initial >= 5500.0
            ):  # note: this technically creates a discontinuity
                avionics_wt = 113.0
            if (
                gross_wt_initial >= 7500.0
            ):  # note: this technically creates a discontinuity
                avionics_wt = 163.0
            if (
                gross_wt_initial >= 11000.0
            ):  # note: this technically creates a discontinuity
                avionics_wt = 340.0

        if PAX >= 20.0 and PAX < 30.0:
            avionics_wt = 400.0
        elif PAX >= 30.0 and PAX <= 50.0:
            avionics_wt = 500.0
        elif PAX > 50.0:
            avionics_wt = 600.0
        if PAX > 100.0:
            avionics_wt = 2.8 * PAX + 1010.0
        if ~(
            -1e-5 < inputs[Aircraft.Avionics.MASS] < 1e-5
        ):  # note: this technically creates a discontinuity !WILL NOT CHANGE
            avionics_wt = inputs[Aircraft.Avionics.MASS] * GRAV_ENGLISH_LBM

        air_conditioning_wt = 5.0

        if gross_wt_initial > 3500.0:  # note: this technically creates a discontinuity
            air_conditioning_wt = (
                inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]
                * (1.5 + p_diff_fus)
                * (0.358 * fus_len * cabin_width**2) ** 0.5
            )

        SSUM = wing_area + htail_area + vtail_area
        icing_wt = 22.7 * (SSUM**0.5) - 385.0

        if smooth:
            pass
        else:
            if icing_wt < 0.0:  # note: this technically creates a discontinuity
                icing_wt = 0.0
        if ~(
            -1e-5 < inputs[Aircraft.AntiIcing.MASS] < 1e-5
        ):  # note: this technically creates a discontinuity !WILL NOT CHANGE
            icing_wt = inputs[Aircraft.AntiIcing.MASS] * GRAV_ENGLISH_LBM

        aux_wt = 0.0

        if smooth:
            aux_wt = 3 * sig((gross_wt_initial - 3000) / 3000)

        else:
            if (
                gross_wt_initial > 3000.0
            ):  # note: this technically creates a discontinuity
                aux_wt = 3.0

        if PAX >= 9.0:
            aux_wt = 10.0
        if PAX > 19.0:
            aux_wt = 20.0
        if PAX > 74.0:
            aux_wt = 50.0

        CPX = 28.0 + 10.516 * (cabin_width - 5.667)

        if smooth:
            CPX = (
                28 * sig((28 - CPX) / 28)
                + CPX * sig((CPX - 28) / 28) * sig((62 - CPX) / 62)
                + 62 * sig((CPX - 62) / 62)
            )

        else:
            if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                CPX = 28.0
            if cabin_width > 8.90:  # note: this technically creates a discontinuity
                CPX = 62.0

        furnishing_wt = CPX * PAX + 310.0
        if PAX > 80:
            furnishing_wt = 118.4 * PAX - 4190.0
        if (
            gross_wt_initial <= 10000.0
        ):  # note: this technically creates a discontinuity #TODO: Doesn't occur in large single aisle
            furnishing_wt = 0.065 * gross_wt_initial - 59.0

        if smooth:
            pass
        else:
            if furnishing_wt <= 30.0:  # note: this technically creates a discontinuity
                furnishing_wt = 30.0
        if ~(
            -1e-5 < inputs[Aircraft.Furnishings.MASS] < 1e-5
        ):  # note: this technically creates a discontinuity #WONT CHANGE
            furnishing_wt = inputs[Aircraft.Furnishings.MASS] * GRAV_ENGLISH_LBM
        fixed_equip_wt = (
            APU_wt
            + instrument_wt
            + hydraulic_wt
            + electrical_wt
            + avionics_wt
            + air_conditioning_wt
            + icing_wt
            + aux_wt
            + furnishing_wt
            + subsystems_wt
        )

        outputs[Aircraft.Design.FIXED_EQUIPMENT_MASS] = fixed_equip_wt / \
            GRAV_ENGLISH_LBM

        num_flight_attendants = 0.0
        if PAX >= 20.0:
            num_flight_attendants = 1.0
        if PAX >= 51.0:
            num_flight_attendants = 2.0
        if PAX >= 101.0:
            num_flight_attendants = 3.0
        if PAX >= 151.0:
            num_flight_attendants = 4.0
        if PAX >= 201.0:
            num_flight_attendants = 5.0
        if PAX >= 251.0:
            num_flight_attendants = 6.0

        # note: the average weight of a pilot was calulated using the following equation:
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

        crew_bag_wt = 25.0 * num_pilots
        if PAX >= 20.0:
            crew_bag_wt = 10.0 * (num_pilots + num_flight_attendants) + 25.0
        if PAX >= 40.0:
            crew_bag_wt = (
                20.0 * (num_flight_attendants + num_pilots) + 25.0 * num_pilots
            )

        if engine_type is GASPEngineType.TURBOJET:
            oil_per_eng_wt = 0.0054 * Fn_SLS + 12.0
        elif engine_type is GASPEngineType.TURBOSHAFT or engine_type is GASPEngineType.TURBOPROP:
            oil_per_eng_wt = 0.0124 * Fn_SLS + 14
        # else:
        #     oil_per_eng_wt = 0.062 * (Fn_SLS - 100) + 11
        else:
            # Other engine types are currently not supported in Aviary
            oil_per_eng_wt = 0

        oil_wt = num_engines * oil_per_eng_wt

        lavatories = 0.0
        if PAX > 25.0:
            lavatories = 1.0
        if PAX >= 51.0:
            lavatories = 2.0
        if PAX >= 101.0:
            lavatories = 3.0
        if PAX >= 151.0:
            lavatories = 4.0
        if PAX >= 201.0:
            lavatories = 5.0
        if PAX >= 251.0:
            lavatories = 6.0

        service_wt = 0.0
        if PAX > 9.0:
            service_wt = inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER] * PAX * \
                GRAV_ENGLISH_LBM + 16.0 * lavatories

        water_wt = 0.0
        if PAX > 19.0:
            water_wt = inputs[Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT] * \
                (PAX + num_pilots + num_flight_attendants) * GRAV_ENGLISH_LBM

        emergency_wt = 0.0
        if PAX > 5.0:
            emergency_wt = 10.0
        if PAX > 9.0:
            emergency_wt = 15.0
        if PAX >= 35.0:
            emergency_wt = 25.0 * num_flight_attendants + 15.0
        if ~(-1e-5 < inputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] < 1e-5):
            emergency_wt = inputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] * \
                GRAV_ENGLISH_LBM

        catering_wt = 0.0
        if PAX > 19.0:
            catering_wt = inputs[Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER] * \
                PAX * GRAV_ENGLISH_LBM

        trapped_fuel_wt = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] * \
            (wing_area**0.5) * fuel_vol_frac / 0.430
        if (
            fuel_vol_frac <= 0.075
        ):  # note: this technically creates a discontinuity # won't change
            trapped_fuel_wt = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] * \
                0.18 * (wing_area**0.5)

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
        )

        outputs[Aircraft.Design.FIXED_USEFUL_LOAD] = useful_wt / \
            GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, partials):
        options = self.options['aviary_options']
        PAX = options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS, units='unitless')
        smooth = options.get_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, units='unitless')
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        num_engines = self.options['aviary_options'].get_val(
            Aircraft.Propulsion.TOTAL_NUM_ENGINES, units='unitless')
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        wingspan = inputs[Aircraft.Wing.SPAN]
        if options.get_val(Aircraft.LandingGear.FIXED_GEAR, units='unitless'):
            gear_type = 1
        else:
            gear_type = 0
        landing_gear_wt = inputs[Aircraft.LandingGear.TOTAL_MASS] * \
            GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        wing_area = inputs[Aircraft.Wing.AREA]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]

        engine_type = options.get_val(Aircraft.Engine.TYPE, units='unitless')[0]

        dAPU_wt_dmass_coeff_0 = 0.0
        if ~(
            -1e-5 < inputs[Aircraft.APU.MASS] < 1e-5
        ):  # note: this technically creates a discontinuity
            dAPU_wt_dmass_coeff_0 = GRAV_ENGLISH_LBM

        num_pilots = 1.0
        if PAX > 9.0:
            num_pilots = 2.0
        if engine_type is GASPEngineType.TURBOJET and PAX > 5.0:
            num_pilots = 2.0
        if PAX >= 251.0:
            num_pilots = 3.0

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

        gear_val = 1 - gear_type

        dhydraulic_wt_dmass_coeff_2 = control_wt
        dhydraulic_wt_dcontrol_wt = inputs[Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT]
        dhydraulic_wt_dmass_coeff_3 = landing_gear_wt * gear_val
        dhydraulic_wt_dlanding_gear_weight = inputs[Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT] * gear_val

        delectrical_wt_dgross_wt_initial = 0.0
        if PAX <= 12.0:
            delectrical_wt_dgross_wt_initial = 0.03217
        if num_engines == 1.0:
            delectrical_wt_dgross_wt_initial = 0.0078

        davionics_wt_dmass_coeff_4 = 0.0

        if smooth:

            davionics_wt_dgross_wt_initial = 0.0071076 * np.exp(
                0.0002 * gross_wt_initial
            )

        else:
            if (
                gross_wt_initial >= 3000.0
            ):  # note: this technically creates a discontinuity
                davionics_wt_dgross_wt_initial = 0.0
            if (
                gross_wt_initial >= 5500.0
            ):  # note: this technically creates a discontinuity
                davionics_wt_dgross_wt_initial = 0.0
            if (
                gross_wt_initial >= 7500.0
            ):  # note: this technically creates a discontinuity
                davionics_wt_dgross_wt_initial = 0.0
            if (
                gross_wt_initial >= 11000.0
            ):  # note: this technically creates a discontinuity
                davionics_wt_dgross_wt_initial = 0.0

        if PAX >= 20.0 and PAX < 30.0:
            davionics_wt_dgross_wt_initial = 0.0
        elif PAX >= 30.0 and PAX <= 50.0:
            davionics_wt_dgross_wt_initial = 0.0
        elif PAX > 50.0:
            davionics_wt_dgross_wt_initial = 0.0
        if PAX > 100.0:
            davionics_wt_dgross_wt_initial = 0.0
        if ~(
            -1e-5 < inputs[Aircraft.Avionics.MASS] < 1e-5
        ):  # note: this technically creates a discontinuity !WILL NOT CHANGE
            davionics_wt_dgross_wt_initial = 0.0
            davionics_wt_dmass_coeff_4 = GRAV_ENGLISH_LBM

        dair_conditioning_wt_dmass_coeff_5 = 0.0
        dair_conditioning_wt_dp_diff_fus = 0.0
        dair_conditioning_wt_dfus_len = 0.0
        dair_conditioning_wt_dcabin_width = 0.0
        if gross_wt_initial > 3500.0:  # note: this technically creates a discontinuity
            dair_conditioning_wt_dmass_coeff_5 = (1.5 + p_diff_fus) * (
                0.358 * fus_len * cabin_width**2
            ) ** 0.5
            dair_conditioning_wt_dp_diff_fus = (
                inputs[Aircraft.AirConditioning.MASS_COEFFICIENT] *
                (0.358 * fus_len * cabin_width**2) ** 0.5
            )
            dair_conditioning_wt_dfus_len = (
                0.5
                * inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]
                * (1.5 + p_diff_fus)
                * (0.358 * fus_len * cabin_width**2) ** -0.5
                * 0.358
                * cabin_width**2
            )
            dair_conditioning_wt_dcabin_width = (
                0.5
                * inputs[Aircraft.AirConditioning.MASS_COEFFICIENT]
                * (1.5 + p_diff_fus)
                * (0.358 * fus_len * cabin_width**2) ** -0.5
                * 2
                * 0.358
                * fus_len
                * cabin_width
            )

        SSUM = wing_area + htail_area + vtail_area
        icing_wt = 22.7 * (SSUM**0.5) - 385.0

        dicing_weight_dwing_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dhtail_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dvtail_area = 0.5 * 22.7 * (SSUM**-0.5)
        dicing_weight_dmass_coeff_6 = 0.0

        if smooth:
            pass
        else:
            if icing_wt < 0.0:  # note: this technically creates a discontinuity
                icing_wt = 0.0
                dicing_weight_dwing_area = 0.0
                dicing_weight_dhtail_area = 0.0
                dicing_weight_dvtail_area = 0.0
                dicing_weight_dmass_coeff_6 = 0.0
        if ~(
            -1e-5 < inputs[Aircraft.AntiIcing.MASS] < 1e-5
        ):  # note: this technically creates a discontinuity !WILL NOT CHANGE
            icing_wt = inputs[Aircraft.AntiIcing.MASS] * GRAV_ENGLISH_LBM
            dicing_weight_dwing_area = 0.0
            dicing_weight_dhtail_area = 0.0
            dicing_weight_dvtail_area = 0.0
            dicing_weight_dmass_coeff_6 = GRAV_ENGLISH_LBM

        if smooth:
            d_aux_wt_dgross_wt_initial = (
                3 * dsig((gross_wt_initial - 3000) / 3000) * 1 / 3000
            )
        else:
            if (
                gross_wt_initial > 3000.0
            ):  # note: this technically creates a discontinuity
                d_aux_wt_dgross_wt_initial = 0.0

        if PAX >= 9.0:
            d_aux_wt_dgross_wt_initial = 0.0
        if PAX > 19.0:
            d_aux_wt_dgross_wt_initial = 0.0
        if PAX > 74.0:
            d_aux_wt_dgross_wt_initial = 0.0

        CPX = 28.0 + 10.516 * (cabin_width - 5.667)
        dCPX_dcabin_width = 10.516

        if smooth:
            CPX_1 = (
                28 * sig((28 - CPX) / 28)
                + CPX * sig((CPX - 28) / 28) * sig((62 - CPX) / 62)
                + 62 * sig((CPX - 62) / 62)
            )

            dCPX_dcabin_width = (
                28 * dsig((28 - CPX) / 28) * -dCPX_dcabin_width
                + (
                    dCPX_dcabin_width * sig((CPX - 28) / 28)
                    + CPX * dsig((CPX - 28) / 28) * dCPX_dcabin_width
                )
                * sig((62 - CPX) / 62)
                + CPX
                * sig((CPX - 28) / 28)
                * dsig((62 - CPX) / 62)
                * -dCPX_dcabin_width
                + 62 * dsig((CPX - 62) / 62) * dCPX_dcabin_width
            )

            CPX = CPX_1
            dfurnishing_wt_dgross_wt_initial = 0.0
            dfurnishing_wt_dgross_wt_initial = 0.0
            dfurnishing_wt_dmass_coeff_7 = 0.0
        else:
            if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                CPX = 28.0
                dCPX_dcabin_width = 0.0
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dmass_coeff_7 = 0.0
            if cabin_width > 8.90:  # note: this technically creates a discontinuity
                CPX = 62.0
                dCPX_dcabin_width = 0.0
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dmass_coeff_7 = 0.0

        furnishing_wt = CPX * PAX + 310.0
        dfurnishing_wt_dgross_wt_initial = 0.0
        dfurnishing_wt_dmass_coeff_7 = 0.0
        dfurnishing_wt_dcabin_width = PAX * dCPX_dcabin_width
        if PAX > 80:
            furnishing_wt = 118.4 * PAX - 4190.0
            dfurnishing_wt_dgross_wt_initial = 0.0
            dfurnishing_wt_dcabin_width = 0.0
            dfurnishing_wt_dmass_coeff_7 = 0.0
        if (
            gross_wt_initial <= 10000.0
        ):  # note: this technically creates a discontinuity #TODO: Doesn't occur in large single aisle
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
            dfurnishing_wt_dgross_wt_initial = 0.065
            dfurnishing_wt_dcabin_width = 0.0
            dfurnishing_wt_dmass_coeff_7 = 0.0

        if smooth:
            pass
        else:
            if furnishing_wt <= 30.0:  # note: this technically creates a discontinuity
                furnishing_wt = 30.0
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dcabin_width = 0.0
                dfurnishing_wt_dmass_coeff_7 = 0.0
        if ~(
            -1e-5 < inputs[Aircraft.Furnishings.MASS] < 1e-5
        ):  # note: this technically creates a discontinuity #WONT CHANGE
            furnishing_wt = inputs[Aircraft.Furnishings.MASS] * GRAV_ENGLISH_LBM
            dfurnishing_wt_dmass_coeff_7 = GRAV_ENGLISH_LBM
            dfurnishing_wt_dcabin_width = 0.0
            dfurnishing_wt_dgross_wt_initial = 0.0

        dfixed_equip_mass_dmass_coeff_0 = dAPU_wt_dmass_coeff_0 / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dmass_coeff_1 = dinstrument_wt_dmass_coeff_1 / GRAV_ENGLISH_LBM
        dfixed_equip_wt_dgross_wt_initial = (
            dinstrument_wt_dgross_wt_initial
            + delectrical_wt_dgross_wt_initial
            + dfurnishing_wt_dgross_wt_initial
            + davionics_wt_dgross_wt_initial
            + d_aux_wt_dgross_wt_initial
            + dfurnishing_wt_dcabin_width
        )
        dfixed_equip_mass_dfus_len = (
            dinstrument_wt_dfus_len + dair_conditioning_wt_dfus_len
        ) / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dwingspan = dinstrument_wt_dwingspan / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dmass_coeff_2 = dhydraulic_wt_dmass_coeff_2 / GRAV_ENGLISH_LBM
        dfixed_equip_wt_dcontrol_wt = dhydraulic_wt_dcontrol_wt
        dfixed_equip_mass_dmass_coeff_3 = dhydraulic_wt_dmass_coeff_3 / GRAV_ENGLISH_LBM
        dfixed_equip_wt_dlanding_gear_weight = dhydraulic_wt_dlanding_gear_weight
        dfixed_equip_mass_dmass_coeff_4 = davionics_wt_dmass_coeff_4 / GRAV_ENGLISH_LBM

        dfixed_equip_mass_dmass_coeff_5 = dair_conditioning_wt_dmass_coeff_5 / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dp_diff_fus = dair_conditioning_wt_dp_diff_fus / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dcabin_width = dair_conditioning_wt_dcabin_width / GRAV_ENGLISH_LBM

        dfixed_equip_mass_dwing_area = dicing_weight_dwing_area / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dhtail_area = dicing_weight_dhtail_area / \
            GRAV_ENGLISH_LBM
        dfixed_equip_mass_dvtail_area = dicing_weight_dvtail_area / \
            GRAV_ENGLISH_LBM
        dfixed_equip_mass_dmass_coeff_6 = dicing_weight_dmass_coeff_6 / GRAV_ENGLISH_LBM
        dfixed_equip_mass_dmass_coeff_7 = dfurnishing_wt_dmass_coeff_7 / GRAV_ENGLISH_LBM

        num_flight_attendants = 0.0
        if PAX >= 20.0:
            num_flight_attendants = 1.0
        if PAX >= 51.0:
            num_flight_attendants = 2.0
        if PAX >= 101.0:
            num_flight_attendants = 3.0
        if PAX >= 151.0:
            num_flight_attendants = 4.0
        if PAX >= 201.0:
            num_flight_attendants = 5.0
        if PAX >= 251.0:
            num_flight_attendants = 6.0

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
            dwater_wt_dmass_coeff_9 = (
                PAX + num_pilots + num_flight_attendants) * GRAV_ENGLISH_LBM

        demergency_wt_dmass_coeff_10 = 0.0
        if ~(-1e-5 < inputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] < 1e-5):
            demergency_wt_dmass_coeff_10 = GRAV_ENGLISH_LBM

        dcatering_wt_dmass_coeff_11 = 0.0
        if PAX > 19.0:
            dcatering_wt_dmass_coeff_11 = PAX * GRAV_ENGLISH_LBM

        dtrapped_fuel_wt_dmass_coeff_12 = (wing_area**0.5) * fuel_vol_frac / 0.430
        dtrapped_fuel_wt_dwing_area = (
            0.5 * inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] *
            (wing_area**-0.5) * fuel_vol_frac / 0.430
        )
        dtrapped_fuel_wt_dfuel_vol_frac = (
            inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] *
            (wing_area**0.5) / 0.430
        )

        if (
            fuel_vol_frac <= 0.075
        ):  # note: this technically creates a discontinuity # won't change
            dtrapped_fuel_wt_dmass_coeff_12 = 0.18 * (wing_area**0.5)
            dtrapped_fuel_wt_dwing_area = (
                0.5 * inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] *
                0.18 * (wing_area**-0.5)
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
        partials[Aircraft.Design.FIXED_USEFUL_LOAD,
                 Aircraft.Engine.SCALED_SLS_THRUST] = duseful_mass_dFn_SLS
        partials[Aircraft.Design.FIXED_USEFUL_LOAD,
                 Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER] = duseful_mass_dmass_coeff_8
        partials[Aircraft.Design.FIXED_USEFUL_LOAD,
                 Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT] = duseful_mass_dmass_coeff_9
        partials[Aircraft.Design.FIXED_USEFUL_LOAD,
                 Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] = duseful_mass_dmass_coeff_10
        partials[Aircraft.Design.FIXED_USEFUL_LOAD,
                 Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER] = duseful_mass_dmass_coeff_11
        partials[Aircraft.Design.FIXED_USEFUL_LOAD,
                 Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] = duseful_mass_dmass_coeff_12

        partials[Aircraft.Design.FIXED_USEFUL_LOAD,
                 Aircraft.Wing.AREA] = duseful_mass_dwing_area
        partials[
            Aircraft.Design.FIXED_USEFUL_LOAD, Aircraft.Fuel.WING_FUEL_FRACTION
        ] = duseful_mass_dfuel_vol_frac

        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.APU.MASS] = dfixed_equip_mass_dmass_coeff_0
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Instruments.MASS_COEFFICIENT] = dfixed_equip_mass_dmass_coeff_1
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT] = dfixed_equip_mass_dmass_coeff_2
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT] = dfixed_equip_mass_dmass_coeff_3
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Avionics.MASS] = dfixed_equip_mass_dmass_coeff_4
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.AirConditioning.MASS_COEFFICIENT] = dfixed_equip_mass_dmass_coeff_5
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.AntiIcing.MASS] = dfixed_equip_mass_dmass_coeff_6
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Furnishings.MASS] = dfixed_equip_mass_dmass_coeff_7
        partials[
            Aircraft.Design.FIXED_EQUIPMENT_MASS, Mission.Design.GROSS_MASS
        ] = dfixed_equip_wt_dgross_wt_initial
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Fuselage.LENGTH] = dfixed_equip_mass_dfus_len
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Wing.SPAN] = dfixed_equip_mass_dwingspan
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Controls.TOTAL_MASS] = dfixed_equip_wt_dcontrol_wt
        partials[
            Aircraft.Design.FIXED_EQUIPMENT_MASS, Aircraft.LandingGear.TOTAL_MASS
        ] = dfixed_equip_wt_dlanding_gear_weight

        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Fuselage.PRESSURE_DIFFERENTIAL] = dfixed_equip_mass_dp_diff_fus
        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Fuselage.AVG_DIAMETER] = dfixed_equip_mass_dcabin_width

        partials[Aircraft.Design.FIXED_EQUIPMENT_MASS,
                 Aircraft.Wing.AREA] = dfixed_equip_mass_dwing_area
        partials[
            Aircraft.Design.FIXED_EQUIPMENT_MASS, Aircraft.HorizontalTail.AREA
        ] = dfixed_equip_mass_dhtail_area
        partials[
            Aircraft.Design.FIXED_EQUIPMENT_MASS, Aircraft.VerticalTail.AREA
        ] = dfixed_equip_mass_dvtail_area
