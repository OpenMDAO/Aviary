import numpy as np
import openmdao.api as om
from openmdao.components.ks_comp import KSfunction

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_ENGLISH
from aviary.utils.functions import dSigmoidXdx, sigmoidX
from aviary.variable_info.enums import FlapType
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class MassParameters(om.ExplicitComponent):
    """
    Computation of various parameters (such as correction factor for the use of
    non optimum material, reduction in bending moment factor for strut braced wing,
    landing gear location factor, engine position factor, and wing chord half sweep
    angle).
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.NUM_FUSELAGE_ENGINES)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Wing.SWEEP, units='rad')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        self.add_input(
            'max_mach',
            val=0.9,
            units='unitless',
            desc='EMM0: maximum operating Mach number',
        )

        add_aviary_input(self, Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, units='unitless')

        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_LOCATION, units='unitless')

        add_aviary_output(self, Aircraft.Wing.MATERIAL_FACTOR, units='unitless')
        self.add_output(
            'c_strut_braced',
            val=0,
            units='unitless',
            desc='SKSTR: reduction in bending moment factor for strut braced wing',
        )
        self.add_output(
            'c_gear_loc',
            val=0,
            units='unitless',
            desc='SKGEAR: landing gear location factor',
        )
        add_aviary_output(
            self,
            Aircraft.Engine.POSITION_FACTOR,
            shape=num_engine_type,
            units='unitless',
        )
        self.add_output('half_sweep', val=0, units='rad', desc='SWC2: wing chord half sweep angle')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Wing.MATERIAL_FACTOR,
            [
                Aircraft.Wing.SPAN,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.ASPECT_RATIO,
            ],
        )
        self.declare_partials(Aircraft.Engine.POSITION_FACTOR, ['max_mach'])
        self.declare_partials(
            'half_sweep',
            [
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.ASPECT_RATIO,
            ],
        )

        self.declare_partials(
            'c_strut_braced',
            [Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, Aircraft.Wing.SPAN],
        )

        self.declare_partials('c_gear_loc', Aircraft.LandingGear.MAIN_GEAR_LOCATION)

    def compute(self, inputs, outputs):
        sweep_c4 = inputs[Aircraft.Wing.SWEEP]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        wingspan = inputs[Aircraft.Wing.SPAN]
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        max_mach = inputs['max_mach']
        strut_x = inputs[Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS]
        loc_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_LOCATION]

        tan_half_sweep = np.tan(sweep_c4) - (1.0 - taper_ratio) / (1.0 + taper_ratio) / AR

        half_sweep = np.arctan(tan_half_sweep)
        cos_half_sweep = np.cos(half_sweep)
        struct_span = wingspan / cos_half_sweep
        c_material = 1.0 + 2.5 / (struct_span**0.5)
        c_strut_braced = 1.0 - strut_x**2

        not_fuselage_mounted = self.options[Aircraft.Engine.NUM_FUSELAGE_ENGINES] == 0

        # note: c_gear_loc doesn't actually depend on any of the inputs... perhaps use a
        # set_input_defaults call to declare this at a higher level
        c_gear_loc = 1.0
        if self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]:
            # smooth transition for c_gear_loc from 0.95 to 1 when gear_location varies
            # between 0 and 1% of span
            c_gear_loc = 0.95 * sigmoidX(loc_main_gear, 0.005, -0.01 / 320.0) + 1 * sigmoidX(
                loc_main_gear, 0.005, 0.01 / 320.0
            )
        else:
            if loc_main_gear == 0:
                c_gear_loc = 0.95

        # why always use sigmoid function?
        c_eng_pos = 1.0 * sigmoidX(max_mach, 0.75, -1.0 / 320.0) + 1.05 * sigmoidX(
            max_mach, 0.75, 1.0 / 320.0
        )
        if not_fuselage_mounted and num_engines == 2 or num_engines == 3:
            c_eng_pos = 0.98 * sigmoidX(max_mach, 0.75, -1.0 / 320.0) + 0.95 * sigmoidX(
                max_mach, 0.75, 1.0 / 320.0
            )
        if not_fuselage_mounted and num_engines == 4:
            c_eng_pos = 0.95 * sigmoidX(max_mach, 0.75, -1.0 / 320.0) + 0.9 * sigmoidX(
                max_mach, 0.75, 1.0 / 320.0
            )

        outputs[Aircraft.Wing.MATERIAL_FACTOR] = c_material
        outputs['c_strut_braced'] = c_strut_braced
        outputs['c_gear_loc'] = c_gear_loc
        outputs[Aircraft.Engine.POSITION_FACTOR] = c_eng_pos
        outputs['half_sweep'] = half_sweep

    def compute_partials(self, inputs, J):
        sweep_c4 = inputs[Aircraft.Wing.SWEEP]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        wingspan = inputs[Aircraft.Wing.SPAN]
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        max_mach = inputs['max_mach']
        strut_x = inputs[Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS]
        loc_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_LOCATION]

        tan_half_sweep = np.tan(sweep_c4) - (1.0 - taper_ratio) / (1.0 + taper_ratio) / AR
        half_sweep = np.arctan(tan_half_sweep)
        cos_half_sweep = np.cos(half_sweep)
        struct_span = wingspan / cos_half_sweep

        not_fuselage_mounted = self.options[Aircraft.Engine.NUM_FUSELAGE_ENGINES] == 0

        dTanHS_dSC4 = 1 / np.cos(sweep_c4) ** 2
        dTanHS_TR = (
            -(1 / AR) * ((1 + taper_ratio) * (-1) - (1 - taper_ratio)) / (1 + taper_ratio) ** 2
        )
        dTanHS_dAR = (1.0 - taper_ratio) / (1.0 + taper_ratio) / AR**2

        J[Aircraft.Wing.MATERIAL_FACTOR, Aircraft.Wing.SPAN] = (
            -0.5 * 2.5 / struct_span ** (1.5) * (1 / cos_half_sweep)
        )
        J[Aircraft.Wing.MATERIAL_FACTOR, Aircraft.Wing.SWEEP] = (
            -0.5
            * 2.5
            / struct_span ** (1.5)
            * (-wingspan / cos_half_sweep**2)
            * (-np.sin(half_sweep))
            / (tan_half_sweep**2 + 1)
            * dTanHS_dSC4
        )
        J[Aircraft.Wing.MATERIAL_FACTOR, Aircraft.Wing.TAPER_RATIO] = (
            -0.5
            * 2.5
            / struct_span ** (1.5)
            * (-wingspan / cos_half_sweep**2)
            * (-np.sin(half_sweep))
            / (tan_half_sweep**2 + 1)
            * dTanHS_TR
        )
        J[Aircraft.Wing.MATERIAL_FACTOR, Aircraft.Wing.ASPECT_RATIO] = (
            -0.5
            * 2.5
            / struct_span ** (1.5)
            * (-wingspan / cos_half_sweep**2)
            * (-np.sin(half_sweep))
            / (tan_half_sweep**2 + 1)
            * dTanHS_dAR
        )

        J[Aircraft.Engine.POSITION_FACTOR, 'max_mach'] = -dSigmoidXdx(
            max_mach, 0.75, 1 / 320.0
        ) + 1.05 * dSigmoidXdx(max_mach, 0.75, 1 / 320.0)
        if not_fuselage_mounted and num_engines == 2 or num_engines == 3:
            J[Aircraft.Engine.POSITION_FACTOR, 'max_mach'] = -0.98 * dSigmoidXdx(
                max_mach, 0.75, 1 / 320.0
            ) + 0.95 * dSigmoidXdx(max_mach, 0.75, 1 / 320.0)
        if not_fuselage_mounted and num_engines == 4:
            J[Aircraft.Engine.POSITION_FACTOR, 'max_mach'] = -0.95 * dSigmoidXdx(
                max_mach, 0.75, 1 / 320.0
            ) + 0.9 * dSigmoidXdx(max_mach, 0.75, 1 / 320.0)

        J['half_sweep', Aircraft.Wing.SWEEP] = 1 / (tan_half_sweep**2 + 1) * dTanHS_dSC4
        J['half_sweep', Aircraft.Wing.TAPER_RATIO] = 1 / (tan_half_sweep**2 + 1) * dTanHS_TR
        J['half_sweep', Aircraft.Wing.ASPECT_RATIO] = 1 / (tan_half_sweep**2 + 1) * dTanHS_dAR

        J['c_strut_braced', Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS] = -2 * strut_x

        if self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]:
            J['c_gear_loc', Aircraft.LandingGear.MAIN_GEAR_LOCATION] = 0.95 * (-100) * dSigmoidXdx(
                loc_main_gear, 0.005, 0.01 / 320.0
            ) + 1 * (100) * dSigmoidXdx(loc_main_gear, 0.005, 0.01 / 320.0)


class PayloadGroup(om.ExplicitComponent):
    """Computation of maximum payload that the aircraft is being asked to carry."""

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, units='lbm')
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.Design.CARGO_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.Design.MAX_CARGO_MASS, units='lbm')

        add_aviary_output(self, Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm')
        add_aviary_output(self, Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, units='lbm')

        self.add_output('payload_mass_des', val=0, units='lbm', desc='WPLDES: design payload')
        self.add_output(
            'payload_mass_max',
            val=0,
            units='lbm',
            desc='WPLMAX: maximum payload that the aircraft is being asked to carry'
            ' (design payload + cargo)',
        )

    def setup_partials(self):
        self.declare_partials(
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
            [Aircraft.CrewPayload.CARGO_MASS],
            val=1.0,
        )
        self.declare_partials('payload_mass_des', [Aircraft.CrewPayload.Design.CARGO_MASS], val=1.0)
        self.declare_partials(
            'payload_mass_max', [Aircraft.CrewPayload.Design.MAX_CARGO_MASS], val=1.0
        )

    def compute(self, inputs, outputs):
        pax_mass, _ = self.options[Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS]
        pax = self.options[Aircraft.CrewPayload.NUM_PASSENGERS]
        pax_des = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        cargo_mass = inputs[Aircraft.CrewPayload.CARGO_MASS]
        cargo_mass_des = inputs[Aircraft.CrewPayload.Design.CARGO_MASS]
        cargo_mass_max = inputs[Aircraft.CrewPayload.Design.MAX_CARGO_MASS]

        outputs[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS] = pax_mass * pax
        outputs['payload_mass_des'] = pax_mass * pax_des + cargo_mass_des
        outputs['payload_mass_max'] = pax_mass * pax_des + cargo_mass_max
        outputs[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS] = pax_mass * pax + cargo_mass


class ElectricAugmentationMass(om.ExplicitComponent):
    """Computation of electrical augmentation system mass."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)

    def setup(self):
        self.add_input(
            'motor_power',
            val=200,
            units='kW',
            desc='MOTRKW: power of augmentation motor',
        )
        self.add_input(
            'motor_voltage',
            val=50,
            units='V',
            desc='VOLTS: voltage of augmentation system',
        )
        self.add_input(
            'max_amp_per_wire',
            val=50,
            units='A',
            desc='AMPSPW: maximum amperage of each cable in augmentation system',
        )
        self.add_input(
            'safety_factor',
            val=1.33,
            units='unitless',
            desc='REDUNCY: cable mass redundancy/safety factor',
        )

        add_aviary_input(self, Aircraft.Electrical.HYBRID_CABLE_LENGTH, units='ft')

        self.add_input(
            'wire_area',
            val=0.0015,
            units='ft**2',
            desc='ACSWIRE: cross sectional area of electrical augmentation system wire',
        )
        self.add_input(
            'rho_wire',
            val=1,
            units='lbm/ft**3',
            desc='DENWIRE: density of wire for electrical augmentation system',
        )
        self.add_input(
            'battery_energy',
            val=1,
            units='MJ',
            desc='EBATT: energy coming from the battery',
        )
        self.add_input(
            'motor_eff',
            val=1,
            units='unitless',
            desc='EFF_MTR: efficiency of electrical augmentation motor',
        )
        self.add_input(
            'inverter_eff',
            val=1,
            units='unitless',
            desc='EFF_INV: efficiency of electrical augmentation inverter/controller',
        )
        self.add_input(
            'transmission_eff',
            val=1,
            units='unitless',
            desc='EFF_TRN: efficiency of electrical augmentation system power transmission',
        )
        self.add_input(
            'battery_eff',
            val=1,
            units='unitless',
            desc='EFF_BAT: efficiency of electrical augmentation battery storage',
        )
        self.add_input(
            'rho_battery',
            val=200,
            units='MJ/lb',
            desc='ENGYDEN: energy density of electrical augmentation system battery',
        )
        self.add_input(
            'motor_spec_mass',
            val=10,
            units='hp/lbm',
            desc='SWT_MTR: specific mass of electrical augmentation motor',
        )
        self.add_input(
            'inverter_spec_mass',
            val=10,
            units='kW/lbm',
            desc='SWT_INV: specific mass of electrical augmentation inverter',
        )
        self.add_input(
            'TMS_spec_mass',
            val=10,
            units='lbm/kW',
            desc='SWT_TMS: specific mass of thermal managements system',
        )

        self.add_output(
            'aug_mass',
            val=0,
            units='lbm',
            desc='WEAUG: mass of electrical augmentation system',
        )

        self.declare_partials('aug_mass', '*')

    def compute(self, inputs, outputs):
        motor_power = inputs['motor_power']
        motor_voltage = inputs['motor_voltage']
        max_amp_per_wire = inputs['max_amp_per_wire']
        safety_factor = inputs['safety_factor']
        cable_len = inputs[Aircraft.Electrical.HYBRID_CABLE_LENGTH]
        wire_area = inputs['wire_area']
        rho_wire = inputs['rho_wire']
        battery_energy = inputs['battery_energy']
        motor_eff = inputs['motor_eff']
        inverter_eff = inputs['inverter_eff']
        transmission_eff = inputs['transmission_eff']
        battery_eff = inputs['battery_eff']
        rho_battery = inputs['rho_battery']
        motor_spec_wt = inputs['motor_spec_mass'] / GRAV_ENGLISH_LBM
        inverter_spec_wt = inputs['inverter_spec_mass'] / GRAV_ENGLISH_LBM
        TMS_spec_wt = inputs['TMS_spec_mass'] * GRAV_ENGLISH_LBM
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]

        motor_current = 1000.0 * motor_power / motor_voltage
        num_wires = motor_current / max_amp_per_wire
        cable_wt = (
            1.15 * safety_factor * num_wires * cable_len * wire_area * rho_wire * GRAV_ENGLISH_LBM
        )
        actual_battery_energy = battery_energy / (
            motor_eff * inverter_eff * transmission_eff * battery_eff
        )
        battery_wt = actual_battery_energy / rho_battery
        motor_wt = motor_power / 0.746 / motor_spec_wt
        inverter_wt = motor_power / inverter_spec_wt
        TMS_wt = TMS_spec_wt * motor_power

        aug_wt = (
            battery_wt
            + cable_wt
            + num_engines * inverter_wt
            + num_engines * motor_wt
            + num_engines * TMS_wt
        )

        outputs['aug_mass'] = aug_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        motor_power = inputs['motor_power']
        motor_voltage = inputs['motor_voltage']
        max_amp_per_wire = inputs['max_amp_per_wire']
        safety_factor = inputs['safety_factor']
        cable_len = inputs[Aircraft.Electrical.HYBRID_CABLE_LENGTH]
        wire_area = inputs['wire_area']
        rho_wire = inputs['rho_wire']
        battery_energy = inputs['battery_energy']
        motor_eff = inputs['motor_eff']
        inverter_eff = inputs['inverter_eff']
        transmission_eff = inputs['transmission_eff']
        battery_eff = inputs['battery_eff']
        rho_battery = inputs['rho_battery']
        motor_spec_wt = inputs['motor_spec_mass'] / GRAV_ENGLISH_LBM
        inverter_spec_wt = inputs['inverter_spec_mass'] / GRAV_ENGLISH_LBM
        TMS_spec_wt = inputs['TMS_spec_mass'] * GRAV_ENGLISH_LBM
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]

        motor_current = 1000.0 * motor_power / motor_voltage
        num_wires = motor_current / max_amp_per_wire
        actual_battery_energy = battery_energy / (
            motor_eff * inverter_eff * transmission_eff * battery_eff
        )

        dCableWt_dMotorPower = (
            1.15
            * safety_factor
            * cable_len
            * wire_area
            * rho_wire
            * GRAV_ENGLISH_LBM
            * 1000.0
            / motor_voltage
            / max_amp_per_wire
        )

        dInverterWt_dMotorPower = 1 / inverter_spec_wt
        dMotorWt_dMotorPower = 1 / 0.746 / motor_spec_wt
        dTMSwt_dMotorPower = TMS_spec_wt

        J['aug_mass', 'motor_power'] = (
            dCableWt_dMotorPower
            + num_engines * dInverterWt_dMotorPower
            + num_engines * dMotorWt_dMotorPower
            + num_engines * dTMSwt_dMotorPower
        ) / GRAV_ENGLISH_LBM
        J['aug_mass', 'motor_voltage'] = (
            -1.15
            * safety_factor
            * cable_len
            * wire_area
            * rho_wire
            * GRAV_ENGLISH_LBM
            / max_amp_per_wire
            * 1000.0
            * motor_power
            / motor_voltage**2
            / GRAV_ENGLISH_LBM
        )
        J['aug_mass', 'max_amp_per_wire'] = (
            -1.15
            * safety_factor
            * cable_len
            * wire_area
            * rho_wire
            * GRAV_ENGLISH_LBM
            * motor_current
            / max_amp_per_wire**2
            / GRAV_ENGLISH_LBM
        )
        J['aug_mass', 'safety_factor'] = 1.15 * num_wires * cable_len * wire_area * rho_wire
        J['aug_mass', Aircraft.Electrical.HYBRID_CABLE_LENGTH] = (
            1.15 * safety_factor * num_wires * wire_area * rho_wire
        )
        J['aug_mass', 'wire_area'] = 1.15 * safety_factor * num_wires * cable_len * rho_wire
        J['aug_mass', 'rho_wire'] = 1.15 * safety_factor * num_wires * cable_len * wire_area
        J['aug_mass', 'battery_energy'] = (
            1
            / (motor_eff * inverter_eff * transmission_eff * battery_eff)
            / rho_battery
            / GRAV_ENGLISH_LBM
        )
        J['aug_mass', 'motor_eff'] = (
            -battery_energy
            / (motor_eff**2 * inverter_eff * transmission_eff * battery_eff)
            / rho_battery
            / GRAV_ENGLISH_LBM
        )
        J['aug_mass', 'inverter_eff'] = (
            -battery_energy
            / (motor_eff * inverter_eff**2 * transmission_eff * battery_eff)
            / rho_battery
            / GRAV_ENGLISH_LBM
        )
        J['aug_mass', 'transmission_eff'] = (
            -battery_energy
            / (motor_eff * inverter_eff * transmission_eff**2 * battery_eff)
            / rho_battery
            / GRAV_ENGLISH_LBM
        )
        J['aug_mass', 'battery_eff'] = (
            -battery_energy
            / (motor_eff * inverter_eff * transmission_eff * battery_eff**2)
            / rho_battery
            / GRAV_ENGLISH_LBM
        )
        J['aug_mass', 'rho_battery'] = -actual_battery_energy / rho_battery**2 / GRAV_ENGLISH_LBM
        J['aug_mass', 'motor_spec_mass'] = (
            -num_engines * motor_power / 0.746 / motor_spec_wt**2 / GRAV_ENGLISH_LBM**2
        )
        J['aug_mass', 'inverter_spec_mass'] = (
            -num_engines * motor_power / inverter_spec_wt**2 / GRAV_ENGLISH_LBM**2
        )
        J['aug_mass', 'TMS_spec_mass'] = num_engines * motor_power


class EngineMass(om.ExplicitComponent):
    """
    Computation of total engine mass, nacelle mass, pylon mass, total engine POD mass,
    additional engine mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Electrical.HAS_HYBRID_SYSTEM)
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.ADDITIONAL_MASS_FRACTION)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        total_num_wing_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES]

        add_aviary_input(
            self, Aircraft.Engine.MASS_SPECIFIC, shape=num_engine_type, units='lbm/lbf'
        )
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )
        add_aviary_input(
            self,
            Aircraft.Nacelle.MASS_SPECIFIC,
            shape=num_engine_type,
            units='lbm/ft**2',
        )
        add_aviary_input(self, Aircraft.Nacelle.SURFACE_AREA, shape=num_engine_type, units='ft**2')
        add_aviary_input(
            self, Aircraft.Engine.PYLON_FACTOR, shape=num_engine_type, units='unitless'
        )
        add_aviary_input(self, Aircraft.Engine.MASS_SCALER, shape=num_engine_type, units='unitless')
        add_aviary_input(self, Aircraft.Propulsion.MISC_MASS_SCALER, units='unitless')

        if total_num_wing_engines > 1:
            add_aviary_input(
                self,
                Aircraft.Engine.WING_LOCATIONS,
                shape=int(total_num_wing_engines / 2),
                units='unitless',
            )
        else:
            add_aviary_input(self, Aircraft.Engine.WING_LOCATIONS, units='unitless')

        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')

        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_LOCATION, units='unitless')

        has_hybrid_system = self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]

        if has_hybrid_system:
            self.add_input(
                'aug_mass',
                val=400,
                units='lbm',
                desc='WEAUG: mass of electrical augmentation system',
            )

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_ENGINE_MASS, units='lbm')
        add_aviary_output(self, Aircraft.Nacelle.MASS, shape=num_engine_type)
        self.add_output(
            'pylon_mass',
            units='lbm',
            desc='WPYLON: mass of each pylon',
            val=np.zeros(num_engine_type),
        )
        add_aviary_output(self, Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, units='lbm', desc='WPES')
        add_aviary_output(self, Aircraft.Engine.ADDITIONAL_MASS, shape=num_engine_type, units='lbm')
        self.add_output(
            'eng_comb_mass',
            val=0,
            units='lbm',
            desc='WPSTAR: combined mass of dry engine and engine installation,'
            ' includes mass of electrical augmentation system',
        )
        self.add_output(
            'wing_mounted_mass',
            val=0,
            units='lbm',
            desc='WM: mass of gear and engine, basically everything mounted on the wing',
        )

        # for multiengine implementation needs this to always be available
        self.add_input(
            'prop_mass',
            # val=np.full(num_engine_type, 0.000000001),
            val=np.zeros(num_engine_type),
            units='lbm',
            desc='WPROP1: mass of one propeller',
        )

        self.add_output('prop_mass_all', val=0, units='lbm', desc='WPROP: mass of all propellers')

    def setup_partials(self):
        has_hybrid_system = self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]

        self.declare_partials('prop_mass_all', ['prop_mass'])
        self.declare_partials('wing_mounted_mass', 'prop_mass')

        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Propulsion.TOTAL_ENGINE_MASS,
            [Aircraft.Engine.MASS_SPECIFIC, Aircraft.Engine.SCALED_SLS_THRUST],
        )

        self.declare_partials(
            Aircraft.Nacelle.MASS,
            [Aircraft.Nacelle.MASS_SPECIFIC, Aircraft.Nacelle.SURFACE_AREA],
            rows=shape,
            cols=shape,
            val=1.0,
        )

        self.declare_partials(
            'pylon_mass',
            [
                Aircraft.Nacelle.MASS_SPECIFIC,
                Aircraft.Nacelle.SURFACE_AREA,
                Aircraft.Engine.PYLON_FACTOR,
                Aircraft.Engine.MASS_SPECIFIC,
                Aircraft.Engine.SCALED_SLS_THRUST,
            ],
            rows=shape,
            cols=shape,
            val=1.0,
        )

        self.declare_partials(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS,
            [
                Aircraft.Nacelle.MASS_SPECIFIC,
                Aircraft.Nacelle.SURFACE_AREA,
                Aircraft.Engine.PYLON_FACTOR,
                Aircraft.Engine.MASS_SPECIFIC,
                Aircraft.Engine.SCALED_SLS_THRUST,
            ],
        )
        self.declare_partials(
            Aircraft.Engine.ADDITIONAL_MASS,
            [Aircraft.Engine.MASS_SPECIFIC, Aircraft.Engine.SCALED_SLS_THRUST],
            rows=shape,
            cols=shape,
            val=1.0,
        )

        self.declare_partials(
            'wing_mounted_mass',
            [
                Aircraft.Engine.WING_LOCATIONS,
                Aircraft.Engine.MASS_SPECIFIC,
                Aircraft.Engine.SCALED_SLS_THRUST,
                Aircraft.Nacelle.MASS_SPECIFIC,
                Aircraft.Nacelle.SURFACE_AREA,
                Aircraft.Engine.PYLON_FACTOR,
                Aircraft.LandingGear.MAIN_GEAR_MASS,
                Aircraft.LandingGear.MAIN_GEAR_LOCATION,
            ],
        )
        if not has_hybrid_system:
            self.declare_partials(
                'eng_comb_mass',
                [
                    Aircraft.Engine.MASS_SCALER,
                    Aircraft.Propulsion.MISC_MASS_SCALER,
                    Aircraft.Engine.MASS_SPECIFIC,
                    Aircraft.Engine.SCALED_SLS_THRUST,
                ],
            )
        else:
            self.declare_partials(
                'eng_comb_mass',
                [
                    Aircraft.Engine.MASS_SCALER,
                    Aircraft.Propulsion.MISC_MASS_SCALER,
                    Aircraft.Engine.MASS_SPECIFIC,
                    Aircraft.Engine.SCALED_SLS_THRUST,
                    'aug_mass',
                ],
            )

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        num_engine_type = len(num_engines)
        c_instl = self.options[Aircraft.Engine.ADDITIONAL_MASS_FRACTION]

        eng_spec_wt = inputs[Aircraft.Engine.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        spec_nacelle_wt = inputs[Aircraft.Nacelle.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        nacelle_area = inputs[Aircraft.Nacelle.SURFACE_AREA]
        pylon_fac = inputs[Aircraft.Engine.PYLON_FACTOR]
        CK5 = inputs[Aircraft.Engine.MASS_SCALER]
        CK7 = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]
        eng_span_frac = inputs[Aircraft.Engine.WING_LOCATIONS]
        main_gear_wt = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS] * GRAV_ENGLISH_LBM
        loc_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_LOCATION]

        dry_wt_eng = eng_spec_wt * Fn_SLS
        dry_wt_eng_all = sum(dry_wt_eng * num_engines)
        nacelle_wt = spec_nacelle_wt * nacelle_area
        pylon_wt = pylon_fac * ((dry_wt_eng + nacelle_wt) ** 0.736)
        pod_wt = nacelle_wt + pylon_wt
        # sec_wt_all = sum((nacelle_wt + pylon_wt) * num_engines)
        # In GASP, WPEI = SKPEI * (WEP + ENP*WTGB), even though WTGB = 0.
        eng_instl_wt = c_instl * dry_wt_eng
        eng_instl_wt_all = sum(eng_instl_wt * num_engines)

        outputs[Aircraft.Propulsion.TOTAL_ENGINE_MASS] = dry_wt_eng_all / GRAV_ENGLISH_LBM
        outputs[Aircraft.Nacelle.MASS] = nacelle_wt / GRAV_ENGLISH_LBM
        outputs['pylon_mass'] = pylon_wt / GRAV_ENGLISH_LBM
        outputs[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS] = (
            sum(pod_wt * num_engines) / GRAV_ENGLISH_LBM
        )
        outputs[Aircraft.Engine.ADDITIONAL_MASS] = eng_instl_wt / GRAV_ENGLISH_LBM
        # In GASP, WPSTAR=CK5*WEP+CK7*WPEI+WPROP+WTGB*ENP, even though the last two terms are 0.
        outputs['eng_comb_mass'] = (
            sum(CK5 * dry_wt_eng * num_engines) + CK7 * eng_instl_wt_all
        ) / GRAV_ENGLISH_LBM

        if self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]:
            aug_wt = inputs['aug_mass'] * GRAV_ENGLISH_LBM
            outputs['eng_comb_mass'] = (
                sum(CK5 * dry_wt_eng * num_engines) + CK7 * eng_instl_wt_all + aug_wt
            ) / GRAV_ENGLISH_LBM

        # prop_wt = np.zeros(num_engine_type)
        # prop_idx = np.where(self.options[Aircraft.Engine.HAS_PROPELLERS))
        # prop_wt[prop_idx] = inputs['prop_mass'] * GRAV_ENGLISH_LBM
        prop_wt = inputs['prop_mass'] * GRAV_ENGLISH_LBM
        outputs['prop_mass_all'] = sum(num_engines * prop_wt) / GRAV_ENGLISH_LBM

        span_frac_factor = eng_span_frac / (eng_span_frac + 0.001)
        # sum span_frac_factor for each engine type
        span_frac_factor_sum = np.zeros(num_engine_type, dtype=Fn_SLS.dtype)
        idx = 0
        for i in range(num_engine_type):
            # fmt: off
            span_frac_factor_sum[i] = sum(span_frac_factor[idx : idx + num_engines[i]])
            # fmt: on
            idx = idx + num_engines[i]

        # In GASP,
        # WM = YP/(YP+.001)*(WEP+WPEI+WPES+WPROP+ENP*WTGB)
        #      + WMG*YMG/(YMG+.001)
        #      + WCMIN*YC/(YC+.001)
        outputs['wing_mounted_mass'] = (
            sum(span_frac_factor_sum * (dry_wt_eng + eng_instl_wt + pod_wt + prop_wt) * num_engines)
            + main_gear_wt * loc_main_gear / (loc_main_gear + 0.001)
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        num_engine_type = len(num_engines)
        c_instl = self.options[Aircraft.Engine.ADDITIONAL_MASS_FRACTION]

        eng_spec_wt = inputs[Aircraft.Engine.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        spec_nacelle_wt = inputs[Aircraft.Nacelle.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        nacelle_area = inputs[Aircraft.Nacelle.SURFACE_AREA]
        pylon_fac = inputs[Aircraft.Engine.PYLON_FACTOR]
        CK5 = inputs[Aircraft.Engine.MASS_SCALER]
        CK7 = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]
        eng_span_frac = inputs[Aircraft.Engine.WING_LOCATIONS]
        main_gear_wt = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS] * GRAV_ENGLISH_LBM
        loc_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_LOCATION]

        dDWEA_dESW = num_engines * Fn_SLS
        dDWEA_dFNSLS = num_engines * eng_spec_wt
        dNW_dNWS = nacelle_area
        dPW_dNWS = (
            0.736
            * pylon_fac
            * (eng_spec_wt * Fn_SLS + spec_nacelle_wt * nacelle_area) ** (-0.264)
            * nacelle_area
        )
        dNW_dNSA = spec_nacelle_wt
        dPW_dNSA = (
            0.736
            * pylon_fac
            * (eng_spec_wt * Fn_SLS + spec_nacelle_wt * nacelle_area) ** (-0.264)
            * spec_nacelle_wt
        )
        dPW_dPF = (eng_spec_wt * Fn_SLS + spec_nacelle_wt * nacelle_area) ** 0.736
        dPW_dEWS = (
            0.736
            * pylon_fac
            * (eng_spec_wt * Fn_SLS + spec_nacelle_wt * nacelle_area) ** (-0.264)
            * Fn_SLS
        )
        dPW_dSLST = (
            pylon_fac
            * 0.736
            * (eng_spec_wt * Fn_SLS + spec_nacelle_wt * nacelle_area) ** (-0.264)
            * eng_spec_wt
        )

        J[Aircraft.Propulsion.TOTAL_ENGINE_MASS, Aircraft.Engine.MASS_SPECIFIC] = dDWEA_dESW
        J[Aircraft.Propulsion.TOTAL_ENGINE_MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            dDWEA_dFNSLS / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.MASS_SPECIFIC] = dNW_dNWS
        J['pylon_mass', Aircraft.Nacelle.MASS_SPECIFIC] = dPW_dNWS
        J[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, Aircraft.Nacelle.MASS_SPECIFIC] = (
            num_engines * (dNW_dNWS + dPW_dNWS)
        )
        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.SURFACE_AREA] = dNW_dNSA / GRAV_ENGLISH_LBM
        J['pylon_mass', Aircraft.Nacelle.SURFACE_AREA] = dPW_dNSA / GRAV_ENGLISH_LBM
        J[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, Aircraft.Nacelle.SURFACE_AREA] = (
            num_engines * (dNW_dNSA + dPW_dNSA) / GRAV_ENGLISH_LBM
        )
        J['pylon_mass', Aircraft.Engine.PYLON_FACTOR] = dPW_dPF / GRAV_ENGLISH_LBM
        J[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, Aircraft.Engine.PYLON_FACTOR] = (
            num_engines * dPW_dPF / GRAV_ENGLISH_LBM
        )
        J['pylon_mass', Aircraft.Engine.MASS_SPECIFIC] = dPW_dEWS
        J[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, Aircraft.Engine.MASS_SPECIFIC] = (
            num_engines * dPW_dEWS
        )
        J['pylon_mass', Aircraft.Engine.SCALED_SLS_THRUST] = dPW_dSLST / GRAV_ENGLISH_LBM
        J[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            num_engines * dPW_dSLST / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Engine.ADDITIONAL_MASS, Aircraft.Engine.MASS_SPECIFIC] = c_instl * Fn_SLS
        J[Aircraft.Engine.ADDITIONAL_MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            c_instl * eng_spec_wt / GRAV_ENGLISH_LBM
        )

        J['eng_comb_mass', Aircraft.Engine.MASS_SCALER] = (
            eng_spec_wt * Fn_SLS * num_engines / GRAV_ENGLISH_LBM
        )
        J['eng_comb_mass', Aircraft.Propulsion.MISC_MASS_SCALER] = (
            sum(c_instl * eng_spec_wt * Fn_SLS * num_engines) / GRAV_ENGLISH_LBM
        )
        J['eng_comb_mass', Aircraft.Engine.MASS_SPECIFIC] = (
            (CK5 + CK7 * c_instl) * num_engines * Fn_SLS
        )
        J['eng_comb_mass', Aircraft.Engine.SCALED_SLS_THRUST] = (
            (CK5 + CK7 * c_instl) * num_engines * eng_spec_wt / GRAV_ENGLISH_LBM
        )

        dry_wt_eng = eng_spec_wt * Fn_SLS
        nacelle_wt = spec_nacelle_wt * nacelle_area
        pylon_wt = pylon_fac * ((dry_wt_eng + nacelle_wt) ** 0.736)
        pod_wt = nacelle_wt + pylon_wt
        eng_instl_wt = c_instl * dry_wt_eng

        # prop_idx = np.where(self.options[Aircraft.Engine.HAS_PROPELLERS))
        prop_wt = inputs['prop_mass'] * GRAV_ENGLISH_LBM
        # prop_wt_all = sum(num_engines * prop_wt) / GRAV_ENGLISH_LBM

        J['prop_mass_all', 'prop_mass'] = num_engines

        dPylonWt_dFnSLS = pylon_fac * 0.736 * (dry_wt_eng + nacelle_wt) ** (0.736 - 1) * eng_spec_wt
        dPylonWt_dEngSpecWt = (
            pylon_fac * 0.736 * ((dry_wt_eng + nacelle_wt) ** (0.736 - 1) * Fn_SLS)
        )

        span_frac_factor = eng_span_frac / (eng_span_frac + 0.001)
        # sum span_frac_factor for each engine type
        span_frac_factor_sum = np.zeros(num_engine_type, dtype=Fn_SLS.dtype)
        wing_mass_deriv = np.zeros(len(span_frac_factor), dtype=Fn_SLS.dtype)
        idx = 0
        # wing_mass_vec = (eng_spec_wt * Fn_SLS * (1 + c_instl) +
        #                 sec_wt + prop_wt) * num_engines
        wing_mass_vec = (dry_wt_eng + eng_instl_wt + pod_wt + prop_wt) * num_engines
        for i in range(num_engine_type):
            span_frac_factor_sum[i] = sum(span_frac_factor[idx : idx + num_engines[i]])
            wing_mass_deriv[idx : idx + num_engines[i]] = wing_mass_vec[i]
            idx = idx + num_engines[i]

        J['wing_mounted_mass', Aircraft.Engine.WING_LOCATIONS] = (
            0.001 / (eng_span_frac + 0.001) ** 2 * wing_mass_deriv / GRAV_ENGLISH_LBM
        )

        J['wing_mounted_mass', Aircraft.Engine.MASS_SPECIFIC] = (
            span_frac_factor_sum * (Fn_SLS + c_instl * Fn_SLS + dPylonWt_dEngSpecWt) * num_engines
        )

        J['wing_mounted_mass', Aircraft.Engine.SCALED_SLS_THRUST] = (
            span_frac_factor_sum
            * (
                num_engines * eng_spec_wt
                + c_instl * num_engines * eng_spec_wt
                + dPylonWt_dFnSLS * num_engines
            )
            / GRAV_ENGLISH_LBM
        )

        J['wing_mounted_mass', Aircraft.Nacelle.MASS_SPECIFIC] = (
            span_frac_factor_sum
            * num_engines
            * (
                nacelle_area
                + pylon_fac
                * 0.736
                * ((dry_wt_eng + spec_nacelle_wt * nacelle_area) ** (0.736 - 1) * nacelle_area)
            )
        )

        J['wing_mounted_mass', Aircraft.Nacelle.SURFACE_AREA] = (
            span_frac_factor_sum
            * num_engines
            * (
                spec_nacelle_wt
                + pylon_fac
                * 0.736
                * ((dry_wt_eng + spec_nacelle_wt * nacelle_area) ** (0.736 - 1) * spec_nacelle_wt)
            )
        ) / GRAV_ENGLISH_LBM

        J['wing_mounted_mass', Aircraft.Engine.PYLON_FACTOR] = (
            span_frac_factor_sum
            * num_engines
            * ((dry_wt_eng + spec_nacelle_wt * nacelle_area) ** 0.736)
        ) / GRAV_ENGLISH_LBM

        J['wing_mounted_mass', Aircraft.LandingGear.MAIN_GEAR_MASS] = loc_main_gear / (
            loc_main_gear + 0.001
        )

        J['wing_mounted_mass', Aircraft.LandingGear.MAIN_GEAR_LOCATION] = (
            main_gear_wt
            / GRAV_ENGLISH_LBM
            * ((loc_main_gear + 0.001) - loc_main_gear)
            / (loc_main_gear + 0.001) ** 2
        )

        J['wing_mounted_mass', 'prop_mass'] = span_frac_factor_sum * num_engines

        if self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]:
            J['eng_comb_mass', 'aug_mass'] = 1


class TailMass(om.ExplicitComponent):
    """Computation of horizontal tail mass and vertical tail mass."""

    def setup(self):
        add_aviary_input(self, Aircraft.VerticalTail.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.SWEEP, units='rad')
        add_aviary_input(self, Aircraft.VerticalTail.SPAN, units='ft')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.HorizontalTail.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.HorizontalTail.SPAN, units='ft')
        add_aviary_input(self, Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.HorizontalTail.AREA, units='ft**2')
        self.add_input('min_dive_vel', val=200, units='kn', desc='VDMIN: dive velocity')
        add_aviary_input(self, Aircraft.HorizontalTail.MOMENT_ARM, units='ft')
        add_aviary_input(self, Aircraft.HorizontalTail.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.ROOT_CHORD, units='ft')
        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.VerticalTail.MOMENT_ARM, units='ft')
        add_aviary_input(self, Aircraft.VerticalTail.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.ROOT_CHORD, units='ft')

        self.add_output(
            'loc_MAC_vtail',
            val=0,
            units='ft',
            desc='XVMAC: location of mean aerodynamic chord on the vertical tail',
        )
        add_aviary_output(self, Aircraft.HorizontalTail.MASS, units='lbm')
        add_aviary_output(self, Aircraft.VerticalTail.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            'loc_MAC_vtail',
            [
                Aircraft.VerticalTail.SPAN,
                Aircraft.VerticalTail.TAPER_RATIO,
                Aircraft.VerticalTail.ASPECT_RATIO,
                Aircraft.VerticalTail.SWEEP,
            ],
        )
        self.declare_partials(
            Aircraft.HorizontalTail.MASS,
            [
                Aircraft.HorizontalTail.AREA,
                Mission.Design.GROSS_MASS,
                Aircraft.HorizontalTail.MASS_COEFFICIENT,
                Aircraft.Fuselage.LENGTH,
                Aircraft.HorizontalTail.SPAN,
                Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER,
                Aircraft.HorizontalTail.TAPER_RATIO,
                'min_dive_vel',
                Aircraft.HorizontalTail.MOMENT_ARM,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.HorizontalTail.ROOT_CHORD,
            ],
        )
        self.declare_partials(
            Aircraft.VerticalTail.MASS,
            [
                Mission.Design.GROSS_MASS,
                Aircraft.VerticalTail.MASS_COEFFICIENT,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Wing.SPAN,
                Aircraft.VerticalTail.SPAN,
                Aircraft.VerticalTail.TAPER_RATIO,
                Aircraft.HorizontalTail.MASS_COEFFICIENT,
                Aircraft.HorizontalTail.SPAN,
                Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER,
                Aircraft.HorizontalTail.TAPER_RATIO,
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
                Aircraft.VerticalTail.AREA,
                'min_dive_vel',
                Aircraft.VerticalTail.MOMENT_ARM,
                Aircraft.VerticalTail.THICKNESS_TO_CHORD,
                Aircraft.VerticalTail.ROOT_CHORD,
            ],
        )

    def compute(self, inputs, outputs):
        taper_ratio_vtail = inputs[Aircraft.VerticalTail.TAPER_RATIO]
        AR_vtail = inputs[Aircraft.VerticalTail.ASPECT_RATIO]
        quarter_sweep_tail = inputs[Aircraft.VerticalTail.SWEEP]
        span_vtail = inputs[Aircraft.VerticalTail.SPAN]
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        coef_htail = inputs[Aircraft.HorizontalTail.MASS_COEFFICIENT]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        span_htail = inputs[Aircraft.HorizontalTail.SPAN]
        hook_fac = inputs[Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER]
        taper_ratio_htail = inputs[Aircraft.HorizontalTail.TAPER_RATIO]
        coef_vtail = inputs[Aircraft.VerticalTail.MASS_COEFFICIENT]
        wingspan = inputs[Aircraft.Wing.SPAN]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        min_dive_vel = inputs['min_dive_vel']
        htail_mom_arm = inputs[Aircraft.HorizontalTail.MOMENT_ARM]
        tc_ratio_root_htail = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]
        root_chord_htail = inputs[Aircraft.HorizontalTail.ROOT_CHORD]
        htail_loc = inputs[Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]
        vtail_mom_arm = inputs[Aircraft.VerticalTail.MOMENT_ARM]
        tc_ratio_root_vtail = inputs[Aircraft.VerticalTail.THICKNESS_TO_CHORD]
        root_chord_vtail = inputs[Aircraft.VerticalTail.ROOT_CHORD]

        tan_sweep_vtail_LE = (1.0 - taper_ratio_vtail) / (
            1.0 + taper_ratio_vtail
        ) / AR_vtail / 2.0 + np.tan(quarter_sweep_tail)
        FH = (
            gross_wt_initial
            * coef_htail
            * fus_len
            * span_htail
            * hook_fac
            * (1.0 + 2.0 * taper_ratio_htail)
            / (1000000.0 * (1.0 + taper_ratio_htail))
        )
        FV = (
            gross_wt_initial
            * coef_vtail
            * (fus_len + wingspan)
            * span_vtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (1000000.0 * (1.0 + taper_ratio_vtail))
        ) / 2

        outputs['loc_MAC_vtail'] = (
            span_vtail
            * tan_sweep_vtail_LE
            * (1.0 + 2.0 * taper_ratio_vtail)
            / 3.0
            / (1.0 + taper_ratio_vtail)
        )
        outputs[Aircraft.HorizontalTail.MASS] = (
            350.0
            / GRAV_ENGLISH_LBM
            * (
                htail_area
                * FH
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
        )
        outputs[Aircraft.VerticalTail.MASS] = (
            380.0
            / GRAV_ENGLISH_LBM
            * (
                (FV + htail_loc * FH / 2.0)
                * vtail_area
                * np.log10(min_dive_vel)
                / (100.0 * vtail_mom_arm * tc_ratio_root_vtail * root_chord_vtail)
            )
            ** 0.54
        )

    def compute_partials(self, inputs, J):
        taper_ratio_vtail = inputs[Aircraft.VerticalTail.TAPER_RATIO]
        AR_vtail = inputs[Aircraft.VerticalTail.ASPECT_RATIO]
        quarter_sweep_tail = inputs[Aircraft.VerticalTail.SWEEP]
        span_vtail = inputs[Aircraft.VerticalTail.SPAN]
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        coef_htail = inputs[Aircraft.HorizontalTail.MASS_COEFFICIENT]
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        span_htail = inputs[Aircraft.HorizontalTail.SPAN]
        hook_fac = inputs[Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER]
        taper_ratio_htail = inputs[Aircraft.HorizontalTail.TAPER_RATIO]
        coef_vtail = inputs[Aircraft.VerticalTail.MASS_COEFFICIENT]
        wingspan = inputs[Aircraft.Wing.SPAN]
        htail_area = inputs[Aircraft.HorizontalTail.AREA]
        min_dive_vel = inputs['min_dive_vel']
        htail_mom_arm = inputs[Aircraft.HorizontalTail.MOMENT_ARM]
        tc_ratio_root_htail = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]
        root_chord_htail = inputs[Aircraft.HorizontalTail.ROOT_CHORD]
        htail_loc = inputs[Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION]
        vtail_area = inputs[Aircraft.VerticalTail.AREA]
        vtail_mom_arm = inputs[Aircraft.VerticalTail.MOMENT_ARM]
        tc_ratio_root_vtail = inputs[Aircraft.VerticalTail.THICKNESS_TO_CHORD]
        root_chord_vtail = inputs[Aircraft.VerticalTail.ROOT_CHORD]

        tan_sweep_vtail_LE = (1.0 - taper_ratio_vtail) / (
            1.0 + taper_ratio_vtail
        ) / AR_vtail / 2.0 + np.tan(quarter_sweep_tail)
        FH = (
            gross_wt_initial
            * coef_htail
            * fus_len
            * span_htail
            * hook_fac
            * (1.0 + 2.0 * taper_ratio_htail)
            / (1000000.0 * (1.0 + taper_ratio_htail))
        )
        FV = (
            gross_wt_initial
            * coef_vtail
            * (fus_len + wingspan)
            * span_vtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (1000000.0 * (1.0 + taper_ratio_vtail))
        ) / 2

        dTanSweepVtailLE_dTaperRatioVtail = (
            -((1.0 + taper_ratio_vtail) * AR_vtail * 2.0) - (1.0 - taper_ratio_vtail) * 2 * AR_vtail
        ) / ((1.0 + taper_ratio_vtail) * AR_vtail * 2.0) ** 2
        dTanSweepVtailLE_dARVtail = (
            -(1.0 - taper_ratio_vtail)
            / ((1.0 + taper_ratio_vtail) * AR_vtail * 2.0) ** 2
            * 2
            * (1.0 + taper_ratio_vtail)
        )
        dTanSweepVtailLE_dQuarterSweepTail = (1 / np.cos(quarter_sweep_tail)) ** 2

        dFH_dGrossWtInitial = (
            coef_htail
            * fus_len
            * span_htail
            * hook_fac
            * (1.0 + 2.0 * taper_ratio_htail)
            / (1000000.0 * (1.0 + taper_ratio_htail))
        )
        dFH_dCoefHtail = (
            gross_wt_initial
            * fus_len
            * span_htail
            * hook_fac
            * (1.0 + 2.0 * taper_ratio_htail)
            / (1000000.0 * (1.0 + taper_ratio_htail))
        )
        dFH_dFusLen = (
            gross_wt_initial
            * coef_htail
            * span_htail
            * hook_fac
            * (1.0 + 2.0 * taper_ratio_htail)
            / (1000000.0 * (1.0 + taper_ratio_htail))
        )
        dFH_dSpanHtail = (
            gross_wt_initial
            * coef_htail
            * fus_len
            * hook_fac
            * (1.0 + 2.0 * taper_ratio_htail)
            / (1000000.0 * (1.0 + taper_ratio_htail))
        )
        dFH_dHookFac = (
            gross_wt_initial
            * coef_htail
            * fus_len
            * span_htail
            * (1.0 + 2.0 * taper_ratio_htail)
            / (1000000.0 * (1.0 + taper_ratio_htail))
        )
        dFH_dTaperRatioHtail = (
            gross_wt_initial
            * coef_htail
            * fus_len
            * span_htail
            * hook_fac
            * (
                (1000000.0 * (1.0 + taper_ratio_htail)) * 2
                - (1.0 + 2.0 * taper_ratio_htail) * 1000000.0
            )
            / (1000000.0 * (1.0 + taper_ratio_htail)) ** 2
        )

        dFV_dGrossWtInitial = (
            coef_vtail
            * (fus_len + wingspan)
            * span_vtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (2 * 1000000.0 * (1.0 + taper_ratio_vtail))
        )
        dFV_dCoefVtail = (
            gross_wt_initial
            * (fus_len + wingspan)
            * span_vtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (2 * 1000000.0 * (1.0 + taper_ratio_vtail))
        )
        dFV_dFusLen = (
            gross_wt_initial
            * coef_vtail
            * span_vtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (2 * 1000000.0 * (1.0 + taper_ratio_vtail))
        )
        dFV_dWingspan = (
            gross_wt_initial
            * coef_vtail
            * span_vtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (2 * 1000000.0 * (1.0 + taper_ratio_vtail))
        )
        dFV_dSpanVtail = (
            gross_wt_initial
            * coef_vtail
            * (fus_len + wingspan)
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (2 * 1000000.0 * (1.0 + taper_ratio_vtail))
        )
        dFV_dTaperRatioVtail = (
            gross_wt_initial
            * coef_vtail
            * (fus_len + wingspan)
            * span_vtail
            * (
                (2 * 1000000.0 * (1.0 + taper_ratio_vtail)) * 2
                - (1.0 + 2.0 * taper_ratio_vtail) * 2 * 1000000.0
            )
            / (2 * 1000000.0 * (1.0 + taper_ratio_vtail)) ** 2
        )

        temp = ((3.0 * (1.0 + taper_ratio_vtail)) * 2 - (1.0 + 2.0 * taper_ratio_vtail) * 3) / (
            3.0 * (1.0 + taper_ratio_vtail)
        ) ** 2

        J['loc_MAC_vtail', Aircraft.VerticalTail.SPAN] = (
            tan_sweep_vtail_LE * (1.0 + 2.0 * taper_ratio_vtail) / 3.0 / (1.0 + taper_ratio_vtail)
        )
        J['loc_MAC_vtail', Aircraft.VerticalTail.TAPER_RATIO] = span_vtail * (
            dTanSweepVtailLE_dTaperRatioVtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (3.0 * (1.0 + taper_ratio_vtail))
            + tan_sweep_vtail_LE * temp
        )
        J['loc_MAC_vtail', Aircraft.VerticalTail.ASPECT_RATIO] = (
            dTanSweepVtailLE_dARVtail
            * span_vtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (3.0 * (1.0 + taper_ratio_vtail))
        )
        J['loc_MAC_vtail', Aircraft.VerticalTail.SWEEP] = (
            dTanSweepVtailLE_dQuarterSweepTail
            * span_vtail
            * (1.0 + 2.0 * taper_ratio_vtail)
            / (3.0 * (1.0 + taper_ratio_vtail))
        )

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.AREA] = (
            350.0
            / GRAV_ENGLISH_LBM
            * 0.54
            * (
                FH
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
            * htail_area ** (-0.46)
        )
        J[Aircraft.HorizontalTail.MASS, Mission.Design.GROSS_MASS] = (
            350.0
            * 0.54
            * (
                htail_area
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
            * FH ** (-0.46)
            * dFH_dGrossWtInitial
        )
        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.MASS_COEFFICIENT] = (
            350.0
            / GRAV_ENGLISH_LBM
            * 0.54
            * (
                htail_area
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
            * FH ** (-0.46)
            * dFH_dCoefHtail
        )
        J[Aircraft.HorizontalTail.MASS, Aircraft.Fuselage.LENGTH] = (
            350.0
            / GRAV_ENGLISH_LBM
            * 0.54
            * (
                htail_area
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
            * FH ** (-0.46)
            * dFH_dFusLen
        )
        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.SPAN] = (
            350.0
            / GRAV_ENGLISH_LBM
            * 0.54
            * (
                htail_area
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
            * FH ** (-0.46)
            * dFH_dSpanHtail
        )
        J[Aircraft.HorizontalTail.MASS, Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER] = (
            350.0
            / GRAV_ENGLISH_LBM
            * 0.54
            * (
                htail_area
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
            * FH ** (-0.46)
            * dFH_dHookFac
        )
        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.TAPER_RATIO] = (
            350.0
            / GRAV_ENGLISH_LBM
            * 0.54
            * (
                htail_area
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
            * FH ** (-0.46)
            * dFH_dTaperRatioHtail
        )
        J[Aircraft.HorizontalTail.MASS, 'min_dive_vel'] = (
            350.0
            / GRAV_ENGLISH_LBM
            * 0.54
            * (htail_area * FH / (100.0 * htail_mom_arm * tc_ratio_root_htail * root_chord_htail))
            ** 0.54
            * (np.log10(min_dive_vel)) ** (-0.46)
            / (min_dive_vel * np.log(10))
        )
        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.MOMENT_ARM] = (
            350.0
            / GRAV_ENGLISH_LBM
            * (
                htail_area
                * FH
                * np.log10(min_dive_vel)
                / (100.0 * tc_ratio_root_htail * root_chord_htail)
            )
            ** 0.54
            * (-0.54)
            * htail_mom_arm ** (-1.54)
        )
        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.THICKNESS_TO_CHORD] = (
            350.0
            / GRAV_ENGLISH_LBM
            * (
                htail_area
                * FH
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * root_chord_htail)
            )
            ** 0.54
            * (-0.54)
            * tc_ratio_root_htail ** (-1.54)
        )
        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.ROOT_CHORD] = (
            350.0
            / GRAV_ENGLISH_LBM
            * (
                htail_area
                * FH
                * np.log10(min_dive_vel)
                / (100.0 * htail_mom_arm * tc_ratio_root_htail)
            )
            ** 0.54
            * (-0.54)
            * root_chord_htail ** (-1.54)
        )

        temp = (
            380.0
            * (
                vtail_area
                * np.log10(min_dive_vel)
                / (100.0 * vtail_mom_arm * tc_ratio_root_vtail * root_chord_vtail)
            )
            ** 0.54
        )

        J[Aircraft.VerticalTail.MASS, Mission.Design.GROSS_MASS] = (
            temp
            * 0.54
            * (FV + htail_loc * FH / 2.0) ** (-0.46)
            * (dFV_dGrossWtInitial + htail_loc * dFH_dGrossWtInitial / 2)
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.MASS_COEFFICIENT] = (
            temp * 0.54 * (FV + htail_loc * FH / 2.0) ** (-0.46) * dFV_dCoefVtail / GRAV_ENGLISH_LBM
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.Fuselage.LENGTH] = (
            temp
            / GRAV_ENGLISH_LBM
            * 0.54
            * (FV + htail_loc * FH / 2.0) ** (-0.46)
            * (dFV_dFusLen + htail_loc * dFH_dFusLen / 2)
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.Wing.SPAN] = (
            temp * 0.54 * (FV + htail_loc * FH / 2.0) ** (-0.46) * dFV_dWingspan / GRAV_ENGLISH_LBM
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.SPAN] = (
            temp * 0.54 * (FV + htail_loc * FH / 2.0) ** (-0.46) * dFV_dSpanVtail / GRAV_ENGLISH_LBM
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.TAPER_RATIO] = (
            temp
            * 0.54
            * (FV + htail_loc * FH / 2.0) ** (-0.46)
            * dFV_dTaperRatioVtail
            / GRAV_ENGLISH_LBM
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.HorizontalTail.MASS_COEFFICIENT] = (
            temp
            / GRAV_ENGLISH_LBM
            * 0.54
            * (FV + htail_loc * FH / 2.0) ** (-0.46)
            * dFH_dCoefHtail
            * htail_loc
            / 2
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.HorizontalTail.SPAN] = (
            temp
            / GRAV_ENGLISH_LBM
            * 0.54
            * (FV + htail_loc * FH / 2.0) ** (-0.46)
            * dFH_dSpanHtail
            * htail_loc
            / 2
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER] = (
            temp
            / GRAV_ENGLISH_LBM
            * 0.54
            * (FV + htail_loc * FH / 2.0) ** (-0.46)
            * dFH_dHookFac
            * htail_loc
            / 2
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.HorizontalTail.TAPER_RATIO] = (
            temp
            / GRAV_ENGLISH_LBM
            * 0.54
            * (FV + htail_loc * FH / 2.0) ** (-0.46)
            * dFH_dTaperRatioHtail
            * htail_loc
            / 2
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION] = (
            380.0
            / GRAV_ENGLISH_LBM
            * (
                vtail_area
                * np.log10(min_dive_vel)
                / (100.0 * vtail_mom_arm * tc_ratio_root_vtail * root_chord_vtail)
            )
            ** 0.54
            * 0.54
            * (FV + htail_loc * FH / 2.0) ** (-0.46)
            * FH
            / 2
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.AREA] = (
            380.0
            / GRAV_ENGLISH_LBM
            * (
                (FV + htail_loc * FH / 2.0)
                * np.log10(min_dive_vel)
                / (100.0 * vtail_mom_arm * tc_ratio_root_vtail * root_chord_vtail)
            )
            ** 0.54
            * 0.54
            * vtail_area ** (-0.46)
        )
        J[Aircraft.VerticalTail.MASS, 'min_dive_vel'] = (
            380.0
            / GRAV_ENGLISH_LBM
            * (
                (FV + htail_loc * FH / 2.0)
                * vtail_area
                / (100.0 * vtail_mom_arm * tc_ratio_root_vtail * root_chord_vtail)
            )
            ** 0.54
            * 0.54
            * (np.log10(min_dive_vel)) ** (-0.46)
            / (min_dive_vel * np.log(10))
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.MOMENT_ARM] = (
            380.0
            / GRAV_ENGLISH_LBM
            * (
                (FV + htail_loc * FH / 2.0)
                * vtail_area
                * np.log10(min_dive_vel)
                / (100.0 * tc_ratio_root_vtail * root_chord_vtail)
            )
            ** 0.54
            * (-0.54)
            * vtail_mom_arm ** (-1.54)
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.THICKNESS_TO_CHORD] = (
            380.0
            / GRAV_ENGLISH_LBM
            * (
                (FV + htail_loc * FH / 2.0)
                * vtail_area
                * np.log10(min_dive_vel)
                / (100.0 * vtail_mom_arm * root_chord_vtail)
            )
            ** 0.54
            * (-0.54)
            * tc_ratio_root_vtail ** (-1.54)
        )
        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.ROOT_CHORD] = (
            380.0
            / GRAV_ENGLISH_LBM
            * (
                (FV + htail_loc * FH / 2.0)
                * vtail_area
                * np.log10(min_dive_vel)
                / (100.0 * vtail_mom_arm * tc_ratio_root_vtail)
            )
            ** 0.54
            * (-0.54)
            * root_chord_vtail ** (-1.54)
        )


class HighLiftMass(om.ExplicitComponent):
    """
    Computation of masses of the high lift devices, trailing edge devices,
    leading edge devices.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.FLAP_TYPE)
        add_aviary_option(self, Aircraft.Wing.NUM_FLAP_SEGMENTS)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SLAT_CHORD_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.FLAP_CHORD_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SLAT_SPAN_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.FLAP_SPAN_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Design.WING_LOADING, units='lbf/ft**2')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Wing.CENTER_CHORD, units='ft')
        add_aviary_input(self, Mission.Landing.LIFT_COEFFICIENT_MAX, units='unitless')
        self.add_input(
            'density',
            val=RHO_SEA_LEVEL_ENGLISH,
            units='slug/ft**3',
            desc='RHO: Density of air',
        )

        add_aviary_output(self, Aircraft.Wing.HIGH_LIFT_MASS, units='lbm')
        self.add_output(
            'flap_mass', val=0, units='lbm', desc='WFLAP: mass of trailing edge devices'
        )
        self.add_output('slat_mass', val=0, units='lbm', desc='WLED: mass of leading edge devices')

    def setup_partials(self):
        self.declare_partials(
            'slat_mass',
            [
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.CENTER_CHORD,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.SLAT_CHORD_RATIO,
                Aircraft.Wing.AREA,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.SLAT_SPAN_RATIO,
            ],
        )

        self.declare_partials(
            'flap_mass',
            [
                Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.CENTER_CHORD,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.FLAP_CHORD_RATIO,
                Aircraft.Wing.AREA,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.FLAP_SPAN_RATIO,
                Aircraft.Design.WING_LOADING,
                'density',
                Mission.Landing.LIFT_COEFFICIENT_MAX,
            ],
        )

        self.declare_partials(
            Aircraft.Wing.HIGH_LIFT_MASS,
            [
                Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.CENTER_CHORD,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.FLAP_CHORD_RATIO,
                Aircraft.Wing.SLAT_CHORD_RATIO,
                Aircraft.Wing.AREA,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.FLAP_SPAN_RATIO,
                Aircraft.Wing.SLAT_SPAN_RATIO,
                Aircraft.Design.WING_LOADING,
                'density',
                Mission.Landing.LIFT_COEFFICIENT_MAX,
            ],
        )

    def compute(self, inputs, outputs):
        flap_type = self.options[Aircraft.Wing.FLAP_TYPE]
        c_mass_trend_high_lift = inputs[Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT]
        wing_area = inputs[Aircraft.Wing.AREA]
        num_flaps = self.options[Aircraft.Wing.NUM_FLAP_SEGMENTS]
        slat_chord_ratio = inputs[Aircraft.Wing.SLAT_CHORD_RATIO]
        flap_chord_ratio = inputs[Aircraft.Wing.FLAP_CHORD_RATIO]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        flap_span_ratio = inputs[Aircraft.Wing.FLAP_SPAN_RATIO]
        slat_span_ratio = inputs[Aircraft.Wing.SLAT_SPAN_RATIO]
        wing_loading = inputs[Aircraft.Design.WING_LOADING]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        wingspan = inputs[Aircraft.Wing.SPAN]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        center_chord = inputs[Aircraft.Wing.CENTER_CHORD]
        CL_max_flaps_landing = inputs[Mission.Landing.LIFT_COEFFICIENT_MAX]
        RHO = inputs['density']

        body_to_span_ratio = (
            2.0
            * np.sqrt(tc_ratio_root * center_chord * (cabin_width - (tc_ratio_root * center_chord)))
            + 0.4
        ) / wingspan

        SFLAP = (
            flap_chord_ratio
            * wing_area
            / (1.0 + taper_ratio)
            * (flap_span_ratio - body_to_span_ratio)
            * (2.0 - ((1.0 - taper_ratio) * (flap_span_ratio + body_to_span_ratio)))
        )

        SLE = (
            slat_chord_ratio
            * wing_area
            / (1.0 + taper_ratio)
            * slat_span_ratio
            * (2.0 - ((1.0 - taper_ratio) * (0.99 + body_to_span_ratio)))
        )
        VSTALL = 0.5921 * np.sqrt(2.0 * wing_loading / (RHO * CL_max_flaps_landing))
        VFLAP = 1.8 * VSTALL

        # Slat Mass
        WLED = 3.28 * SLE**1.13
        outputs['slat_mass'] = WLED / GRAV_ENGLISH_LBM

        # Flap Mass
        if flap_type is FlapType.PLAIN:
            outputs['flap_mass'] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * SFLAP
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
        elif flap_type is FlapType.SPLIT:
            if VFLAP > 160:
                outputs['flap_mass'] = (
                    c_mass_trend_high_lift * SFLAP * (VFLAP**2.195) / 45180.0 / GRAV_ENGLISH_LBM
                )
            else:
                outputs['flap_mass'] = (
                    c_mass_trend_high_lift * SFLAP * 0.369 * VFLAP**0.2733 / GRAV_ENGLISH_LBM
                )

        elif (
            flap_type is FlapType.SINGLE_SLOTTED
            or flap_type is FlapType.DOUBLE_SLOTTED
            or flap_type is FlapType.TRIPLE_SLOTTED
        ):
            outputs['flap_mass'] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * SFLAP
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
        elif flap_type is FlapType.FOWLER or flap_type is FlapType.DOUBLE_SLOTTED_FOWLER:
            outputs['flap_mass'] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * SFLAP**1.19
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
        else:
            raise ValueError(flap_type + ' is not a valid flap type')

        outputs[Aircraft.Wing.HIGH_LIFT_MASS] = outputs['flap_mass'] + WLED / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        flap_type = self.options[Aircraft.Wing.FLAP_TYPE]
        c_mass_trend_high_lift = inputs[Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT]
        wing_area = inputs[Aircraft.Wing.AREA]
        num_flaps = self.options[Aircraft.Wing.NUM_FLAP_SEGMENTS]
        slat_chord_ratio = inputs[Aircraft.Wing.SLAT_CHORD_RATIO]
        flap_chord_ratio = inputs[Aircraft.Wing.FLAP_CHORD_RATIO]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        flap_span_ratio = inputs[Aircraft.Wing.FLAP_SPAN_RATIO]
        slat_span_ratio = inputs[Aircraft.Wing.SLAT_SPAN_RATIO]
        wing_loading = inputs[Aircraft.Design.WING_LOADING]
        tc_ratio_root = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        wingspan = inputs[Aircraft.Wing.SPAN]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        center_chord = inputs[Aircraft.Wing.CENTER_CHORD]
        CL_max_flaps_landing = inputs[Mission.Landing.LIFT_COEFFICIENT_MAX]
        RHO = inputs['density']

        u1 = tc_ratio_root * center_chord * (cabin_width - (tc_ratio_root * center_chord))
        body_to_span_ratio = (2 * np.sqrt(u1) + 0.4) / wingspan
        dBTSR_dwingspan = -(1 / wingspan) * body_to_span_ratio

        dBTSR_dTCRR = (
            (1 / wingspan)
            * (1 / np.sqrt(u1))
            * (center_chord * cabin_width - 2 * tc_ratio_root * center_chord**2)
        )

        dBTSR_dCC = (
            (1 / wingspan)
            * (1 / np.sqrt(u1))
            * (tc_ratio_root * cabin_width - 2 * tc_ratio_root**2 * center_chord)
        )

        dBTSR_dCW = (1 / wingspan) * (1 / np.sqrt(u1)) * (tc_ratio_root * center_chord)

        SFLAP = (
            flap_chord_ratio
            * wing_area
            / (1.0 + taper_ratio)
            * (flap_span_ratio - body_to_span_ratio)
            * (2 - ((1 - taper_ratio) * (flap_span_ratio + body_to_span_ratio)))
        )

        dSFLAP_dFCR = (
            wing_area
            / (1.0 + taper_ratio)
            * (flap_span_ratio - body_to_span_ratio)
            * (2 - ((1 - taper_ratio) * (flap_span_ratio + body_to_span_ratio)))
        )

        dSFLAP_dWA = (
            flap_chord_ratio
            / (1.0 + taper_ratio)
            * (flap_span_ratio - body_to_span_ratio)
            * (2 - ((1 - taper_ratio) * (flap_span_ratio + body_to_span_ratio)))
        )

        dSFLAP_dFSR = (
            flap_chord_ratio
            * wing_area
            * (2 / (1.0 + taper_ratio) * (1 - 2 * flap_span_ratio) + 2 * flap_span_ratio)
        )
        dSFLAP_dBTSR = (
            flap_chord_ratio
            * wing_area
            * (2 / (1.0 + taper_ratio) * (2 * body_to_span_ratio - 1) - 2 * body_to_span_ratio)
        )
        dSFLAP_dTR = (
            flap_chord_ratio
            * wing_area
            * -2
            / (1.0 + taper_ratio) ** 2
            * (flap_span_ratio - body_to_span_ratio)
            * (1 - flap_span_ratio - body_to_span_ratio)
        )

        SLE = (
            slat_chord_ratio
            * wing_area
            / (1 + taper_ratio)
            * slat_span_ratio
            * (2 - ((1 - taper_ratio) * (0.99 + body_to_span_ratio)))
        )
        dSLE_dSCR = (
            wing_area
            / (1 + taper_ratio)
            * slat_span_ratio
            * (2 - ((1 - taper_ratio) * (0.99 + body_to_span_ratio)))
        )

        dSLE_dWA = (
            slat_chord_ratio
            / (1 + taper_ratio)
            * slat_span_ratio
            * (2 - ((1 - taper_ratio) * (0.99 + body_to_span_ratio)))
        )

        dSLE_dSSR = (
            (slat_chord_ratio * wing_area)
            / (1 + taper_ratio)
            * (2 - ((1 - taper_ratio) * (0.99 + body_to_span_ratio)))
        )
        dSLE_dBTSR = (
            -(slat_chord_ratio * wing_area)
            / (1 + taper_ratio)
            * slat_span_ratio
            * (1 - taper_ratio)
        )
        dSLE_dTR = (
            (2 * slat_chord_ratio * wing_area * slat_span_ratio)
            / ((1 + taper_ratio) ** 2)
            * (0.99 + body_to_span_ratio - 1)
        )

        u2 = 2 * wing_loading / (RHO * CL_max_flaps_landing)
        VFLAP = 1.8 * (0.5921 * np.sqrt(u2))
        dVFLAP_dWL = (1.8 * 0.5921) * (1 / np.sqrt(u2)) * (1 / (RHO * CL_max_flaps_landing))
        dVFLAP_drho = (
            -(1.8 * 0.5921) * (1 / np.sqrt(u2)) * (wing_loading / (RHO**2 * CL_max_flaps_landing))
        )
        dVFLAP_dCMFL = (
            -(1.8 * 0.5921) * (1 / np.sqrt(u2)) * (wing_loading / (RHO * CL_max_flaps_landing**2))
        )

        # Slat Mass
        J['slat_mass', Aircraft.Wing.SLAT_CHORD_RATIO] = (
            3.28 * 1.13 * (SLE**0.13) * dSLE_dSCR / GRAV_ENGLISH_LBM
        )
        J['slat_mass', Aircraft.Wing.AREA] = 3.28 * 1.13 * (SLE**0.13) * dSLE_dWA / GRAV_ENGLISH_LBM
        J['slat_mass', Aircraft.Wing.SLAT_SPAN_RATIO] = (
            3.28 * 1.13 * (SLE**0.13) * dSLE_dSSR / GRAV_ENGLISH_LBM
        )
        J['slat_mass', Aircraft.Wing.TAPER_RATIO] = (
            3.28 * 1.13 * (SLE**0.13) * dSLE_dTR / GRAV_ENGLISH_LBM
        )
        J['slat_mass', Aircraft.Wing.SPAN] = (
            3.28 * 1.13 * (SLE**0.13) * dSLE_dBTSR * dBTSR_dwingspan / GRAV_ENGLISH_LBM
        )
        J['slat_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
            3.28 * 1.13 * (SLE**0.13) * dSLE_dBTSR * dBTSR_dTCRR / GRAV_ENGLISH_LBM
        )
        J['slat_mass', Aircraft.Wing.CENTER_CHORD] = (
            3.28 * 1.13 * (SLE**0.13) * dSLE_dBTSR * dBTSR_dCC / GRAV_ENGLISH_LBM
        )
        J['slat_mass', Aircraft.Fuselage.AVG_DIAMETER] = (
            3.28 * 1.13 * (SLE**0.13) * dSLE_dBTSR * dBTSR_dCW / GRAV_ENGLISH_LBM
        )

        # Flap Mass
        if flap_type is FlapType.PLAIN:
            # c_wt_trend_high_lift * (VFLAP/100.)**2*SFLAP*num_flaps**(-.5)
            J['flap_mass', Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT] = (
                (VFLAP / 100) ** 2 * SFLAP * num_flaps ** (-0.5) / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Design.WING_LOADING] = (
                c_mass_trend_high_lift
                * (2 * VFLAP / 100**2)
                * dVFLAP_dWL
                * SFLAP
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', 'density'] = (
                c_mass_trend_high_lift
                * (2 * VFLAP / 100**2)
                * dVFLAP_drho
                * SFLAP
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Mission.Landing.LIFT_COEFFICIENT_MAX] = (
                c_mass_trend_high_lift
                * (2 * VFLAP / 100**2)
                * dVFLAP_dCMFL
                * SFLAP
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.FLAP_CHORD_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100) ** 2
                * dSFLAP_dFCR
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.AREA] = (
                c_mass_trend_high_lift
                * (VFLAP / 100) ** 2
                * dSFLAP_dWA
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.FLAP_SPAN_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100) ** 2
                * dSFLAP_dFSR
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.TAPER_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100) ** 2
                * dSFLAP_dTR
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.SPAN] = (
                c_mass_trend_high_lift
                * (VFLAP / 100) ** 2
                * dSFLAP_dBTSR
                * dBTSR_dwingspan
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )

            J['flap_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
                c_mass_trend_high_lift
                * (VFLAP / 100) ** 2
                * dSFLAP_dBTSR
                * dBTSR_dTCRR
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )

            J['flap_mass', Aircraft.Wing.CENTER_CHORD] = (
                c_mass_trend_high_lift
                * (VFLAP / 100) ** 2
                * dSFLAP_dBTSR
                * dBTSR_dCC
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Fuselage.AVG_DIAMETER] = (
                c_mass_trend_high_lift
                * (VFLAP / 100) ** 2
                * dSFLAP_dBTSR
                * dBTSR_dCW
                * num_flaps ** (-0.5)
                / GRAV_ENGLISH_LBM
            )
        elif flap_type is FlapType.SPLIT:
            if VFLAP > 160:
                # c_wt_trend_high_lift*SFLAP*(VFLAP**2.195)/45180.
                J['flap_mass', Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT] = (
                    SFLAP * (VFLAP**2.195) / 45180.0 / GRAV_ENGLISH_LBM
                )
                J['flap_mass', Aircraft.Design.WING_LOADING] = (
                    c_mass_trend_high_lift
                    * SFLAP
                    * (2.195 * VFLAP**1.195 * dVFLAP_dWL)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )
                J['flap_mass', 'density'] = (
                    c_mass_trend_high_lift
                    * SFLAP
                    * (2.195 * VFLAP**1.195 * dVFLAP_drho)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Mission.Landing.LIFT_COEFFICIENT_MAX] = (
                    c_mass_trend_high_lift
                    * SFLAP
                    * (2.195 * VFLAP**1.195 * dVFLAP_dCMFL)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.FLAP_CHORD_RATIO] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dFCR
                    * (VFLAP**2.195)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.AREA] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dWA
                    * (VFLAP**2.195)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.FLAP_SPAN_RATIO] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dFSR
                    * (VFLAP**2.195)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.TAPER_RATIO] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dTR
                    * (VFLAP**2.195)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )
                J['flap_mass', Aircraft.Wing.SPAN] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dBTSR
                    * dBTSR_dwingspan
                    * (VFLAP**2.195)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dBTSR
                    * dBTSR_dTCRR
                    * (VFLAP**2.195)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.CENTER_CHORD] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dBTSR
                    * dBTSR_dCC
                    * (VFLAP**2.195)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Fuselage.AVG_DIAMETER] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dBTSR
                    * dBTSR_dCW
                    * (VFLAP**2.195)
                    / 45180.0
                    / GRAV_ENGLISH_LBM
                )
            else:
                # c_wt_trend_high_lift*SFLAP*0.369*VFLAP**0.2733
                J['flap_mass', Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT] = (
                    SFLAP * 0.369 * VFLAP**0.2733 / GRAV_ENGLISH_LBM
                )
                J['flap_mass', Aircraft.Design.WING_LOADING] = (
                    c_mass_trend_high_lift
                    * SFLAP
                    * 0.369
                    * (0.2733 * VFLAP ** (-0.7267) * dVFLAP_dWL)
                    / GRAV_ENGLISH_LBM
                )
                J['flap_mass', 'density'] = (
                    c_mass_trend_high_lift
                    * SFLAP
                    * 0.369
                    * (0.2733 * VFLAP ** (-0.7267) * dVFLAP_drho)
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Mission.Landing.LIFT_COEFFICIENT_MAX] = (
                    c_mass_trend_high_lift
                    * SFLAP
                    * 0.369
                    * (0.2733 * VFLAP ** (-0.7267) * dVFLAP_dCMFL)
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.FLAP_CHORD_RATIO] = (
                    c_mass_trend_high_lift * dSFLAP_dFCR * 0.369 * VFLAP**0.2733 / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.AREA] = (
                    c_mass_trend_high_lift * dSFLAP_dWA * 0.369 * VFLAP**0.2733 / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.FLAP_SPAN_RATIO] = (
                    c_mass_trend_high_lift * dSFLAP_dFSR * 0.369 * VFLAP**0.2733 / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.TAPER_RATIO] = (
                    c_mass_trend_high_lift * dSFLAP_dTR * 0.369 * VFLAP**0.2733 / GRAV_ENGLISH_LBM
                )
                J['flap_mass', Aircraft.Wing.SPAN] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dBTSR
                    * dBTSR_dwingspan
                    * 0.369
                    * VFLAP**0.2733
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dBTSR
                    * dBTSR_dTCRR
                    * 0.369
                    * VFLAP**0.2733
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Wing.CENTER_CHORD] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dBTSR
                    * dBTSR_dCC
                    * 0.369
                    * VFLAP**0.2733
                    / GRAV_ENGLISH_LBM
                )

                J['flap_mass', Aircraft.Fuselage.AVG_DIAMETER] = (
                    c_mass_trend_high_lift
                    * dSFLAP_dBTSR
                    * dBTSR_dCW
                    * 0.369
                    * VFLAP**0.2733
                    / GRAV_ENGLISH_LBM
                )

        elif (
            flap_type is FlapType.SINGLE_SLOTTED
            or flap_type is FlapType.DOUBLE_SLOTTED
            or flap_type is FlapType.TRIPLE_SLOTTED
        ):
            # c_wt_trend_high_lift*(VFLAP/100.)**2*SFLAP*num_flaps**.5
            J['flap_mass', Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT] = (
                (VFLAP / 100.0) ** 2 * SFLAP * num_flaps**0.5 / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Design.WING_LOADING] = (
                c_mass_trend_high_lift
                * (2 * VFLAP / 100**2)
                * dVFLAP_dWL
                * SFLAP
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', 'density'] = (
                c_mass_trend_high_lift
                * (2 * VFLAP / 100**2)
                * dVFLAP_drho
                * SFLAP
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Mission.Landing.LIFT_COEFFICIENT_MAX] = (
                c_mass_trend_high_lift
                * (2 * VFLAP / 100**2)
                * dVFLAP_dCMFL
                * SFLAP
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.FLAP_CHORD_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * dSFLAP_dFCR
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.AREA] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * dSFLAP_dWA
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.FLAP_SPAN_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * dSFLAP_dFSR
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.TAPER_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * dSFLAP_dTR
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.SPAN] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * dSFLAP_dBTSR
                * dBTSR_dwingspan
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )

            J['flap_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * dSFLAP_dBTSR
                * dBTSR_dTCRR
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )

            J['flap_mass', Aircraft.Wing.CENTER_CHORD] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * dSFLAP_dBTSR
                * dBTSR_dCC
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Fuselage.AVG_DIAMETER] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2
                * dSFLAP_dBTSR
                * dBTSR_dCW
                * num_flaps**0.5
                / GRAV_ENGLISH_LBM
            )
        elif flap_type is FlapType.FOWLER or flap_type is FlapType.DOUBLE_SLOTTED_FOWLER:
            # c_wt_trend_high_lift * (VFLAP/100.)**2.38*SFLAP**1.19/(num_flaps**.595)
            J['flap_mass', Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT] = (
                (VFLAP / 100.0) ** 2.38 * SFLAP**1.19 / (num_flaps**0.595) / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Design.WING_LOADING] = (
                c_mass_trend_high_lift
                * (2.38 * VFLAP**1.38 / 100.0**2.38)
                * dVFLAP_dWL
                * SFLAP**1.19
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', 'density'] = (
                c_mass_trend_high_lift
                * (2.38 * VFLAP**1.38 / 100.0**2.38)
                * dVFLAP_drho
                * SFLAP**1.19
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Mission.Landing.LIFT_COEFFICIENT_MAX] = (
                c_mass_trend_high_lift
                * (2.38 * VFLAP**1.38 / 100.0**2.38)
                * dVFLAP_dCMFL
                * SFLAP**1.19
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.FLAP_CHORD_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * (1.19 * SFLAP**0.19)
                * dSFLAP_dFCR
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.AREA] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * (1.19 * SFLAP**0.19)
                * dSFLAP_dWA
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.FLAP_SPAN_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * (1.19 * SFLAP**0.19)
                * dSFLAP_dFSR
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.TAPER_RATIO] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * (1.19 * SFLAP**0.19)
                * dSFLAP_dTR
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Wing.SPAN] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * (1.19 * SFLAP**0.19)
                * dSFLAP_dBTSR
                * dBTSR_dwingspan
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )

            J['flap_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * (1.19 * SFLAP**0.19)
                * dSFLAP_dBTSR
                * dBTSR_dTCRR
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )

            J['flap_mass', Aircraft.Wing.CENTER_CHORD] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * (1.19 * SFLAP**0.19)
                * dSFLAP_dBTSR
                * dBTSR_dCC
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )
            J['flap_mass', Aircraft.Fuselage.AVG_DIAMETER] = (
                c_mass_trend_high_lift
                * (VFLAP / 100.0) ** 2.38
                * (1.19 * SFLAP**0.19)
                * dSFLAP_dBTSR
                * dBTSR_dCW
                / (num_flaps**0.595)
                / GRAV_ENGLISH_LBM
            )

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT] = J[
            'flap_mass', Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT
        ]

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Design.WING_LOADING] = J[
            'flap_mass', Aircraft.Design.WING_LOADING
        ]
        J[Aircraft.Wing.HIGH_LIFT_MASS, 'density'] = J['flap_mass', 'density']

        J[Aircraft.Wing.HIGH_LIFT_MASS, Mission.Landing.LIFT_COEFFICIENT_MAX] = J[
            'flap_mass', Mission.Landing.LIFT_COEFFICIENT_MAX
        ]

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.FLAP_CHORD_RATIO] = J[
            'flap_mass', Aircraft.Wing.FLAP_CHORD_RATIO
        ]

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.SLAT_CHORD_RATIO] = J[
            'slat_mass', Aircraft.Wing.SLAT_CHORD_RATIO
        ]

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.AREA] = (
            J['flap_mass', Aircraft.Wing.AREA] + J['slat_mass', Aircraft.Wing.AREA]
        )

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.FLAP_SPAN_RATIO] = J[
            'flap_mass', Aircraft.Wing.FLAP_SPAN_RATIO
        ]

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.SLAT_SPAN_RATIO] = J[
            'slat_mass', Aircraft.Wing.SLAT_SPAN_RATIO
        ]

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.TAPER_RATIO] = (
            J['flap_mass', Aircraft.Wing.TAPER_RATIO] + J['slat_mass', Aircraft.Wing.TAPER_RATIO]
        )

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.SPAN] = (
            J['flap_mass', Aircraft.Wing.SPAN] + J['slat_mass', Aircraft.Wing.SPAN]
        )

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = (
            J['flap_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
            + J['slat_mass', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT]
        )

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Wing.CENTER_CHORD] = (
            J['flap_mass', Aircraft.Wing.CENTER_CHORD] + J['slat_mass', Aircraft.Wing.CENTER_CHORD]
        )

        J[Aircraft.Wing.HIGH_LIFT_MASS, Aircraft.Fuselage.AVG_DIAMETER] = (
            J['flap_mass', Aircraft.Fuselage.AVG_DIAMETER]
            + J['slat_mass', Aircraft.Fuselage.AVG_DIAMETER]
        )


class ControlMass(om.ExplicitComponent):
    """
    Computation of total mass of cockpit controls, fixed wing controls, and SAS,
    and mass of surface controls.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')
        self.add_input('min_dive_vel', val=700, units='kn', desc='VDMIN: dive velocity')
        add_aviary_input(self, Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, units='unitless')
        add_aviary_input(
            self,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,
            units='unitless',
        )
        add_aviary_input(self, Aircraft.Controls.CONTROL_MASS_INCREMENT, units='lbm')

        add_aviary_output(self, Aircraft.Controls.TOTAL_MASS, units='lbm')
        add_aviary_output(self, Aircraft.Wing.SURFACE_CONTROL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Controls.TOTAL_MASS, '*')
        self.declare_partials(
            Aircraft.Wing.SURFACE_CONTROL_MASS,
            [
                Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT,
                Aircraft.Wing.AREA,
                Mission.Design.GROSS_MASS,
                Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                'min_dive_vel',
                Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,
            ],
        )

    def compute(self, inputs, outputs):
        c_mass_trend_wing_control = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT]

        wing_area = inputs[Aircraft.Wing.AREA]
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]

        c_mass_trend_cockpit_control = inputs[Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT]

        stab_aug_wt = (
            inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS] * GRAV_ENGLISH_LBM
        )
        CK15 = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER]
        CK18 = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER]
        CK19 = inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER]
        delta_control_wt = inputs[Aircraft.Controls.CONTROL_MASS_INCREMENT] * GRAV_ENGLISH_LBM
        min_dive_vel = inputs['min_dive_vel']

        dive_param = (1.15 * min_dive_vel) ** 2 / 391.0

        intermediate_control_wt = (
            c_mass_trend_wing_control
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param**0.345
        )
        cockpit_control_wt = c_mass_trend_cockpit_control * (gross_wt_initial / 1000.0) ** 0.41
        wing_control_wt = intermediate_control_wt - cockpit_control_wt
        outputs[Aircraft.Wing.SURFACE_CONTROL_MASS] = wing_control_wt / GRAV_ENGLISH_LBM
        stab_control_wt = stab_aug_wt

        outputs[Aircraft.Controls.TOTAL_MASS] = (
            CK15 * cockpit_control_wt
            + CK18 * wing_control_wt
            + CK19 * stab_control_wt
            + delta_control_wt
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        c_mass_trend_wing_control = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT]

        wing_area = inputs[Aircraft.Wing.AREA]
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]

        c_mass_trend_cockpit_control = inputs[Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT]

        stab_aug_wt = inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS]
        CK15 = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER]
        CK18 = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER]
        CK19 = inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER]
        min_dive_vel = inputs['min_dive_vel']

        dive_param = (1.15 * min_dive_vel) ** 2 / 391.0

        intermediate_control_wt = (
            c_mass_trend_wing_control
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param**0.345
        )
        cockpit_control_wt = c_mass_trend_cockpit_control * (gross_wt_initial / 1000.0) ** 0.41
        wing_control_wt = intermediate_control_wt - cockpit_control_wt
        dSCW_dSWCC = (
            wing_area**0.317 * (gross_wt_initial / 1000.0) ** 0.602 * ULF**0.525 * dive_param**0.345
        )
        dSCW_dWA = 0.317 * (
            c_mass_trend_wing_control
            * wing_area ** (0.317 - 1)
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param**0.345
        )
        dSCW_dWG = 0.602 * c_mass_trend_wing_control * wing_area**0.317 * (
            gross_wt_initial / 1000.0
        ) ** (0.602 - 1) * (
            1 / 1000
        ) * ULF**0.525 * dive_param**0.345 - 0.41 * c_mass_trend_cockpit_control * (
            gross_wt_initial / 1000.0
        ) ** (0.41 - 1) * (1 / 1000)
        dSCW_dULF = (
            0.525
            * c_mass_trend_wing_control
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF ** (0.525 - 1)
            * dive_param**0.345
        )
        dSCW_dMDV = (
            0.345
            * c_mass_trend_wing_control
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param ** (0.345 - 1)
            * 2
            * 1.15
            * min_dive_vel
            * 1.15
            / 391
        )
        dSCW_dCCWC = -((gross_wt_initial / 1000.0) ** 0.41)

        J[
            Aircraft.Wing.SURFACE_CONTROL_MASS,
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT,
        ] = dSCW_dSWCC / GRAV_ENGLISH_LBM

        J[Aircraft.Controls.TOTAL_MASS, Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT] = (
            CK18 * dSCW_dSWCC / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.AREA] = dSCW_dWA / GRAV_ENGLISH_LBM

        J[Aircraft.Controls.TOTAL_MASS, Aircraft.Wing.AREA] = CK18 * dSCW_dWA / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Mission.Design.GROSS_MASS] = dSCW_dWG

        J[Aircraft.Controls.TOTAL_MASS, Mission.Design.GROSS_MASS] = (
            0.41
            * CK15
            * (
                c_mass_trend_cockpit_control
                * (gross_wt_initial / 1000.0) ** (0.41 - 1)
                * (1 / 1000)
            )
            + CK18 * dSCW_dWG
        )
        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = (
            dSCW_dULF / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Controls.TOTAL_MASS, Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = (
            CK18 * dSCW_dULF / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Wing.SURFACE_CONTROL_MASS, 'min_dive_vel'] = dSCW_dMDV / GRAV_ENGLISH_LBM
        J[Aircraft.Controls.TOTAL_MASS, 'min_dive_vel'] = CK18 * dSCW_dMDV / GRAV_ENGLISH_LBM

        J[
            Aircraft.Wing.SURFACE_CONTROL_MASS,
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,
        ] = dSCW_dCCWC / GRAV_ENGLISH_LBM

        J[
            Aircraft.Controls.TOTAL_MASS,
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,
        ] = (CK15 * (gross_wt_initial / 1000.0) ** 0.41 + CK18 * dSCW_dCCWC) / GRAV_ENGLISH_LBM

        J[
            Aircraft.Controls.TOTAL_MASS,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS,
        ] = CK19

        J[Aircraft.Controls.TOTAL_MASS, Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER] = (
            cockpit_control_wt / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Controls.TOTAL_MASS, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER] = (
            wing_control_wt / GRAV_ENGLISH_LBM
        )

        J[
            Aircraft.Controls.TOTAL_MASS,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,
        ] = stab_aug_wt
        J[Aircraft.Controls.TOTAL_MASS, Aircraft.Controls.CONTROL_MASS_INCREMENT] = 1


class GearMass(om.ExplicitComponent):
    """Computation of total mass of landing gear and mass of main landing gear."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Wing.VERTICAL_MOUNT_LOCATION, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(
            self,
            Aircraft.Nacelle.CLEARANCE_RATIO,
            shape=num_engine_type,
            units='unitless',
        )
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')

        add_aviary_output(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')
        add_aviary_output(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.LandingGear.TOTAL_MASS,
            [
                Aircraft.LandingGear.MASS_COEFFICIENT,
                Mission.Design.GROSS_MASS,
                Aircraft.Nacelle.CLEARANCE_RATIO,
                Aircraft.Nacelle.AVG_DIAMETER,
            ],
        )
        self.declare_partials(Aircraft.LandingGear.MAIN_GEAR_MASS, '*')

    def compute(self, inputs, outputs):
        wing_loc = inputs[Aircraft.Wing.VERTICAL_MOUNT_LOCATION]
        c_gear_mass = inputs[Aircraft.LandingGear.MASS_COEFFICIENT]
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        c_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT]
        clearance_ratio = inputs[Aircraft.Nacelle.CLEARANCE_RATIO]
        nacelle_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]

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

        landing_gear_wt = c_gear_mass_modified * gross_wt_initial

        outputs[Aircraft.LandingGear.TOTAL_MASS] = landing_gear_wt / GRAV_ENGLISH_LBM
        outputs[Aircraft.LandingGear.MAIN_GEAR_MASS] = (
            c_main_gear * landing_gear_wt / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        c_gear_mass = inputs[Aircraft.LandingGear.MASS_COEFFICIENT]
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        c_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT]
        wing_loc = inputs[Aircraft.Wing.VERTICAL_MOUNT_LOCATION]
        clearance_ratio = inputs[Aircraft.Nacelle.CLEARANCE_RATIO]
        nacelle_diam = inputs[Aircraft.Nacelle.AVG_DIAMETER]

        val = (1.0 + clearance_ratio) * nacelle_diam
        gear_height_temp = KSfunction.compute(val, 50.0)
        dKS, _ = KSfunction.derivatives(val, 50.0)

        gear_height_temp = gear_height_temp[0]
        gear_height = gear_height_temp * sigmoidX(gear_height_temp, 6, 1 / 320.0)
        +6 * sigmoidX(gear_height_temp, 6, -1 / 320.0)

        dLGW_dCGW = (
            (0.85 * (1.0 + 0.1765 * gear_height / 6.0)) * sigmoidX(wing_loc, 0.005, -0.01 / 320.0)
            + sigmoidX(wing_loc, 0.005, 0.01 / 320.0)
        ) * gross_wt_initial

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
            (c_gear_mass * 0.85 * 0.1765 / 6 * dGH_dCR * gross_wt_initial)
            * sigmoidX(wing_loc, 0.005, -0.01 / 320.0)
        )

        dLGW_dND = max(
            (c_gear_mass * 0.85 * 0.1765 / 6 * dGH_dND * gross_wt_initial)
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

        J[Aircraft.LandingGear.TOTAL_MASS, Mission.Design.GROSS_MASS] = c_gear_mass_modified

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.Nacelle.CLEARANCE_RATIO] = (
            c_main_gear * dLGW_dCR / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.Nacelle.AVG_DIAMETER] = (
            c_main_gear * dLGW_dND / GRAV_ENGLISH_LBM
        )

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.MASS_COEFFICIENT] = (
            c_main_gear * dLGW_dCGW * sigmoidX(wing_loc, 0.005, -0.01 / 320.0)
            + c_main_gear * gross_wt_initial * sigmoidX(wing_loc, 0.005, 0.01 / 320.0)
        ) / GRAV_ENGLISH_LBM

        J[
            Aircraft.LandingGear.MAIN_GEAR_MASS,
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT,
        ] = c_gear_mass_modified * gross_wt_initial / GRAV_ENGLISH_LBM

        J[Aircraft.LandingGear.MAIN_GEAR_MASS, Mission.Design.GROSS_MASS] = (
            c_gear_mass_modified * c_main_gear
        )


class FixedMassGroup(om.Group):
    """Group of all fixed mass components for GASP-based mass."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Electrical.HAS_HYBRID_SYSTEM)

    def setup(self):
        self.add_subsystem(
            'params',
            MassParameters(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'payload',
            PayloadGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'tail',
            TailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'HL',
            HighLiftMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'controls',
            ControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'gear',
            GearMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        has_hybrid_system = self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]

        if has_hybrid_system:
            self.add_subsystem(
                'augmentation',
                ElectricAugmentationMass(),
                promotes_inputs=['*'],
                promotes_outputs=['aug_mass'],
            )

        self.add_subsystem(
            'engine',
            EngineMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        if has_hybrid_system:
            self.promotes(
                'engine',
                inputs=['aug_mass'],
            )

        self.set_input_defaults('min_dive_vel', val=420, units='kn')
