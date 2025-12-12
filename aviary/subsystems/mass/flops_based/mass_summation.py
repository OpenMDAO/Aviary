import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class MassSummation(om.Group):
    """Group to compute top-level mass groups for FLOPS mass estimation."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.USE_ALT_MASS)

    def setup(self):
        alt_mass = self.options[Aircraft.Design.USE_ALT_MASS]

        self.add_subsystem(
            'empennage_mass', EmpennageGroupMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'structure_mass', StructureMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'propulsion_mass', PropulsionMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        # Systems and equipment mass calculations are not combined into a single group to avoid an
        # additional layer of groups (standard is one component, alternate would be a group)
        if alt_mass:
            self.add_subsystem(
                'systems_and_equipment_mass_base',
                AltSystemsEquipmentMassBase(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'systems_and_equipment_mass',
                AltSystemsEquipmentMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        else:
            self.add_subsystem(
                'systems_and_equipment_mass',
                SystemsEquipmentMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        # TODO should these be combined into a single group with alt equation option?
        if alt_mass:
            self.add_subsystem(
                'empty_mass', AltEmptyMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        else:
            self.add_subsystem(
                'empty_mass', EmptyMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        self.add_subsystem(
            'useful_load_mass', UsefulLoadMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'operating_mass', OperatingMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'zero_fuel_mass', ZeroFuelMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem('fuel_mass', FuelMass(), promotes_inputs=['*'], promotes_outputs=['*'])


class EmpennageGroupMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Canard.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fins.MASS, units='lbm')
        add_aviary_input(self, Aircraft.HorizontalTail.MASS, units='lbm')
        add_aviary_input(self, Aircraft.VerticalTail.MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.EMPENNAGE_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.EMPENNAGE_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        canard_mass = inputs[Aircraft.Canard.MASS]
        fin_mass = inputs[Aircraft.Fins.MASS]
        htail_mass = inputs[Aircraft.HorizontalTail.MASS]
        vtail_mass = inputs[Aircraft.VerticalTail.MASS]

        outputs[Aircraft.Design.EMPENNAGE_MASS] = canard_mass + fin_mass + htail_mass + vtail_mass


class StructureMass(om.ExplicitComponent):
    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Design.EMPENNAGE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.MASS, units='lbm')
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Nacelle.MASS, shape=num_engine_type, units='lbm')
        add_aviary_input(self, Aircraft.Paint.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')

    def setup_partials(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        self.declare_partials(Aircraft.Design.STRUCTURE_MASS, '*', val=1)
        self.declare_partials(
            Aircraft.Design.STRUCTURE_MASS, Aircraft.Nacelle.MASS, val=np.ones(num_engine_type)
        )

    def compute(self, inputs, outputs):
        empennage_mass = inputs[Aircraft.Design.EMPENNAGE_MASS]
        fuselage_mass = inputs[Aircraft.Fuselage.MASS]
        landing_gear_mass = inputs[Aircraft.LandingGear.TOTAL_MASS]
        nacelle_mass = inputs[Aircraft.Nacelle.MASS]
        paint_mass = inputs[Aircraft.Paint.MASS]
        wing_mass = inputs[Aircraft.Wing.MASS]

        outputs[Aircraft.Design.STRUCTURE_MASS] = (
            wing_mass
            + empennage_mass
            + fuselage_mass
            + landing_gear_mass
            + np.sum(nacelle_mass)
            + paint_mass
        )


class PropulsionMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_MISC_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Propulsion.MASS, units='lbm')

    def setup_partials(self):
        prop_wrt = [
            Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
            Aircraft.Propulsion.TOTAL_MISC_MASS,
            Aircraft.Fuel.FUEL_SYSTEM_MASS,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS,
        ]

        self.declare_partials(Aircraft.Propulsion.MASS, prop_wrt, val=1)

    def compute(self, inputs, outputs):
        fuel_sys_mass = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS]
        misc_prop_mass = inputs[Aircraft.Propulsion.TOTAL_MISC_MASS]
        thrust_rev_mass = inputs[Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS]
        total_eng_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_MASS]

        outputs[Aircraft.Propulsion.MASS] = (
            thrust_rev_mass + misc_prop_mass + fuel_sys_mass + total_eng_mass
        )


class SystemsEquipmentMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS, units='lbm')
        add_aviary_input(self, Aircraft.AntiIcing.MASS, units='lbm')
        add_aviary_input(self, Aircraft.APU.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Avionics.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Electrical.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Furnishings.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Hydraulics.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Instruments.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        AC_mass = inputs[Aircraft.AirConditioning.MASS]
        anti_icing_mass = inputs[Aircraft.AntiIcing.MASS]
        APU_mass = inputs[Aircraft.APU.MASS]
        avionics_mass = inputs[Aircraft.Avionics.MASS]
        elec_mass = inputs[Aircraft.Electrical.MASS]
        furnish_mass = inputs[Aircraft.Furnishings.MASS]
        hydraulics_mass = inputs[Aircraft.Hydraulics.MASS]
        instrument_mass = inputs[Aircraft.Instruments.MASS]
        surf_control_mass = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS]
        subsystems_mass = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]

        outputs[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS] = (
            surf_control_mass
            + APU_mass
            + instrument_mass
            + hydraulics_mass
            + elec_mass
            + avionics_mass
            + furnish_mass
            + AC_mass
            + anti_icing_mass
            + subsystems_mass
        )


class AltSystemsEquipmentMassBase(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS, units='lbm')
        add_aviary_input(self, Aircraft.AntiIcing.MASS, units='lbm')
        add_aviary_input(self, Aircraft.APU.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Avionics.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Electrical.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Furnishings.MASS_BASE, units='lbm')
        add_aviary_input(self, Aircraft.Hydraulics.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Instruments.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE, '*', val=1)

    def compute(self, inputs, outputs):
        AC_mass = inputs[Aircraft.AirConditioning.MASS]
        anti_icing_mass = inputs[Aircraft.AntiIcing.MASS]
        APU_mass = inputs[Aircraft.APU.MASS]
        avionics_mass = inputs[Aircraft.Avionics.MASS]
        elec_mass = inputs[Aircraft.Electrical.MASS]
        furnish_mass_base = inputs[Aircraft.Furnishings.MASS_BASE]
        hydraulics_mass = inputs[Aircraft.Hydraulics.MASS]
        instrument_mass = inputs[Aircraft.Instruments.MASS]
        surf_control_mass = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS]

        outputs[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE] = (
            surf_control_mass
            + APU_mass
            + instrument_mass
            + hydraulics_mass
            + elec_mass
            + avionics_mass
            + furnish_mass_base
            + AC_mass
            + anti_icing_mass
        )


class AltSystemsEquipmentMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE, units='lbm')
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS,
            [Aircraft.Design.STRUCTURE_MASS, Aircraft.Propulsion.MASS],
            val=0.01,
        )

        self.declare_partials(
            Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS,
            Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE,
            val=1.01,
        )

    def compute(self, inputs, outputs):
        sys_equip_mass_base = inputs[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE]
        structure_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        prop_mass = inputs[Aircraft.Propulsion.MASS]

        outputs[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS] = sys_equip_mass_base + 0.01 * (
            structure_mass + prop_mass + sys_equip_mass_base
        )


class EmptyMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Design.EMPTY_MASS_MARGIN, units='lbm')
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.EMPTY_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.EMPTY_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        structure_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        prop_mass = inputs[Aircraft.Propulsion.MASS]
        sys_equip_mass = inputs[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS]
        empty_mass_margin = inputs[Aircraft.Design.EMPTY_MASS_MARGIN]
        subsystems_mass = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]

        outputs[Aircraft.Design.EMPTY_MASS] = (
            structure_mass + prop_mass + sys_equip_mass + empty_mass_margin + subsystems_mass
        )


class AltEmptyMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Design.EMPTY_MASS_MARGIN, units='lbm')
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE, units='lbm')
        add_aviary_input(self, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.EMPTY_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Design.EMPTY_MASS,
            [
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE,
            ],
            val=1.01,
        )
        self.declare_partials(
            Aircraft.Design.EMPTY_MASS,
            [Aircraft.Design.EMPTY_MASS_MARGIN, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS],
            val=1.0,
        )

    def compute(self, inputs, outputs):
        structure_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        prop_mass = inputs[Aircraft.Propulsion.MASS]
        sys_equip_mass_base = inputs[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE]
        empty_mass_margin = inputs[Aircraft.Design.EMPTY_MASS_MARGIN]
        subsystems_mass = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]

        outputs[Aircraft.Design.EMPTY_MASS] = (
            1.01 * (structure_mass + prop_mass + sys_equip_mass_base)
            + empty_mass_margin
            + subsystems_mass
        )


class UsefulLoadMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.CABIN_CREW_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.FLIGHT_CREW_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, units='lbm')

        add_aviary_output(self, Mission.Summary.USEFUL_LOAD, units='lbm')

    def setup_partials(self):
        self.declare_partials(Mission.Summary.USEFUL_LOAD, '*', val=1)

    def compute(self, inputs, outputs):
        cargo_container_mass = inputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS]
        cabin_crew_mass = inputs[Aircraft.CrewPayload.CABIN_CREW_MASS]
        flight_crew_mass = inputs[Aircraft.CrewPayload.FLIGHT_CREW_MASS]
        oil_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS]
        pass_service_mass = inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS]
        unusable_fuel_mass = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS]

        outputs[Mission.Summary.USEFUL_LOAD] = (
            cabin_crew_mass
            + flight_crew_mass
            + unusable_fuel_mass
            + oil_mass
            + pass_service_mass
            + cargo_container_mass
        )


class OperatingMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Design.EMPTY_MASS, units='lbm')
        add_aviary_input(self, Mission.Summary.USEFUL_LOAD, units='lbm')

        add_aviary_output(self, Mission.Summary.OPERATING_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Mission.Summary.OPERATING_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        useful_load = inputs[Mission.Summary.USEFUL_LOAD]
        empty_mass = inputs[Aircraft.Design.EMPTY_MASS]

        outputs[Mission.Summary.OPERATING_MASS] = empty_mass + useful_load


class ZeroFuelMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_MASS_TOTAL, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.BAGGAGE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_MASS, units='lbm')
        add_aviary_input(self, Mission.Summary.OPERATING_MASS, units='lbm')

        add_aviary_output(self, Mission.Summary.ZERO_FUEL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Mission.Summary.ZERO_FUEL_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        pass_mass = inputs[Aircraft.CrewPayload.PASSENGER_MASS_TOTAL]
        bag_mass = inputs[Aircraft.CrewPayload.BAGGAGE_MASS]
        cargo_mass = inputs[Aircraft.CrewPayload.CARGO_MASS]
        operating_mass = inputs[Mission.Summary.OPERATING_MASS]

        outputs[Mission.Summary.ZERO_FUEL_MASS] = operating_mass + pass_mass + bag_mass + cargo_mass


# TODO DO NOT CALCULATE MISSION FUEL THIS WAY!!! ONLY COMPUTE DIRECTLY FROM INTEGRATING FUEL BURN
class FuelMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Mission.Summary.ZERO_FUEL_MASS, units='lbm')

        add_aviary_output(self, Mission.Summary.FUEL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Mission.Summary.FUEL_MASS, Mission.Design.GROSS_MASS, val=1)
        self.declare_partials(Mission.Summary.FUEL_MASS, Mission.Summary.ZERO_FUEL_MASS, val=-1)

    def compute(self, inputs, outputs):
        zero_fuel_mass = inputs[Mission.Summary.ZERO_FUEL_MASS]
        gross_mass = inputs[Mission.Design.GROSS_MASS]

        outputs[Mission.Summary.FUEL_MASS] = gross_mass - zero_fuel_mass
