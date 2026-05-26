import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class MassSummation(om.Group):
    """
    Group to compute top-level mass groups for GASP mass estimation. No new masses are computed
    here, just grouping and summing already calculated masses.
    """

    def setup(self):
        self.add_subsystem(
            'empennage_mass', EmpennageMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'structure_mass', StructureMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'propulsion_mass', PropulsionMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'systems_and_equipment_mass',
            SystemsEquipmentMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'empty_mass_group', EmptyMass(), promotes_inputs=['*'], promotes_outputs=['*']
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


class EmpennageMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.HorizontalTail.MASS, units='lbm')
        add_aviary_input(self, Aircraft.VerticalTail.MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.EMPENNAGE_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.EMPENNAGE_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        htail_mass = inputs[Aircraft.HorizontalTail.MASS]
        vtail_mass = inputs[Aircraft.VerticalTail.MASS]

        outputs[Aircraft.Design.EMPENNAGE_MASS] = htail_mass + vtail_mass


class StructureMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Design.EMPENNAGE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.MASS, units='lbm')
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.STRUCTURAL_MASS_INCREMENT, units='lbm')

        add_aviary_output(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.STRUCTURE_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        empennage_mass = inputs[Aircraft.Design.EMPENNAGE_MASS]
        fuselage_mass = inputs[Aircraft.Fuselage.MASS]
        landing_gear_mass = inputs[Aircraft.LandingGear.TOTAL_MASS]
        pod_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS]
        wing_mass = inputs[Aircraft.Wing.MASS]
        delta_struct_wt = inputs[Aircraft.Design.STRUCTURAL_MASS_INCREMENT]

        outputs[Aircraft.Design.STRUCTURE_MASS] = (
            wing_mass
            + empennage_mass
            + fuselage_mass
            + landing_gear_mass
            + pod_mass
            + delta_struct_wt
        )


class PropulsionMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm')
        # TODO These variables need cleanup, proper names, etc.
        #      They should probably be removed and we pass the couple individual masses that make it
        #      up here instead
        self.add_input('eng_comb_mass', units='lbm')
        self.add_input('prop_mass_all', units='lbm')
        add_aviary_input(self, Aircraft.Battery.MASS, units='lbm')

        add_aviary_output(self, Aircraft.Propulsion.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Propulsion.MASS, ['*'], val=1)

    def compute(self, inputs, outputs):
        fuel_sys_mass = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS]
        eng_comb_mass = inputs['eng_comb_mass']
        prop_mass_all = inputs['prop_mass_all']
        battery_mass = inputs[Aircraft.Battery.MASS]

        outputs[Aircraft.Propulsion.MASS] = (
            fuel_sys_mass + eng_comb_mass + prop_mass_all + battery_mass
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
        add_aviary_input(self, Aircraft.Controls.MASS, units='lbm')
        add_aviary_input(self, Aircraft.OxygenSystem.MASS, units='lbm')

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
        controls_mass = inputs[Aircraft.Controls.MASS]
        oxygen_mass = inputs[Aircraft.OxygenSystem.MASS]

        outputs[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS] = (
            APU_mass
            + instrument_mass
            + hydraulics_mass
            + elec_mass
            + avionics_mass
            + furnish_mass
            + AC_mass
            + anti_icing_mass
            + controls_mass
            + oxygen_mass
        )


class EmptyMass(om.ExplicitComponent):
    def setup(self):
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
        subsystems_mass = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]

        outputs[Aircraft.Design.EMPTY_MASS] = (
            structure_mass + prop_mass + sys_equip_mass + subsystems_mass
        )


class UsefulLoadMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.CABIN_CREW_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.FLIGHT_CREW_MASS, units='lbm')
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.EMERGENCY_EQUIPMENT_MASS)

        add_aviary_output(self, Mission.USEFUL_LOAD, units='lbm')

    def setup_partials(self):
        self.declare_partials(Mission.USEFUL_LOAD, '*', val=1)

    def compute(self, inputs, outputs):
        cargo_container_mass = inputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS]
        cabin_crew_mass = inputs[Aircraft.CrewPayload.CABIN_CREW_MASS]
        flight_crew_mass = inputs[Aircraft.CrewPayload.FLIGHT_CREW_MASS]
        oil_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS]
        pass_service_mass = inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS]
        unusable_fuel_mass = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS]
        emergency_equip_mass = inputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS]

        outputs[Mission.USEFUL_LOAD] = (
            cabin_crew_mass
            + flight_crew_mass
            + unusable_fuel_mass
            + oil_mass
            + pass_service_mass
            + cargo_container_mass
            + emergency_equip_mass
        )


class OperatingMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Design.EMPTY_MASS, units='lbm')
        add_aviary_input(self, Mission.USEFUL_LOAD, units='lbm')

        add_aviary_output(self, Mission.OPERATING_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Mission.OPERATING_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        useful_load = inputs[Mission.USEFUL_LOAD]
        empty_mass = inputs[Aircraft.Design.EMPTY_MASS]

        outputs[Mission.OPERATING_MASS] = empty_mass + useful_load


class ZeroFuelMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, units='lbm')
        add_aviary_input(self, Mission.OPERATING_MASS, units='lbm')

        add_aviary_output(self, Mission.ZERO_FUEL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Mission.ZERO_FUEL_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        payload_mass = inputs[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS]
        operating_mass = inputs[Mission.OPERATING_MASS]

        outputs[Mission.ZERO_FUEL_MASS] = operating_mass + payload_mass
