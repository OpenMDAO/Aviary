import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.enums import AircraftTypes
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission
from aviary.subsystems.mass.gasp_based.air_conditioning import ACMass, BWBACMass
from aviary.subsystems.mass.gasp_based.anti_icing import AntiIcingMass
from aviary.subsystems.mass.gasp_based.apu import APUMass
from aviary.subsystems.mass.gasp_based.avionics import AvionicsMass
from aviary.subsystems.mass.gasp_based.cargo import CargoMass
from aviary.subsystems.mass.gasp_based.crew import FlightCrewMass
from aviary.subsystems.mass.gasp_based.crew import NonFlightCrewMass
from aviary.subsystems.mass.gasp_based.electrical import ElectricalMass
from aviary.subsystems.mass.gasp_based.emergency_equipment import EmergencyEquipment
from aviary.subsystems.mass.gasp_based.engine_oil import EngineOilMass
from aviary.subsystems.mass.gasp_based.fuel_capacity import TrappedFuelCapacity
from aviary.subsystems.mass.gasp_based.furnishings import FurnishingMass, BWBFurnishingMass
from aviary.subsystems.mass.gasp_based.hydraulics import HydraulicsMass
from aviary.subsystems.mass.gasp_based.instruments import InstrumentMass
from aviary.subsystems.mass.gasp_based.oxygen_system import OxygenSystemMass
from aviary.subsystems.mass.gasp_based.passenger_service import PassengerServiceMass


class EquipMassSum(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS, units='lbm')
        add_aviary_input(self, Aircraft.APU.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Furnishings.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Instruments.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Hydraulics.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Electrical.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Avionics.MASS, units='lbm')
        add_aviary_input(self, Aircraft.AntiIcing.MASS, units='lbm')
        add_aviary_input(self, Aircraft.OxygenSystem.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, units='lbm')

        self.declare_partials(Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, '*')

    def compute(self, inputs, outputs):
        air_conditioning_mass = inputs[Aircraft.AirConditioning.MASS]
        furnishing_mass = inputs[Aircraft.Furnishings.MASS]
        APU_wt = inputs[Aircraft.APU.MASS]
        instrument_wt = inputs[Aircraft.Instruments.MASS]
        hydraulic_wt = inputs[Aircraft.Hydraulics.MASS]
        electrical_wt = inputs[Aircraft.Electrical.MASS]
        avionics_wt = inputs[Aircraft.Avionics.MASS]
        icing_wt = inputs[Aircraft.AntiIcing.MASS]
        oxygen_system_wt = inputs[Aircraft.OxygenSystem.MASS]
        subsystems_wt = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]

        equip_mass_sum = (
            air_conditioning_mass
            + furnishing_mass
            + APU_wt
            + instrument_wt
            + hydraulic_wt
            + electrical_wt
            + avionics_wt
            + icing_wt
            + oxygen_system_wt
            + subsystems_wt
        )

        outputs[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS] = equip_mass_sum

    def compute_partials(self, inputs, J):
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.AirConditioning.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.Furnishings.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.APU.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.Instruments.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.Hydraulics.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.Electrical.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.Avionics.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.AntiIcing.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.OxygenSystem.MASS] = 1
        J[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS] = 1


class EquipMassGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'ac',
            ACMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'apu',
            APUMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'avionics',
            AvionicsMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'electrical',
            ElectricalMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'hydraulic',
            HydraulicsMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'instrument',
            InstrumentMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'oxygen_system',
            OxygenSystemMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )


class UsefulLoadMass(om.ExplicitComponent):
    """
    Computation of fixed equipment mass and useful load for GASP-based mass.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.FLIGHT_CREW_MASS)
        add_aviary_input(self, Aircraft.CrewPayload.CABIN_CREW_MASS)
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS)
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS)
        add_aviary_input(self, Aircraft.Design.EMERGENCY_EQUIPMENT_MASS)
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS)
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS)

        add_aviary_output(self, Mission.Summary.USEFUL_LOAD, units='lbm')

        self.declare_partials(Mission.Summary.USEFUL_LOAD, '*')

    def compute(self, inputs, outputs):
        pilot_wt = inputs[Aircraft.CrewPayload.FLIGHT_CREW_MASS]
        flight_attendant_wt = inputs[Aircraft.CrewPayload.CABIN_CREW_MASS]
        oil_wt = inputs[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS]
        service_wt = inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS]
        emergency_wt = inputs[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS]
        trapped_fuel_wt = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS]
        cargo_handling_wt = inputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS]

        useful_wt = (
            pilot_wt
            + flight_attendant_wt
            + oil_wt
            + service_wt
            + emergency_wt
            + trapped_fuel_wt
            + cargo_handling_wt
        )

        outputs[Mission.Summary.USEFUL_LOAD] = useful_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        J[Mission.Summary.USEFUL_LOAD, Aircraft.CrewPayload.FLIGHT_CREW_MASS] = 1
        J[Mission.Summary.USEFUL_LOAD, Aircraft.CrewPayload.CABIN_CREW_MASS] = 1
        J[Mission.Summary.USEFUL_LOAD, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS] = 1
        J[Mission.Summary.USEFUL_LOAD, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS] = 1
        J[Mission.Summary.USEFUL_LOAD, Aircraft.Design.EMERGENCY_EQUIPMENT_MASS] = 1
        J[Mission.Summary.USEFUL_LOAD, Aircraft.Fuel.UNUSABLE_FUEL_MASS] = 1
        J[Mission.Summary.USEFUL_LOAD, Aircraft.CrewPayload.CARGO_CONTAINER_MASS] = 1


class UsefulLoadMassGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'cargo',
            CargoMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'flight_crew',
            FlightCrewMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'non_flight_crew',
            NonFlightCrewMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'engine_oil',
            EngineOilMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'emergency_equipment',
            EmergencyEquipment(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'passenger_service',
            PassengerServiceMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'trapped_fuel',
            TrappedFuelCapacity(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )


class BWBEquipMassGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'ac',
            BWBACMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'furnishing',
            BWBFurnishingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'apu',
            APUMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'avionics',
            AvionicsMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'electrical',
            ElectricalMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'hydraulic',
            HydraulicsMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'instrument',
            InstrumentMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'oxygen_system',
            OxygenSystemMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
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
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'equip',
                EquipMassGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        self.add_subsystem(
            'useful',
            UsefulLoadMassGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
