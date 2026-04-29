import openmdao.api as om

from aviary.subsystems.mass.gasp_based.air_conditioning import ACMass, BWBACMass
from aviary.subsystems.mass.gasp_based.anti_icing import AntiIcingMass
from aviary.subsystems.mass.gasp_based.apu import APUMass
from aviary.subsystems.mass.gasp_based.avionics import AvionicsMass
from aviary.subsystems.mass.gasp_based.cargo_containers import CargoContainerMass
from aviary.subsystems.mass.gasp_based.crew import CabinCrewMass, FlightCrewMass
from aviary.subsystems.mass.gasp_based.electrical import ElectricalMass
from aviary.subsystems.mass.gasp_based.emergency_equipment import EmergencyEquipment
from aviary.subsystems.mass.gasp_based.engine_oil import EngineOilMass
from aviary.subsystems.mass.gasp_based.fuel_capacity import TrappedFuelCapacity
from aviary.subsystems.mass.gasp_based.furnishings import BWBFurnishingMass, FurnishingMass
from aviary.subsystems.mass.gasp_based.hydraulics import HydraulicsMass
from aviary.subsystems.mass.gasp_based.instruments import InstrumentMass
from aviary.subsystems.mass.gasp_based.oxygen_system import OxygenSystemMass
from aviary.subsystems.mass.gasp_based.passenger_service import PassengerServiceMass
from aviary.variable_info.enums import AircraftTypes
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class EquipMassGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'aircon',
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
            'oxygen',
            OxygenSystemMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )


class UsefulLoadMassGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'cargo_containers',
            CargoContainerMass(),
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
            'cabin_crew',
            CabinCrewMass(),
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
            'aircon',
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
            'oxygen',
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
