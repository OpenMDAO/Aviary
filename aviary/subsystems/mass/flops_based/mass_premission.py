import openmdao.api as om

from aviary.subsystems.mass.flops_based.air_conditioning import (
    AltAirCondMass, TransportAirCondMass)
from aviary.subsystems.mass.flops_based.anti_icing import AntiIcingMass
from aviary.subsystems.mass.flops_based.apu import TransportAPUMass
from aviary.subsystems.mass.flops_based.avionics import TransportAvionicsMass
from aviary.subsystems.mass.flops_based.canard import CanardMass
from aviary.subsystems.mass.flops_based.cargo import CargoMass
from aviary.subsystems.mass.flops_based.cargo_containers import TransportCargoContainersMass
from aviary.subsystems.mass.flops_based.crew import NonFlightCrewMass, FlightCrewMass
from aviary.subsystems.mass.flops_based.electrical import (
    AltElectricalMass, ElectricalMass)
from aviary.subsystems.mass.flops_based.engine import EngineMass
from aviary.subsystems.mass.flops_based.engine_controls import TransportEngineCtrlsMass
from aviary.subsystems.mass.flops_based.engine_oil import (
    TransportEngineOilMass, AltEngineOilMass)
from aviary.subsystems.mass.flops_based.fin import FinMass
from aviary.subsystems.mass.flops_based.fuel_capacity import FuelCapacityGroup
from aviary.subsystems.mass.flops_based.fuel_system import (
    AltFuelSystemMass, TransportFuelSystemMass)
from aviary.subsystems.mass.flops_based.furnishings import (
    AltFurnishingsGroupMass, AltFurnishingsGroupMassBase,
    TransportFurnishingsGroupMass)
from aviary.subsystems.mass.flops_based.fuselage import (
    AltFuselageMass, TransportFuselageMass)
from aviary.subsystems.mass.flops_based.horizontal_tail import (
    AltHorizontalTailMass, HorizontalTailMass)
from aviary.subsystems.mass.flops_based.hydraulics import (
    AltHydraulicsGroupMass, TransportHydraulicsGroupMass)
from aviary.subsystems.mass.flops_based.instruments import TransportInstrumentMass
from aviary.subsystems.mass.flops_based.landing_group import LandingMassGroup
from aviary.subsystems.mass.flops_based.mass_summation import MassSummation
from aviary.subsystems.mass.flops_based.misc_engine import EngineMiscMass
from aviary.subsystems.mass.flops_based.nacelle import NacelleMass
from aviary.subsystems.mass.flops_based.paint import PaintMass
from aviary.subsystems.mass.flops_based.passenger_service import (
    AltPassengerServiceMass, PassengerServiceMass)
from aviary.subsystems.mass.flops_based.starter import TransportStarterMass
from aviary.subsystems.mass.flops_based.surface_controls import (
    AltSurfaceControlMass, SurfaceControlMass)
from aviary.subsystems.mass.flops_based.thrust_reverser import ThrustReverserMass
from aviary.subsystems.mass.flops_based.unusable_fuel import (
    AltUnusableFuelMass, TransportUnusableFuelMass)
from aviary.subsystems.mass.flops_based.vertical_tail import (
    AltVerticalTailMass, VerticalTailMass)
from aviary.subsystems.mass.flops_based.wing_group import WingMassGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Mission


class MassPremission(om.Group):
    """
    Pre-mission group of top-level mass estimation groups and components for FLOPS-based analysis:
    CargoMass, TransportCargoContainersMass, TransportEngineCtrlsMass, TransportAvionicsMass,
    FuelCapacityGroup, EngineMass, etc.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        aviary_options: AviaryValues = self.options['aviary_options']
        alt_mass = aviary_options.get_val(Aircraft.Design.USE_ALT_MASS)

        self.add_subsystem(
            'cargo',
            CargoMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'cargo_containers',
            TransportCargoContainersMass(
                aviary_options=aviary_options),
            promotes_inputs=['*', ], promotes_outputs=['*', ])

        self.add_subsystem(
            'engine_controls',
            TransportEngineCtrlsMass(aviary_options=aviary_options),
            promotes_inputs=['*', ], promotes_outputs=['*', ])

        self.add_subsystem(
            'avionics',
            TransportAvionicsMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'fuel_capacity_group',
            FuelCapacityGroup(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'engine_mass',
            EngineMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        if alt_mass:
            self.add_subsystem(
                'fuel_system',
                AltFuelSystemMass(aviary_options=aviary_options),
                promotes_inputs=['*', ], promotes_outputs=['*', ])

            self.add_subsystem(
                'AC',
                AltAirCondMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'engine_oil',
                AltEngineOilMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'furnishing_base',
                AltFurnishingsGroupMassBase(
                    aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'furnishings',
                AltFurnishingsGroupMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'hydraulics',
                AltHydraulicsGroupMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'pass_service',
                AltPassengerServiceMass(
                    aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'unusable_fuel',
                AltUnusableFuelMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'electrical',
                AltElectricalMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

        else:
            self.add_subsystem(
                'fuel_system',
                TransportFuelSystemMass(aviary_options=aviary_options),
                promotes_inputs=['*', ], promotes_outputs=['*', ])

            self.add_subsystem(
                'AC',
                TransportAirCondMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'engine_oil',
                TransportEngineOilMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'furnishings',
                TransportFurnishingsGroupMass(
                    aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'hydraulics',
                TransportHydraulicsGroupMass(
                    aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'pass_service',
                PassengerServiceMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'unusable_fuel',
                TransportUnusableFuelMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'electrical',
                ElectricalMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'starter',
            TransportStarterMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'anti_icing',
            AntiIcingMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'apu',
            TransportAPUMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'nonflight_crew',
            NonFlightCrewMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'flight_crew',
            FlightCrewMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'instruments',
            TransportInstrumentMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'misc_engine',
            EngineMiscMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'nacelle',
            NacelleMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'paint',
            PaintMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'thrust_rev',
            ThrustReverserMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'landing_group',
            LandingMassGroup(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        if alt_mass:
            self.add_subsystem(
                'surf_ctrl',
                AltSurfaceControlMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'fuselage',
                AltFuselageMass(aviary_options=aviary_options),
                promotes_inputs=['*', ], promotes_outputs=['*', ])

            self.add_subsystem(
                'htail',
                AltHorizontalTailMass(aviary_options=aviary_options),
                promotes_inputs=['*', ], promotes_outputs=['*', ])

            self.add_subsystem(
                'vert_tail',
                AltVerticalTailMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

        else:
            self.add_subsystem(
                'surf_ctrl',
                SurfaceControlMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'fuselage',
                TransportFuselageMass(aviary_options=aviary_options),
                promotes_inputs=['*', ], promotes_outputs=['*', ])

            self.add_subsystem(
                'htail',
                HorizontalTailMass(aviary_options=aviary_options),
                promotes_inputs=['*', ], promotes_outputs=['*', ])

            self.add_subsystem(
                'vert_tail',
                VerticalTailMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'canard',
            CanardMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'fin',
            FinMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'wing_group',
            WingMassGroup(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'total_mass',
            MassSummation(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])
