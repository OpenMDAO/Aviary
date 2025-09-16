import openmdao.api as om

from aviary.subsystems.mass.flops_based.air_conditioning import AltAirCondMass, TransportAirCondMass
from aviary.subsystems.mass.flops_based.anti_icing import AntiIcingMass
from aviary.subsystems.mass.flops_based.apu import TransportAPUMass
from aviary.subsystems.mass.flops_based.avionics import TransportAvionicsMass
from aviary.subsystems.mass.flops_based.canard import CanardMass
from aviary.subsystems.mass.flops_based.cargo import PayloadGroup
from aviary.subsystems.mass.flops_based.cargo_containers import TransportCargoContainersMass
from aviary.subsystems.mass.flops_based.crew import FlightCrewMass, NonFlightCrewMass
from aviary.subsystems.mass.flops_based.electrical import AltElectricalMass, ElectricalMass
from aviary.subsystems.mass.flops_based.engine import EngineMass
from aviary.subsystems.mass.flops_based.engine_controls import TransportEngineCtrlsMass
from aviary.subsystems.mass.flops_based.engine_oil import AltEngineOilMass, TransportEngineOilMass
from aviary.subsystems.mass.flops_based.fin import FinMass
from aviary.subsystems.mass.flops_based.fuel_capacity import FuelCapacityGroup
from aviary.subsystems.mass.flops_based.fuel_system import (
    AltFuelSystemMass,
    TransportFuelSystemMass,
)
from aviary.subsystems.mass.flops_based.furnishings import (
    AltFurnishingsGroupMass,
    AltFurnishingsGroupMassBase,
    TransportFurnishingsGroupMass,
)
from aviary.subsystems.mass.flops_based.fuselage import AltFuselageMass, TransportFuselageMass
from aviary.subsystems.mass.flops_based.horizontal_tail import (
    AltHorizontalTailMass,
    HorizontalTailMass,
)
from aviary.subsystems.mass.flops_based.hydraulics import (
    AltHydraulicsGroupMass,
    TransportHydraulicsGroupMass,
)
from aviary.subsystems.mass.flops_based.instruments import TransportInstrumentMass
from aviary.subsystems.mass.flops_based.landing_group import LandingMassGroup
from aviary.subsystems.mass.flops_based.mass_summation import MassSummation
from aviary.subsystems.mass.flops_based.misc_engine import EngineMiscMass
from aviary.subsystems.mass.flops_based.nacelle import NacelleMass
from aviary.subsystems.mass.flops_based.paint import PaintMass
from aviary.subsystems.mass.flops_based.passenger_service import (
    AltPassengerServiceMass,
    PassengerServiceMass,
)
from aviary.subsystems.mass.flops_based.starter import TransportStarterMass
from aviary.subsystems.mass.flops_based.surface_controls import (
    AltSurfaceControlMass,
    SurfaceControlMass,
)
from aviary.subsystems.mass.flops_based.thrust_reverser import ThrustReverserMass
from aviary.subsystems.mass.flops_based.unusable_fuel import (
    AltUnusableFuelMass,
    TransportUnusableFuelMass,
)
from aviary.subsystems.mass.flops_based.vertical_tail import AltVerticalTailMass, VerticalTailMass
from aviary.subsystems.mass.flops_based.wing_group import WingMassGroup
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft


class MassPremission(om.Group):
    """
    Pre-mission group of top-level mass estimation groups and components for FLOPS-based analysis:
    PayloadGroup, TransportCargoContainersMass, TransportEngineCtrlsMass, TransportAvionicsMass,
    FuelCapacityGroup, EngineMass, etc.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.USE_ALT_MASS)

    def setup(self):
        alt_mass = self.options[Aircraft.Design.USE_ALT_MASS]

        self.add_subsystem('cargo', PayloadGroup(), promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'cargo_containers',
            TransportCargoContainersMass(),
            promotes_inputs=[
                '*',
            ],
            promotes_outputs=[
                '*',
            ],
        )

        self.add_subsystem(
            'engine_controls',
            TransportEngineCtrlsMass(),
            promotes_inputs=[
                '*',
            ],
            promotes_outputs=[
                '*',
            ],
        )

        self.add_subsystem(
            'avionics', TransportAvionicsMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'fuel_capacity_group',
            FuelCapacityGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'engine_mass', EngineMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        if alt_mass:
            self.add_subsystem(
                'fuel_system',
                AltFuelSystemMass(),
                promotes_inputs=[
                    '*',
                ],
                promotes_outputs=[
                    '*',
                ],
            )

            self.add_subsystem(
                'AC', AltAirCondMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

            self.add_subsystem(
                'engine_oil', AltEngineOilMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

            self.add_subsystem(
                'furnishing_base',
                AltFurnishingsGroupMassBase(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'furnishings',
                AltFurnishingsGroupMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'hydraulics',
                AltHydraulicsGroupMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'pass_service',
                AltPassengerServiceMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'unusable_fuel',
                AltUnusableFuelMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'electrical', AltElectricalMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        else:
            self.add_subsystem(
                'fuel_system',
                TransportFuelSystemMass(),
                promotes_inputs=[
                    '*',
                ],
                promotes_outputs=[
                    '*',
                ],
            )

            self.add_subsystem(
                'AC', TransportAirCondMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

            self.add_subsystem(
                'engine_oil',
                TransportEngineOilMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'furnishings',
                TransportFurnishingsGroupMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'hydraulics',
                TransportHydraulicsGroupMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'pass_service',
                PassengerServiceMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'unusable_fuel',
                TransportUnusableFuelMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            self.add_subsystem(
                'electrical', ElectricalMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        self.add_subsystem(
            'starter', TransportStarterMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'anti_icing', AntiIcingMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem('apu', TransportAPUMass(), promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'nonflight_crew', NonFlightCrewMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'flight_crew', FlightCrewMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'instruments', TransportInstrumentMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'misc_engine', EngineMiscMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem('nacelle', NacelleMass(), promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('paint', PaintMass(), promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'thrust_rev', ThrustReverserMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'landing_group', LandingMassGroup(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        if alt_mass:
            self.add_subsystem(
                'surf_ctrl', AltSurfaceControlMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

            self.add_subsystem(
                'fuselage',
                AltFuselageMass(),
                promotes_inputs=[
                    '*',
                ],
                promotes_outputs=[
                    '*',
                ],
            )

            self.add_subsystem(
                'htail',
                AltHorizontalTailMass(),
                promotes_inputs=[
                    '*',
                ],
                promotes_outputs=[
                    '*',
                ],
            )

            self.add_subsystem(
                'vert_tail', AltVerticalTailMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        else:
            self.add_subsystem(
                'surf_ctrl', SurfaceControlMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

            self.add_subsystem(
                'fuselage',
                TransportFuselageMass(),
                promotes_inputs=[
                    '*',
                ],
                promotes_outputs=[
                    '*',
                ],
            )

            self.add_subsystem(
                'htail',
                HorizontalTailMass(),
                promotes_inputs=[
                    '*',
                ],
                promotes_outputs=[
                    '*',
                ],
            )

            self.add_subsystem(
                'vert_tail', VerticalTailMass(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        self.add_subsystem('canard', CanardMass(), promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('fin', FinMass(), promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'wing_group', WingMassGroup(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        self.add_subsystem(
            'total_mass', MassSummation(), promotes_inputs=['*'], promotes_outputs=['*']
        )
