from openmdao.core.system import System

from aviary.subsystems.energy.fuel_tank_model import FuelTankModel
from aviary.subsystems.mass.gasp_based.fuel import TankCapacity
from aviary.utils.aviary_values import AviaryValues


class FuelTankGASP(FuelTankModel):
    def build_pre_mission(
        self, aviary_inputs: AviaryValues | None = None, subsystem_options: dict | None = None
    ) -> None | System:
        return TankCapacity()
