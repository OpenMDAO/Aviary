"""
Define utilities for building fuel tank models.

Classes
-------
TankModel : the interface for a fuel tank model builder.
"""

import openmdao.api as om
from openmdao.core.system import System

from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import FuelType, LegacyCode, Verbosity
from aviary.variable_info.variables import Settings


class FuelTankModel(SubsystemBuilder):
    """
    Define the interface for a fuel tank model builder.

    Attributes
    ----------
    name : str ('tank_model')
        object label.
    options : AviaryValues (<empty>)
        inputs and options related to engine model.

    Methods
    -------
    build_pre_mission
    build_mission
    build_post_mission
    get_val
    get_item
    set_val
    update
    """

    _default_name = 'tank_model'

    def __init__(
        self,
        name: str | None = None,
        meta_data: dict | None = None,
        fuel_type: FuelType = FuelType.JET_A,
        **kwargs,
    ):
        # by default, name tank after fuel type
        if name is None:
            name = fuel_type.value + '_tank_model'

        super().__init__(name, meta_data=meta_data)

        self.fuel_type = fuel_type

    # def build_pre_mission(
    #     self, aviary_inputs: AviaryValues | None = None, subsystem_options: dict | None = None
    # ) -> None | System:
    #     if aviary_inputs.get_val(Settings.MASS_METHOD) is LegacyCode.FLOPS:
    #         pre_mission = om.Group()
    #         return pre_mission
    #     elif aviary_inputs.get_val(Settings.MASS_METHOD) is LegacyCode.GASP:
    #         pre_mission = om.Group()
    #         return pre_mission
    #     else:
    #         return None

    def build_mission(
        self,
        num_nodes: int,
        aviary_inputs: AviaryValues | None = None,
        user_options: dict | None = None,
        subsystem_options: dict | None = None,
    ) -> None | System:
        return super().build_mission(num_nodes, aviary_inputs, user_options, subsystem_options)
