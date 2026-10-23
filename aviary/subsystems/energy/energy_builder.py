"""
Define subsystem builder for Aviary core energy.

Classes
-------
EnergyBuilder : the interface for a energy subsystem builder.

CoreEnergyBuilder : the interface for Aviary's core energy subsystem builder
"""

# import warnings

import openmdao.api as om
from openmdao.core.system import System

from aviary.mission.utils import separate_reserve_phases
from aviary.subsystems.energy.flops_based.flops_tank import FuelTankFLOPS
from aviary.subsystems.energy.fuel_summation import FuelSummationGroup
from aviary.subsystems.energy.gasp_based.gasp_tank import FuelTankGASP
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Dynamic, Settings


class EnergyBuilder(SubsystemBuilder):
    """
    Base energy builder.

    Methods
    -------
    __init__(self, name=None, meta_data=None):
        Initializes the EnergyBuilder object with a given name.
    """

    _default_name = 'energy'


class CoreEnergyBuilder(EnergyBuilder):
    """Core energy subsystem builder."""

    def build_pre_mission(
        self, aviary_inputs: AviaryValues | None = None, subsystem_options: dict | None = None
    ) -> None | System:
        if aviary_inputs.get_val(Settings.MASS_METHOD) is LegacyCode.FLOPS:
            tank = FuelTankFLOPS(meta_data=self.meta_data)
            # TODO subsystem options needs sub-dict like propulsion
            return tank.build_pre_mission(
                aviary_inputs=aviary_inputs, subsystem_options=subsystem_options
            )
        if aviary_inputs.get_val(Settings.MASS_METHOD) is LegacyCode.GASP:
            tank = FuelTankGASP(meta_data=self.meta_data)
            # TODO subsystem options needs sub-dict like propulsion
            return tank.build_pre_mission(
                aviary_inputs=aviary_inputs, subsystem_options=subsystem_options
            )

    def build_post_mission(
        self,
        aviary_inputs: AviaryValues | None = None,
        mission_info=None,
        subsystem_options: dict | None = None,
        phase_mission_bus_lengths: dict | None = None,
    ) -> None | System:
        energy_group = om.Group()
        fuelgroup = FuelSummationGroup(mission_info=mission_info)

        energy_group.add_subsystem('fuel_summation', fuelgroup, promotes=['*'])

        return energy_group

        # def report(self, prob, reports_folder, **kwargs):
        #     """
        #     Generate the report for Aviary core energy.

        #     Parameters
        #     ----------
        #     prob : AviaryProblem
        #         The AviaryProblem that will be used to generate the report
        #     reports_folder : Path
        #         Location of the subsystems_report folder this report will be placed in
        #     """

    def get_post_mission_bus_variables(
        self, aviary_inputs: AviaryValues | None = None, mission_info: dict | None = None
    ) -> dict:
        post_mission_bus = {}
        main_phases, reserve_phases = separate_reserve_phases(mission_info)

        post_mission_bus[main_phases[-1]] = {
            Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                'post_mission_name': f'{self.name}.main_mission_fuel.mission_fuel',
                'src_indices': [-1],
            }
        }

        if reserve_phases:
            post_mission_bus[reserve_phases[0]] = {
                Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                    'post_mission_name': f'{self.name}.reserve_mission_fuel.fuel_initial',
                    'src_indices': [0],
                }
            }

            post_mission_bus[reserve_phases[-1]] = {
                Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                    'post_mission_name': f'{self.name}.reserve_mission_fuel.fuel_final',
                    'src_indices': [-1],
                }
            }

        return post_mission_bus

    ### TODO ###
    # v-- The following should only happen when fuel is actually present on the aircraft!! --v

    def get_states(
        self,
        aviary_inputs: AviaryValues | None = None,
        user_options: dict | None = None,
        subsystem_options: dict | None = None,
    ) -> dict:
        mass_ref = user_options.get_val('mass_ref', units='kg')
        if mass_ref is None:
            mass_ref = 1.0e4
        state_dict = {
            Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                'fix_initial': True,
                'fix_final': False,
                'lower': 0.0,
                'ref': mass_ref,
                'units': 'kg',
                'rate_source': Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_TOTAL,
                'val': 0.0,
                # 'targets': Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED,
            }
        }

        return state_dict

    def get_linked_variables(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        # link cumulative fuel burn between phases
        return [Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED]
