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
from aviary.variable_info.variables import Dynamic, Mission, Settings


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

    # get_constraints() seems to only work for mission constraints?
    # def get_constraints(
    #     self,
    #     aviary_inputs: AviaryValues | None = None,
    #     user_options: dict | None = None,
    #     subsystem_options: dict | None = None,
    # ) -> dict:
    #     '''
    #     Returns
    #     -------
    #     dict
    #         A dictionary where the keys are the names of the constraint variables and the values are
    #         dictionaries with the following keys:

    #         - type : str
    #             The type of constraint. Must be one of 'path' or 'boundary'.
    #         - any additional keyword arguments required by OpenMDAO for the constraint
    #           variable.
    #     '''
    #     # determine if the user wants the excess_fuel_capacity constraint active, and if so add it
    #     # to the problem
    #     verbosity = aviary_inputs.get_val(Settings.VERBOSITY)
    #     constraints = {}

    #     if Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT in aviary_inputs:
    #         # ignore_capacity_constraint = aviary_inputs.get_val(
    #         #     Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT, units='unitless'
    #         # )

    #         if not aviary_inputs.get_val(Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT):
    #             constraints[Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY] =
    #             self.add_constraint(
    #                 Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY, lower=0, ref=1.0e5, units='lbm'
    #             )

    #         else:
    #             if verbosity >= Verbosity.BRIEF:
    #                 warnings.warn(
    #                     'Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT = True, therefore '
    #                     'EXCESS_FUEL_MASS_CAPACITY constraint was not added to the Aviary problem. The '
    #                     'aircraft may not have enough space for fuel, so check the value of '
    #                     'Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY for details.'
    #                 )

    #     # else:
    #     #     ignore_capacity_constraint = self.meta_data[
    #     #         Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT
    #     #     ]['default_value']
    #     #     aviary_inputs.set_val(
    #     #         Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT,
    #     #         val=ignore_capacity_constraint,
    #     #         units='unitless',
    #     #     )

    #     return constraints

    ### TODO ###
    ### v-- The following should only happen when fuel is actually present on the aircraft!! --v ###

    def get_states(
        self,
        aviary_inputs: AviaryValues | None = None,
        user_options: dict | None = None,
        subsystem_options: dict | None = None,
    ) -> dict:
        state_dict = {
            Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                'fix_initial': True,
                'fix_final': False,
                'lower': 0.0,
                'ref': -1e4,
                'defect_ref': 1e6,
                'units': 'lbm',
                'rate_source': Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL,
                'input_initial': 0.0,
                # 'targets': Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED,
            }
        }

        return state_dict

    def get_linked_variables(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        # link cumulative fuel burn between phases
        return [Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED]
