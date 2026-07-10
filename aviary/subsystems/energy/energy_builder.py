"""
Define subsystem builder for Aviary core energy.

Classes
-------
EnergyBuilder : the interface for a energy subsystem builder.

CoreEnergyBuilder : the interface for Aviary's core energy subsystem builder
"""

# import warnings

import openmdao.api as om

from aviary.mission.utils import separate_reserve_phases
from aviary.subsystems.energy.fuel_summation import FuelSummationGroup
from aviary.subsystems.subsystem_builder import SubsystemBuilder

# from aviary.utils.aviary_values import AviaryValues
# from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variables import Mission


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

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
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

    def get_post_mission_bus_variables(self, aviary_inputs=None, mission_info=None):
        post_mission_bus = {}
        main_phases, reserve_phases = separate_reserve_phases(mission_info)

        post_mission_names = [
            f'{self.name}.fuel_burned.mass_final',
        ]
        if aviary_inputs.get_val(Mission.RESERVE_FUEL_MARGIN) != 0:
            post_mission_names.append(f'{self.name}.reserve_fuel_frac.final_mass')

        post_mission_bus[main_phases[-1]] = {
            f'mass': {
                'post_mission_name': post_mission_names,
                'src_indices': [-1],
            }
        }

        if reserve_phases:
            post_mission_bus[reserve_phases[0]] = {
                f'mass': {
                    'post_mission_name': f'{self.name}.reserve_fuel_burned.mass_initial',
                    'src_indices': [0],
                }
            }
            post_mission_bus[reserve_phases[-1]] = {
                f'mass': {
                    'post_mission_name': f'{self.name}.reserve_fuel_burned.mass_final',
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
