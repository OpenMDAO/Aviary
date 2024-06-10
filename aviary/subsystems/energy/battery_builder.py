import numpy as np
import openmdao.api as om

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.energy.battery_sizing import SizeBattery
from aviary.variable_info.variables import Aircraft, Dynamic


class BatteryBuilder(SubsystemBuilderBase):
    def build_pre_mission(self, aviary_inputs=None):
        return SizeBattery(aviary_inputs=aviary_inputs)

    def get_mass_names(self):
        return [Aircraft.Battery.MASS]

    def build_mission(self, num_nodes, aviary_inputs=None) -> om.Group:
        battery_group = om.Group()
        soc = om.ExecComp('state_of_charge = (energy_capacity - (electric_energy/efficiency)) / energy_capacity',
                          state_of_charge={'val': np.zeros(
                              num_nodes), 'units': 'unitless'},
                          energy_capacity={'val': 10.0, 'units': 'kJ'},
                          electric_energy={'val': np.zeros(num_nodes), 'units': 'kJ'},
                          efficiency={'val': 0.95, 'units': 'unitless'})

        battery_group.add_subsystem('state_of_charge',
                                    subsys=soc,
                                    promotes_inputs=[('energy_capacity', Aircraft.Battery.ENERGY_CAPACITY),
                                                     ('electric_energy',
                                                      Dynamic.Mission.ELECTRIC_ENERGY),
                                                     ('efficiency', Aircraft.Battery.EFFICIENCY)],
                                    promotes_outputs=[('state_of_charge', Dynamic.Mission.BATTERY_STATE_OF_CHARGE)])

        return battery_group

    def get_states(self):
        state_dict = {Dynamic.Mission.ELECTRIC_ENERGY: {'fix_initial': True,
                                                        'fix_final': False,
                                                        'lower': 0.0,
                                                        'ref': 1e4,
                                                        'defect_ref': 1e6,
                                                        'units': 'kJ',
                                                        'rate_source': Dynamic.Mission.ELECTRIC_POWER_TOTAL,
                                                        'input_initial': 0.0}}

        return state_dict

    def get_constraints(self):
        # discharge_limit = aviary_options.get_val(Aircraft.Battery.DISCHARGE_LIMIT)
        # constraint_dict = {f'path_{Dynamic.Mission.BATTERY_STATE_OF_CHARGE} = {Dynamic.Mission.BATTERY_STATE_OF_CHARGE} - {Aircraft.Battery.DISCHARGE_LIMIT}':
        #                    {'type': 'path',
        #                     'lower': 0.0}}
        constraint_dict = {Dynamic.Mission.BATTERY_STATE_OF_CHARGE:
                           {'type': 'path',
                            'lower': 0.2}}
        return constraint_dict

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        return {Aircraft.Battery.ENERGY_CAPACITY: {'units': 'kJ', 'val': 1.e4},
                Aircraft.Battery.EFFICIENCY: {'units': 'unitless', 'val': 0.95}}
