import numpy as np
import openmdao.api as om

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.energy.battery_sizing import SizeBattery
from aviary.variable_info.variables import Aircraft, Dynamic


class BatteryBuilder(SubsystemBuilderBase):
    default_name = 'battery'

    def build_pre_mission(self, aviary_inputs=None):
        return SizeBattery(aviary_inputs=aviary_inputs)

    def get_mass_names(self):
        return [Aircraft.Battery.MASS]

    def build_mission(self, num_nodes, aviary_inputs=None) -> om.Group:
        battery_group = om.Group()
        # Here, the efficiency variable is used as an overall efficiency for the battery
        soc = om.ExecComp('state_of_charge = (energy_capacity - (cumulative_electric_energy_used/efficiency)) / energy_capacity',
                          state_of_charge={'val': np.zeros(
                              num_nodes), 'units': 'unitless'},
                          energy_capacity={'val': 10.0, 'units': 'kJ'},
                          cumulative_electric_energy_used={
                              'val': np.zeros(num_nodes), 'units': 'kJ'},
                          efficiency={'val': 0.95, 'units': 'unitless'})

        battery_group.add_subsystem('state_of_charge',
                                    subsys=soc,
                                    promotes_inputs=[('energy_capacity', Aircraft.Battery.ENERGY_CAPACITY),
                                                     ('cumulative_electric_energy_used',
                                                      Dynamic.Mission.CUMULATIVE_ELECTRIC_ENERGY_USED),
                                                     ('efficiency', Aircraft.Battery.EFFICIENCY)],
                                    promotes_outputs=[('state_of_charge', Dynamic.Mission.BATTERY_STATE_OF_CHARGE)])

        return battery_group

    def get_states(self):
        state_dict = {Dynamic.Mission.CUMULATIVE_ELECTRIC_ENERGY_USED: {'fix_initial': True,
                                                                        'fix_final': False,
                                                                        'lower': 0.0,
                                                                        'ref': 1e4,
                                                                        'defect_ref': 1e6,
                                                                        'units': 'kJ',
                                                                        'rate_source': Dynamic.Mission.ELECTRIC_POWER_IN_TOTAL,
                                                                        'input_initial': 0.0}}

        return state_dict

    def get_constraints(self):
        constraint_dict = {
            # Can add constraints here; state of charge is a common one in many battery applications
            f'battery.{Dynamic.Mission.BATTERY_STATE_OF_CHARGE}':
            {'type': 'boundary',
             'loc': 'final',
             'lower': 0.2},
        }
        return constraint_dict
