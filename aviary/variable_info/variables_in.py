'''
Dummy explicit component which serves as an input port for all variables in
the aircraft and mission hierarchy.
'''
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.core_promotes import core_mission_inputs


class VariablesIn(om.ExplicitComponent):
    '''
    Provides a central place to connect input variable information to a component
    but doesn't actually do anything on its own.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')
        self.options.declare(
            'meta_data', types=dict, default=_MetaData,
            desc='variable metadata associated with the variables to be passed through this port'
        )
        self.options.declare(
            'context', default='full', values=['full', 'mission'],
            desc='Limit to a subset of the aircraft and mission variables.'
        )

    def setup(self):
        aviary_options: AviaryValues = self.options['aviary_options']
        meta_data = self.options['meta_data']
        context = self.options['context']

        if context == 'mission':
            inputs = core_mission_inputs
        else:
            inputs = meta_data

        for key in inputs:
            # TODO temp line to ignore dynamic mission variables, will not work
            #      if names change to 'dynamic:mission:*'
            if ':' not in key:
                continue
            info = meta_data[key]

            if not info['option'] and ('aircraft:' in key or 'mission:' in key):
                # Since all the variable initial values are stored in aviary_options,
                # we can use the initial values to get the correct shape.
                val = info['default_value']
                if val is None:
                    val = 0.0
                item = val, info['units']
                val, units = aviary_options.get_item(key, item)
                if units == 'unitless' and info['units'] != 'unitless':
                    units = info['units']
                add_aviary_input(self, key, val=val, units=units, meta_data=meta_data)
