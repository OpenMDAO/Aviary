"""
Utilities to create a component that converts an option into an output.
"""

import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_output


def create_opts2vals(all_options: list, output_units: dict = {}):
    """
    create_opts2vals creates a component that converts options to outputs.

    Parameters
    ----------
    all_options : list of strings
        Each string is the name of an option in aviary_options.
    output_units : dict of units, optional
        This optional input allows the user to specify the units that will be used while
        adding the outputs. Only the outputs that shouldn't use their default units need
        to be specified. Each key should match one of the names in all_options, and each
        value must be a string representing a valid unit in openMDAO.

    Returns
    -------
    OptionsToValues : ExplicitComponent
        An explicit component that takes in an AviaryValues object that contains any
        options that need to be converted to outputs. There are no inputs to this
        component, only outputs. If the resulting component is added directly to a
        Group, the output variables will have the same name as the options they
        represent. If you need to rename them to prevent conflicting names in the
        group, running add_opts2vals will add the prefix "option:" to the name.
    """

    def configure_output(option_name: str, aviary_options: AviaryValues):
        option_data = aviary_options.get_item(option_name)
        out_units = (
            output_units[option_name] if option_name in output_units.keys() else option_data[1]
        )
        return {'val': option_data[0], 'units': out_units}

    class OptionsToValues(om.ExplicitComponent):
        def initialize(self):
            self.options.declare(
                'aviary_options',
                types=AviaryValues,
                desc='collection of Aircraft/Mission specific options',
            )

        def setup(self):
            for option_name in all_options:
                output_data = configure_output(option_name, self.options['aviary_options'])
                add_aviary_output(
                    self,
                    option_name,
                    val=output_data['val'],
                    units=output_data['units'],
                )

        def compute(self, inputs, outputs):
            aviary_options: AviaryValues = self.options['aviary_options']
            for option_name in all_options:
                output_data = configure_output(option_name, aviary_options)
                # uses default value if not present
                if option_name in aviary_options:
                    outputs[option_name] = aviary_options.get_val(
                        option_name, units=output_data['units']
                    )

    return OptionsToValues


def add_opts2vals(Group: om.Group, OptionsToValues, aviary_options: AviaryValues):
    """
    Add the OptionsToValues component to the specified Group.

    Parameters
    ----------
    Group : Group
        The group or model the component should be added to.
    OptionsToValues : ExplicitComponent
        This is the explicit component that was created by create_opts2vals.
    aviary_options : AviaryValues
        aviary_options is an AviaryValues object that contains all of the options
        that need to be converted to outputs.

    Returns
    -------
    Opts2Vals : Group
        A group that wraps the OptionsToValues component in order to rename its
        variables with a prefix to keep them separate from any similarly named
        variables in the original group the component is being added to.
    """

    class Opts2Vals(om.Group):
        def initialize(self):
            self.options.declare(
                'aviary_options',
                types=AviaryValues,
                desc='collection of Aircraft/Mission specific options',
            )

        def setup(self):
            self.add_subsystem('options_to_values', OptionsToValues(aviary_options=aviary_options))

        def configure(self):
            all_output_data = self.options_to_values.list_outputs(out_stream=None)
            list_of_outputs = [(name, 'option:' + name) for name, data in all_output_data]
            self.promotes('options_to_values', list_of_outputs)

    Group.add_subsystem(
        'opts2vals', Opts2Vals(aviary_options=aviary_options), promotes_outputs=['*']
    )

    return Group


