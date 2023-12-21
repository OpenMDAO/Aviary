import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.named_values import NamedValues, get_keys, get_items


class DataInterpolator(om.Group):
    '''
    Group that contains interpolators that get passed training data directly through
    openMDAO connections
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare(
            'aviary_options', types=AviaryValues,
            default=None,
            desc='Collection of Aircraft/Mission specific options')

        self.options.declare(
            'interpolator_inputs', types=NamedValues,
            desc='NamedValues object containing data for the independent variables of '
                 'interpolation, including name, value, and units'
        )

        self.options.declare(
            'interpolator_outputs', types=dict,
            desc='Dictionary describing which variables will be avaliable to the '
                 'interpolator as training data at runtime, and their units'
        )

        self.options.declare(
            'interpolation_method',
            values=['slinear', 'lagrange2', 'lagrange3', 'akima'],
            default='slinear',
            desc='Interpolation method for metamodel'
        )

    def setup(self):
        num_nodes = self.options['num_nodes']
        input_data = self.options['interpolator_inputs']
        output_data = self.options['interpolator_outputs']
        interp_method = self.options['interpolation_method']

        # interpolator object for engine data
        engine = om.MetaModelSemiStructuredComp(
            method=interp_method, extrapolate=True,
            vec_size=num_nodes, training_data_gradients=True)

        # Calculation of max thrust currently done with a duplicate of the engine
        # model and scaling components
        max_thrust_engine = om.MetaModelSemiStructuredComp(
            method=interp_method, extrapolate=True,
            vec_size=num_nodes, training_data_gradients=True)

        # check that data in table are all vectors of the same length
        for idx, item in enumerate(get_items(input_data)):
            val = item[1][0]
            if idx != 0:
                prev_model_length = model_length
            else:
                prev_model_length = len(val)
            model_length = len(val)
            if model_length != prev_model_length:
                raise IndexError('Lengths of data provided for engine performance '
                                 'interpolation do not match.')

        # add inputs and outputs to interpolator
        for input in get_keys(input_data):
            values, units = input_data.get_item(input)
            engine.add_input(input,
                             training_data=values,
                             units=units)

            if input == 'throttle':
                input = 'throttle_max'
            if input == 'hybrid_throttle':
                input = 'hybrid_throttle_max'
            max_thrust_engine.add_input(input,
                                        training_data=values,
                                        units=units)

        for output in output_data:
            engine.add_output(output,
                              training_data=np.zeros(model_length),
                              units=output_data[output])
            if output == 'thrust_net':
                max_thrust_engine.add_output('thrust_net_max',
                                             training_data=np.zeros(model_length),
                                             units=output_data[output])

        # create IndepVarComp to pass maximum throttle is to max thrust interpolator
        # currently assuming max throttle and max hybrid throttle is always 1 at every
        #   flight condition
        fixed_throttles = om.IndepVarComp()
        fixed_throttles.add_output('throttle_max',
                                   val=np.ones(num_nodes),
                                   units='unitless',
                                   desc='Engine maximum throttle')
        if 'hybrid_throttle' in input_data:
            fixed_throttles.add_output('hybrid_throttle_max',
                                       val=np.ones(num_nodes),
                                       units='unitless',
                                       desc='Engine maximum hybrid throttle')

        # add created subsystems to engine_group
        self.add_subsystem('interpolation',
                           engine,
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem('fixed_max_throttles',
                           fixed_throttles,
                           promotes_outputs=['*'])

        self.add_subsystem('max_thrust_interpolation',
                           max_thrust_engine,
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
