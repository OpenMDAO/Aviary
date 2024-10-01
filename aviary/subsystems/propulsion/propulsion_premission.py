import sys

import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings
from aviary.variable_info.enums import Verbosity


class PropulsionPreMission(om.Group):
    '''
    Group that contains propulsion calculations for pre-mission analysis, such as
    computing scaling factors, and sums propulsion-system level totals.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')
        self.options.declare(
            'engine_models', types=list,
            desc='list of EngineModels on aircraft'
        )

    def setup(self):
        options = self.options['aviary_options']
        engine_models = self.options['engine_models']
        num_engine_type = len(engine_models)

        # Each engine model pre_mission component only needs to accept and output single
        # value relevant to that variable - this group's configure step will handle
        # promoting/connecting just the relevant index in vectorized inputs/outputs for
        # each component here
        # Promotions are handled in configure()
        for engine in engine_models:
            subsys = engine.build_pre_mission(options)
            if subsys:
                if num_engine_type > 1:
                    proms = None
                else:
                    proms = ['*']
                self.add_subsystem(engine.name,
                                   subsys=subsys,
                                   promotes_outputs=proms,
                                   )

        if num_engine_type > 1:
            # Add an empty mux comp, which will be customized to handle all required
            # outputs in configure()
            self.add_subsystem(
                'pre_mission_mux',
                subsys=om.MuxComp(),
                promotes_outputs=['*']
            )

        self.add_subsystem(
            'propulsion_sum',
            subsys=PropulsionSum(
                aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=['*']
        )

    def configure(self):
        engine_models = self.options['engine_models']
        num_engine_type = len(engine_models)

        # determine if openMDAO messages and warnings should be suppressed
        verbosity = self.options['aviary_options'].get_val(Settings.VERBOSITY)
        out_stream = None

        # DEBUG
        if verbosity > Verbosity.VERBOSE:
            out_stream = sys.stdout

        # Patterns to identify which inputs/outputs are vectorized and need to be
        # split then re-muxed
        pattern = ['engine:', 'nacelle:']

        # Dictionary of all unique inputs/outputs from all new components, keys are
        # units for each var
        unique_outputs = {}
        unique_inputs = {}

        # dictionaries of inputs/outputs for engine in prop pre-mission
        input_dict = {}
        output_dict = {}

        for idx, engine in enumerate(engine_models):
            eng_model = self._get_subsystem(engine.name)

            # pull out all inputs (in dict format) in component
            eng_inputs = eng_model.list_inputs(
                return_format='dict', units=True, out_stream=out_stream, all_procs=True
            )
            # only keep inputs if they contain the pattern
            input_dict[engine.name] = dict(
                (key, eng_inputs[key])
                for key in eng_inputs
                if any([x in key for x in pattern])
            )
            # Track list of ALL inputs present in prop pre-mission in a "flat" dict.
            # Repeating inputs will just override what's already in the dict - we don't
            # care if units get overridden, if they differ openMDAO will convert
            # (if they aren't compatible, then a component specified the wrong units and
            # needs to be fixed there)
            unique_inputs.update(
                [
                    (key, input_dict[engine.name][key]['units'])
                    for key in input_dict[engine.name]
                ]
            )

            # do the same thing with outputs
            eng_outputs = eng_model.list_outputs(
                return_format='dict', units=True, out_stream=out_stream, all_procs=True
            )
            output_dict[engine.name] = dict(
                (key, eng_outputs[key])
                for key in eng_outputs
                if any([x in key for x in pattern])
            )
            unique_outputs.update(
                [
                    (key, output_dict[engine.name][key]['units'])
                    for key in output_dict[engine.name]
                ]
            )

            # slice incoming inputs for this component, so it only gets the correct index
            self.promotes(
                engine.name,
                inputs=input_dict[engine.name].keys(),
                src_indices=om.slicer[idx],
            )

            # promote all other inputs/outputs for this component normally (handle vectorized outputs later)
            self.promotes(
                engine.name,
                inputs=[
                    eng_inputs[input]['prom_name']
                    for input in eng_inputs
                    if input not in input_dict[engine.name]
                ],
                outputs=[
                    eng_outputs[output]['prom_name']
                    for output in eng_outputs
                    if output not in output_dict[engine.name]
                ],
            )

        # add variables to the mux component and make connections to individual
        # component outputs
        if num_engine_type > 1:
            for output in unique_outputs:
                self.pre_mission_mux.add_var(output, units=unique_outputs[output])
                # promote/alias outputs for each comp that has relevant outputs
                for i, eng in enumerate(output_dict):
                    if output in output_dict[engine.name]:
                        # if this component provides the output, connect it to the correct mux input
                        self.connect(
                            eng + '.' + output,
                            'pre_mission_mux.' + output + '_' + str(i),
                        )
                    else:
                        # If this component does not provide the output, pass the existing
                        # value for that index to the mux
                        self.connect(
                            output,
                            'pre_mission_mux.' + output + '_' + str(i),
                            src_indices=om.slicer[i],
                        )


class PropulsionSum(om.ExplicitComponent):
    '''
    Calculates propulsion system level sums of individual engine performance parameters.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        num_engine_type = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))

        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST,
                         val=np.zeros(num_engine_type))

        add_aviary_output(
            self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=0.0)

    def setup_partials(self):
        num_engines = self.options['aviary_options'].get_val(Aircraft.Engine.NUM_ENGINES)

        self.declare_partials(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
                              Aircraft.Engine.SCALED_SLS_THRUST, val=num_engines)

    def compute(self, inputs, outputs):
        num_engines = self.options['aviary_options'].get_val(Aircraft.Engine.NUM_ENGINES)

        thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        outputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST] = np.dot(
            thrust, num_engines)
