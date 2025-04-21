import sys

import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings


class PropulsionPreMission(om.Group):
    """
    Group that contains propulsion calculations for pre-mission analysis, such as
    computing scaling factors, and sums propulsion-system level totals.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )

        self.options.declare('engine_models', types=list, desc='list of EngineModels on aircraft')

        # engine options is optional
        self.options.declare(
            'engine_options',
            types=dict,
            default={},
            desc='dictionary of options for each EngineModel',
        )

        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        aviary_options = self.options['aviary_options']
        engine_models = self.options['engine_models']
        engine_options = self.options['engine_options']
        num_engine_type = len(engine_models)

        # Each engine model pre_mission component only needs to accept and output single
        # value relevant to that variable - this group's configure step will handle
        # promoting/connecting just the relevant index in vectorized inputs/outputs for
        # each component here
        # Promotions are handled in configure()
        for engine in engine_models:
            options = {}
            if engine.name in engine_options:
                options = engine_options[engine.name]
            subsys = engine.build_pre_mission(aviary_options, **options)
            if subsys:
                if num_engine_type > 1:
                    proms = None
                else:
                    proms = ['*']
                self.add_subsystem(
                    engine.name,
                    subsys=subsys,
                    promotes_outputs=proms,
                )

        if num_engine_type > 1:
            # Add an empty mux comp, which will be customized to handle all required
            # outputs in configure()
            self.add_subsystem('pre_mission_mux', subsys=om.MuxComp(), promotes_outputs=['*'])

        self.add_subsystem(
            'propulsion_sum',
            subsys=PropulsionSum(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

    def configure(self):
        # Special configure step needed to handle multiple, unique engine models.
        # Each engine's pre_mission component should only handle single instance of engine,
        # so vectorized inputs/outputs are a problem. Slice all needed vector inputs and pass
        # pre_mission components only the value they need, then mux all the outputs back together

        engine_models = self.options['engine_models']
        num_engine_type = len(engine_models)

        # determine if openMDAO messages and warnings should be suppressed
        verbosity = self.options[Settings.VERBOSITY]
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
        # unique_inputs = {}

        # dictionaries of inputs/outputs for engine in prop pre-mission
        input_dict = {}
        output_dict = {}

        for idx, engine_model in enumerate(engine_models):
            engine = self._get_subsystem(engine_model.name)
            # Patterns to identify which inputs/outputs are vectorized and need to be
            # split then re-muxed
            pattern = ['engine:', 'nacelle:']

            # pull out all inputs (in dict format) in component
            eng_inputs = engine.list_inputs(
                return_format='dict',
                units=True,
                out_stream=out_stream,
                all_procs=True,
            )
            # switch dictionary keys to promoted name rather than full path
            # only handle variables that were top-level promoted inside engine model
            eng_inputs = dict(
                [
                    (eng_inputs[key]['prom_name'], eng_inputs[key])
                    for key in eng_inputs
                    if '.' not in eng_inputs[key]['prom_name']
                ]
            )
            # only keep inputs if they contain the pattern
            input_dict[engine.name] = dict(
                (key, eng_inputs[key]) for key in eng_inputs if any([x in key for x in pattern])
            )

            # do the same thing with outputs
            eng_outputs = engine.list_outputs(
                return_format='dict', units=True, out_stream=out_stream, all_procs=True
            )
            eng_outputs = dict(
                [
                    (eng_outputs[key]['prom_name'], eng_outputs[key])
                    for key in eng_outputs
                    if '.' not in eng_outputs[key]['prom_name']
                ]
            )
            output_dict[engine.name] = dict(
                (key, eng_outputs[key]) for key in eng_outputs if any([x in key for x in pattern])
            )
            unique_outputs.update(
                [
                    (
                        key,
                        output_dict[engine.name][key]['units'],
                    )
                    for key in output_dict[engine.name]
                ]
            )

            # slice incoming inputs for this engine, so it only gets the correct index
            if num_engine_type > 1:
                src_indices = om.slicer[idx]
            else:
                src_indices = None

            self.promotes(
                engine.name,
                inputs=[*input_dict[engine.name]],
                src_indices=src_indices,
            )

            # promote all other inputs/outputs for this engine normally (handle vectorized outputs later)
            self.promotes(
                engine.name,
                inputs=[input for input in eng_inputs if input not in input_dict[engine.name]],
                outputs=[
                    output for output in eng_outputs if output not in output_dict[engine.name]
                ],
            )

        # add variables to the mux component and make connections to individual
        # component outputs
        if num_engine_type > 1:
            for output in unique_outputs:
                self.pre_mission_mux.add_var(output, units=unique_outputs[output])
                # promote/alias outputs for each comp that has relevant outputs
                for i, engine in enumerate(output_dict):
                    if output in output_dict[engine]:
                        # if this component provides the output, connect it to the correct mux input
                        self.connect(
                            engine + '.' + output,
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
    """Calculates propulsion system level sums of individual engine performance parameters."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST, val=np.zeros(num_engine_type))

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=0.0)

    def setup_partials(self):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        self.declare_partials(
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
            Aircraft.Engine.SCALED_SLS_THRUST,
            val=num_engines,
        )

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        outputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST] = np.dot(thrust, num_engines)
