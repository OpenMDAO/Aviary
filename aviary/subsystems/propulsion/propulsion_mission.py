import sys

import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


class PropulsionMission(om.Group):
    """
    Group that tracks all engine models used during mission analysis. Accounts for
    number of engines for each type and returns aircraft-total dynamic values such
    as net thrust and fuel flow rate.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, lower=0)

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

    def setup(self):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_models = self.options['engine_models']
        engine_options = self.options['engine_options']
        num_engine_type = len(engine_models)

        if num_engine_type > 1:
            # We need a component to add parameters to problem. Dymos can't find it when
            # it is already sliced across several components.
            # TODO is this problem fixable from dymos end (introspection includes parameters)?

            # create set of params
            # TODO get_parameters() should have access to aviary options + phase info
            param_dict = {}
            # save parameters for use in configure()
            parameters = self.parameters = set()
            for engine in engine_models:
                eng_params = engine.get_parameters()
                param_dict.update(eng_params)

            parameters.update(param_dict.keys())

            # if params exist, create execcomp, fill with placeholder equations
            if len(parameters) != 0:
                comp = om.ExecComp(has_diag_partials=True)

                for i, param in enumerate(parameters):
                    # try to find units information
                    try:
                        units = param_dict[param]['units']
                    except KeyError:
                        units = 'unitless'

                    attrs = {
                        f'x_{i}': {
                            'val': np.ones(num_engine_type),
                            'units': units,
                        },
                        f'y_{i}': {
                            'val': np.ones(num_engine_type),
                            'units': units,
                        },
                    }
                    comp.add_expr(
                        f'y_{i}=x_{i}',
                        **attrs,
                    )

            self.add_subsystem(
                'parameter_passthrough',
                comp,
            )

            for i, engine in enumerate(engine_models):
                kwargs = {}
                if engine.name in engine_options:
                    kwargs = engine_options[engine.name]
                self.add_subsystem(
                    engine.name,
                    subsys=engine.build_mission(num_nodes=nn, aviary_inputs=options, **kwargs),
                    promotes_inputs=['*'],
                )

                # split vectorized throttles and connect to the correct engine model
                self.promotes(
                    engine.name,
                    inputs=[Dynamic.Vehicle.Propulsion.THROTTLE],
                    src_indices=om.slicer[:, i],
                )

                # loop through params and slice as needed
                params = engine.get_parameters()
                for param in params:
                    self.promotes(
                        engine.name,
                        inputs=[(param, param + '_passthrough')],
                        src_indices=om.slicer[i],
                    )

                # TODO if only some engine use hybrid throttle, source vector will have an
                #      index for that engine that is unused, will this confuse optimizer?
                if engine.use_hybrid_throttle:
                    self.promotes(
                        engine.name,
                        inputs=[Dynamic.Vehicle.Propulsion.HYBRID_THROTTLE],
                        src_indices=om.slicer[:, i],
                    )
        else:
            engine = engine_models[0]
            kwargs = {}
            if engine.name in engine_options:
                kwargs = engine_options[engine.name]
            self.add_subsystem(
                engine.name,
                subsys=engine.build_mission(num_nodes=nn, aviary_inputs=options, **kwargs),
                promotes_inputs=['*'],
            )

            self.promotes(engine.name, inputs=[Dynamic.Vehicle.Propulsion.THROTTLE])

            if engine.use_hybrid_throttle:
                self.promotes(engine.name, inputs=[Dynamic.Vehicle.Propulsion.HYBRID_THROTTLE])

        # TODO might be able to avoid hardcoding using propulsion Enums
        # mux component to vectorize individual engine outputs into 2d arrays
        perf_mux = om.MuxComp(vec_size=num_engine_type)
        # add each engine data variable to mux component
        perf_mux.add_var(Dynamic.Vehicle.Propulsion.THRUST, val=0, shape=(nn,), axis=1, units='lbf')
        perf_mux.add_var(
            Dynamic.Vehicle.Propulsion.THRUST_MAX,
            val=0,
            shape=(nn,),
            axis=1,
            units='lbf',
        )
        perf_mux.add_var(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
            val=0,
            shape=(nn,),
            axis=1,
            units='lbm/h',
        )
        perf_mux.add_var(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
            val=0,
            shape=(nn,),
            axis=1,
            units='kW',
        )
        perf_mux.add_var(
            Dynamic.Vehicle.Propulsion.NOX_RATE,
            val=0,
            shape=(nn,),
            axis=1,
            units='lb/h',
        )
        perf_mux.add_var(
            Dynamic.Vehicle.Propulsion.TEMPERATURE_T4,
            val=0,
            shape=(nn,),
            axis=1,
            units='degR',
        )
        perf_mux.add_var(
            Dynamic.Vehicle.Propulsion.SHAFT_POWER,
            val=0,
            shape=(nn,),
            axis=1,
            units='hp',
        )
        perf_mux.add_var(
            Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX,
            val=0,
            shape=(nn,),
            axis=1,
            units='hp',
        )
        # perf_mux.add_var(
        #     'exit_area_unscaled',
        #     shape=(nn,),
        #     axis=1,
        #     units='ft**2')

        self.add_subsystem('vectorize_performance', subsys=perf_mux, promotes_outputs=['*'])

        self.add_subsystem(
            'propulsion_sum',
            subsys=PropulsionSum(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

    def configure(self):
        # Special configure step needed to handle multiple, unique engine models.
        # Handle checking each EngineModel for compatible outputs with
        # vectorize_performance component and connecting those outputs

        # TODO this list shouldn't be hardcoded so it can be extended by users
        supported_outputs = [
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
            Dynamic.Vehicle.Propulsion.NOX_RATE,
            Dynamic.Vehicle.Propulsion.SHAFT_POWER,
            Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX,
            Dynamic.Vehicle.Propulsion.TEMPERATURE_T4,
            Dynamic.Vehicle.Propulsion.THRUST,
            Dynamic.Vehicle.Propulsion.THRUST_MAX,
        ]

        engine_models = self.options['engine_models']
        engine_names = [engine.name for engine in engine_models]
        num_engine_type = len(engine_models)

        # determine if openMDAO messages and warnings should be suppressed
        verbosity = self.options['aviary_options'].get_val(Settings.VERBOSITY)
        out_stream = None
        # DEBUG
        if verbosity.value > 2:
            out_stream = sys.stdout

        comp_list = [self._get_subsystem(engine) for engine in engine_names]

        # dictionaries of outputs for each engine in prop mission
        output_dict = {}
        # Dictionary of all unique inputs/outputs from all new components, keys are
        # units for each var
        unique_outputs = {}

        # idx to be used for slicing inputs in next round of improvements
        for idx, comp in enumerate(comp_list):
            # identify outputs to connect to muxcomp
            comp_outputs = comp.list_outputs(
                return_format='dict', units=True, out_stream=out_stream, all_procs=True
            )
            # grab only outputs that have been promoted out of component
            promoted_outputs = [
                key for key in comp_outputs if '.' not in comp_outputs[key]['prom_name']
            ]
            output_dict[comp.name] = dict(
                (comp_outputs[key]['prom_name'], comp_outputs[key]) for key in promoted_outputs
            )
            unique_outputs.update(
                [
                    (
                        comp_outputs[key]['prom_name'],
                        comp_outputs[key]['units'],
                    )
                    for key in promoted_outputs
                ]
            )

        # add variables to the mux component and make connections to individual
        # component outputs
        # if num_engine_type > 1:
        for output in unique_outputs:
            if output in supported_outputs:
                # self.vectorize_performance.add_var(output, units=unique_outputs[output])
                # promote/alias outputs for each comp that has relevant outputs
                for i, comp in enumerate(output_dict):
                    if output in output_dict[comp]:
                        # if this component provides the output, connect it to the correct mux input
                        self.connect(
                            comp + '.' + output,
                            'vectorize_performance.' + output + '_' + str(i),
                        )
            # TODO handle setting of other variables from engine outputs (e.g. Aircraft.Engine.****)

        if num_engine_type > 1:
            # commented out block of code is for experimenting with automatically finding
            # inputs that need a passthrough, rather than relying on get_parameters()
            # being properly set up

            # custom promote parameters with aliasing to connect to passthrough component
            # for engine in engine_models:
            # get inputs to engine model
            # engine_comp = self._get_subsystem(engine.name)
            # input_dict = engine_comp.list_inputs(
            #     return_format='dict', units=True, out_stream=None, all_procs=True
            # )
            # # TODO this catches not fully promoted variables are caught - is this
            # #      wanted?
            # input_list = list(
            #     set(
            #         input_dict[key]['prom_name']
            #         for key in input_dict
            #         if '.' not in input_dict[key]['prom_name']
            #     )
            # )
            # promotions = []
            for i, param in enumerate(self.parameters):
                self.promotes(
                    'parameter_passthrough',
                    inputs=[(f'x_{i}', param)],
                    outputs=[(f'y_{i}', param + '_passthrough')],
                )
                #     if param in input_dict:
                #         promotions.append((param, param + '_passthrough'))
                # self.promotes(engine.name, inputs=promotions)


class PropulsionSum(om.ExplicitComponent):
    """Calculates propulsion system level sums of individual engine performance parameters."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, lower=0)
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        nn = self.options['num_nodes']
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        self.add_input(
            Dynamic.Vehicle.Propulsion.THRUST,
            val=np.zeros((nn, num_engine_type)),
            units='lbf',
        )
        self.add_input(
            Dynamic.Vehicle.Propulsion.THRUST_MAX,
            val=np.zeros((nn, num_engine_type)),
            units='lbf',
        )
        self.add_input(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
            val=np.zeros((nn, num_engine_type)),
            units='lbm/h',
        )
        self.add_input(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
            val=np.zeros((nn, num_engine_type)),
            units='kW',
        )
        self.add_input(
            Dynamic.Vehicle.Propulsion.NOX_RATE,
            val=np.zeros((nn, num_engine_type)),
            units='lbm/h',
        )

        self.add_output(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, val=np.zeros(nn), units='lbf')
        self.add_output(
            Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
            val=np.zeros(nn),
            units='lbf',
        )
        self.add_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            val=np.zeros(nn),
            units='lbm/h',
        )
        self.add_output(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
            val=np.zeros(nn),
            units='kW',
        )
        self.add_output(Dynamic.Vehicle.Propulsion.NOX_RATE_TOTAL, val=np.zeros(nn), units='lbm/h')

    def setup_partials(self):
        nn = self.options['num_nodes']
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        num_engine_type = len(num_engines)
        deriv = np.tile(num_engines, nn)

        r = np.repeat(np.arange(nn, dtype=int), num_engine_type)
        c = np.arange(nn * num_engine_type, dtype=int)

        self.declare_partials(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            Dynamic.Vehicle.Propulsion.THRUST,
            val=deriv,
            rows=r,
            cols=c,
        )
        self.declare_partials(
            Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
            Dynamic.Vehicle.Propulsion.THRUST_MAX,
            val=deriv,
            rows=r,
            cols=c,
        )
        self.declare_partials(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
            val=deriv,
            rows=r,
            cols=c,
        )
        self.declare_partials(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
            val=deriv,
            rows=r,
            cols=c,
        )
        self.declare_partials(
            Dynamic.Vehicle.Propulsion.NOX_RATE_TOTAL,
            Dynamic.Vehicle.Propulsion.NOX_RATE,
            val=deriv,
            rows=r,
            cols=c,
        )

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST]
        thrust_max = inputs[Dynamic.Vehicle.Propulsion.THRUST_MAX]
        fuel_flow = inputs[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE]
        electric = inputs[Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN]
        nox = inputs[Dynamic.Vehicle.Propulsion.NOX_RATE]

        outputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = np.dot(thrust, num_engines)
        outputs[Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL] = np.dot(thrust_max, num_engines)
        outputs[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL] = np.dot(
            fuel_flow, num_engines
        )
        outputs[Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL] = np.dot(electric, num_engines)
        outputs[Dynamic.Vehicle.Propulsion.NOX_RATE_TOTAL] = np.dot(nox, num_engines)
