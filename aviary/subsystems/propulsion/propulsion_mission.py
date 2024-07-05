import sys

import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


class PropulsionMission(om.Group):
    '''
    Group that tracks all engine models used during mission analysis. Accounts for
    number of engines for each type and returns aircraft-total dynamic values such
    as net thrust and fuel flow rate.
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int, lower=0)

        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )

        self.options.declare(
            'engine_models', types=list, desc='list of EngineModels on aircraft'
        )

    def setup(self):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_models = self.options['engine_models']
        num_engine_type = len(engine_models)

        if num_engine_type > 1:

            # We need a single component with scale_factor. Dymos can't find it when it is
            # already sliced across several component.
            # TODO this only works for engine decks. Need to fix problem in generic way
            comp = om.ExecComp(
                "y=x",
                y={'val': np.ones(num_engine_type), 'units': 'unitless'},
                x={'val': np.ones(num_engine_type), 'units': 'unitless'},
                has_diag_partials=True,
            )
            self.add_subsystem(
                "scale_passthrough",
                comp,
                promotes_inputs=[('x', Aircraft.Engine.SCALE_FACTOR)],
                promotes_outputs=[('y', 'passthrough_scale_factor')],
            )

            for i, engine in enumerate(engine_models):
                self.add_subsystem(
                    engine.name,
                    subsys=engine.build_mission(num_nodes=nn, aviary_inputs=options),
                    promotes_inputs=['*'],
                )

                # split vectorized throttles and connect to the correct engine model
                self.promotes(
                    engine.name,
                    inputs=[Dynamic.Mission.THROTTLE],
                    src_indices=om.slicer[:, i],
                )

                self.promotes(
                    engine.name,
                    inputs=[(Aircraft.Engine.SCALE_FACTOR, 'passthrough_scale_factor')],
                    src_indices=om.slicer[i],
                )

                # TODO if only some engine use hybrid throttle, source vector will have an
                #      index for that engine that is unused, will this confuse optimizer?
                if engine.use_hybrid_throttle:
                    self.promotes(
                        engine.name,
                        inputs=[Dynamic.Mission.HYBRID_THROTTLE],
                        src_indices=om.slicer[:, i],
                    )
        else:
            engine = engine_models[0]

            for i, engine in enumerate(engine_models):
                self.add_subsystem(
                    engine.name,
                    subsys=engine.build_mission(num_nodes=nn, aviary_inputs=options),
                    promotes_inputs=['*'],
                )

                self.promotes(engine.name, inputs=[Dynamic.Mission.THROTTLE])
                if engine.use_hybrid_throttle:
                    self.promotes(engine.name, inputs=[Dynamic.Mission.HYBRID_THROTTLE])

        # TODO might be able to avoid hardcoding using propulsion Enums
        # mux component to vectorize individual outputs into 2d arrays
        perf_mux = om.MuxComp(vec_size=num_engine_type)
        # add each engine data variable to mux component
        perf_mux.add_var(Dynamic.Mission.THRUST, shape=(nn,), axis=1, units='lbf')
        perf_mux.add_var(Dynamic.Mission.THRUST_MAX, shape=(nn,), axis=1, units='lbf')
        perf_mux.add_var(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE, shape=(nn,), axis=1, units='lbm/h'
        )
        perf_mux.add_var(
            Dynamic.Mission.ELECTRIC_POWER_IN, shape=(nn,), axis=1, units='kW'
        )
        perf_mux.add_var(Dynamic.Mission.NOX_RATE, shape=(nn,), axis=1, units='lb/h')
        perf_mux.add_var(
            Dynamic.Mission.TEMPERATURE_T4, shape=(nn,), axis=1, units='degR'
        )
        perf_mux.add_var(Dynamic.Mission.SHAFT_POWER, shape=(nn,), axis=1, units='hp')
        # perf_mux.add_var(
        #     Dynamic.Mission.SHAFT_POWER_CORRECTED,
        #     shape=(nn,),
        #     axis=1,
        #     units='hp'
        # )
        # perf_mux.add_var(
        #     'exit_area_unscaled',
        #     shape=(nn,),
        #     axis=1,
        #     units='ft**2')

        self.add_subsystem(
            'vectorize_performance', subsys=perf_mux, promotes_outputs=['*']
        )

        # connect engine outputs to mux component inputs
        for i, engine in enumerate(engine_models):
            self.connect(
                engine.name + '.' + Dynamic.Mission.THRUST,
                'vectorize_performance.' + Dynamic.Mission.THRUST + '_' + str(i),
            )
            self.connect(
                engine.name + '.' + Dynamic.Mission.THRUST_MAX,
                'vectorize_performance.' + Dynamic.Mission.THRUST_MAX + '_' + str(i),
            )
            self.connect(
                engine.name + '.' + Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                'vectorize_performance.'
                + Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE
                + '_'
                + str(i),
            )
            self.connect(
                engine.name + '.' + Dynamic.Mission.ELECTRIC_POWER_IN,
                'vectorize_performance.'
                + Dynamic.Mission.ELECTRIC_POWER_IN
                + '_'
                + str(i),
            )
            self.connect(
                engine.name + '.' + Dynamic.Mission.NOX_RATE,
                'vectorize_performance.' + Dynamic.Mission.NOX_RATE + '_' + str(i),
            )

            if getattr(engine, 'use_t4', False):
                self.connect(
                    engine.name + '.' + Dynamic.Mission.TEMPERATURE_T4,
                    'vectorize_performance.'
                    + Dynamic.Mission.TEMPERATURE_T4
                    + '_'
                    + str(i),
                )

            if getattr(engine, 'use_shp', False):
                self.connect(
                    engine.name + '.' + Dynamic.Mission.SHAFT_POWER,
                    'vectorize_performance.'
                    + Dynamic.Mission.SHAFT_POWER
                    + '_'
                    + str(i),
                )
                # self.connect(engine.name + '.' + Dynamic.Mission.SHAFT_POWER_CORRECTED,
                #              'vectorize_performance.' + Dynamic.Mission.SHAFT_POWER_CORRECTED + '_' + str(i))

        self.add_subsystem(
            'propulsion_sum',
            subsys=PropulsionSum(num_nodes=nn, aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )


class PropulsionSum(om.ExplicitComponent):
    '''
    Calculates propulsion system level sums of individual engine performance parameters.
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int, lower=0)

        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )

    def setup(self):
        nn = self.options['num_nodes']
        num_engine_type = len(
            self.options['aviary_options'].get_val(Aircraft.Engine.NUM_ENGINES)
        )

        self.add_input(
            Dynamic.Mission.THRUST, val=np.zeros((nn, num_engine_type)), units='lbf'
        )
        self.add_input(
            Dynamic.Mission.THRUST_MAX, val=np.zeros((nn, num_engine_type)), units='lbf'
        )
        self.add_input(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
            val=np.zeros((nn, num_engine_type)),
            units='lbm/h',
        )
        self.add_input(
            Dynamic.Mission.ELECTRIC_POWER_IN,
            val=np.zeros((nn, num_engine_type)),
            units='kW',
        )
        self.add_input(
            Dynamic.Mission.NOX_RATE, val=np.zeros((nn, num_engine_type)), units='lbm/h'
        )

        self.add_output(Dynamic.Mission.THRUST_TOTAL, val=np.zeros(nn), units='lbf')
        self.add_output(Dynamic.Mission.THRUST_MAX_TOTAL, val=np.zeros(nn), units='lbf')
        self.add_output(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            val=np.zeros(nn),
            units='lbm/h',
        )
        self.add_output(
            Dynamic.Mission.ELECTRIC_POWER_IN_TOTAL, val=np.zeros(nn), units='kW'
        )
        self.add_output(Dynamic.Mission.NOX_RATE_TOTAL, val=np.zeros(nn), units='lbm/h')

    def setup_partials(self):
        nn = self.options['num_nodes']
        num_engines = self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES
        )
        num_engine_type = len(num_engines)
        deriv = np.tile(num_engines, nn)

        r = np.repeat(np.arange(nn, dtype=int), num_engine_type)
        c = np.arange(nn * num_engine_type, dtype=int)

        self.declare_partials(
            Dynamic.Mission.THRUST_TOTAL,
            Dynamic.Mission.THRUST,
            val=deriv,
            rows=r,
            cols=c,
        )
        self.declare_partials(
            Dynamic.Mission.THRUST_MAX_TOTAL,
            Dynamic.Mission.THRUST_MAX,
            val=deriv,
            rows=r,
            cols=c,
        )
        self.declare_partials(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
            val=deriv,
            rows=r,
            cols=c,
        )
        self.declare_partials(
            Dynamic.Mission.ELECTRIC_POWER_IN_TOTAL,
            Dynamic.Mission.ELECTRIC_POWER_IN,
            val=deriv,
            rows=r,
            cols=c,
        )
        self.declare_partials(
            Dynamic.Mission.NOX_RATE_TOTAL,
            Dynamic.Mission.NOX_RATE,
            val=deriv,
            rows=r,
            cols=c,
        )

    def compute(self, inputs, outputs):
        num_engines = self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES
        )

        thrust = inputs[Dynamic.Mission.THRUST]
        thrust_max = inputs[Dynamic.Mission.THRUST_MAX]
        fuel_flow = inputs[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE]
        electric = inputs[Dynamic.Mission.ELECTRIC_POWER_IN]
        nox = inputs[Dynamic.Mission.NOX_RATE]

        outputs[Dynamic.Mission.THRUST_TOTAL] = np.dot(thrust, num_engines)
        outputs[Dynamic.Mission.THRUST_MAX_TOTAL] = np.dot(thrust_max, num_engines)
        outputs[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL] = np.dot(
            fuel_flow, num_engines
        )
        outputs[Dynamic.Mission.ELECTRIC_POWER_IN_TOTAL] = np.dot(electric, num_engines)
        outputs[Dynamic.Mission.NOX_RATE_TOTAL] = np.dot(nox, num_engines)
