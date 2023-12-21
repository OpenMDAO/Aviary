import cmath

import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.engine_sizing import SizeEngine
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class PropulsionPreMission(om.Group):
    '''
    Group that contains propulsion calculations for pre-mission analysis, such as
    computing scaling factors, and sums propulsion-system level totals.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        options = self.options['aviary_options']
        engine_models = options.get_val('engine_models')

        for engine in engine_models:
            self.add_subsystem(engine.name,
                               subsys=engine.build_pre_mission(options),
                               promotes_inputs=['*'],
                               promotes_outputs=['*'],
                               )

        self.add_subsystem(
            'propulsion_sum',
            subsys=PropulsionSum(
                aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=['*']
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
        count = len(self.options['aviary_options'].get_val('engine_models'))

        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST, val=np.zeros((count)))

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
