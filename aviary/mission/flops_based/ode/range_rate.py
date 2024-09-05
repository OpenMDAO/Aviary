import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic


class RangeRate(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            Dynamic.Atmosphere.ALTITUDE_RATE,
            val=np.ones(nn),
            desc='climb rate',
            units='m/s',
        )
        self.add_input(
            Dynamic.Atmosphere.VELOCITY,
            val=np.ones(nn),
            desc='current velocity',
            units='m/s',
        )
        self.add_output(
            Dynamic.Mission.DISTANCE_RATE,
            val=np.ones(nn),
            desc='current horizontal velocity (assumed no wind)',
            units='m/s')

    def compute(self, inputs, outputs):
        climb_rate = inputs[Dynamic.Atmosphere.ALTITUDEUDE_RATE]
        velocity = inputs[Dynamic.Atmosphere.VELOCITY]
        climb_rate_2 = climb_rate**2
        velocity_2 = velocity**2
        if (climb_rate_2 >= velocity_2).any():
            raise om.AnalysisError(
                "WARNING: climb rate exceeds velocity (range_rate.py)")
        outputs[Dynamic.Mission.DISTANCE_RATE] = (velocity_2 - climb_rate_2)**0.5

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(
            Dynamic.Mission.DISTANCE_RATE,
            [Dynamic.Atmosphere.ALTITUDEUDE_RATE, Dynamic.Atmosphere.VELOCITY],
            rows=arange,
            cols=arange,
        )

    def compute_partials(self, inputs, J):
        climb_rate = inputs[Dynamic.Atmosphere.ALTITUDEUDE_RATE]
        velocity = inputs[Dynamic.Atmosphere.VELOCITY]

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Atmosphere.ALTITUDEUDE_RATE] = (
            -climb_rate / (velocity**2 - climb_rate**2) ** 0.5
        )
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Atmosphere.VELOCITY] = (
            velocity / (velocity**2 - climb_rate**2) ** 0.5
        )
