import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic, Mission


class MachNumber(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            Dynamic.Atmosphere.VELOCITY,
            val=np.ones(nn),
            desc='true airspeed',
            units='m/s',
        )
        self.add_input(
            Dynamic.Atmosphere.SPEED_OF_SOUND,
            val=np.ones(nn),
            desc='speed of sound',
            units='m/s',
        )
        self.add_output(
            Dynamic.Atmosphere.MACH,
            val=np.ones(nn),
            desc='current Mach number',
            units='unitless',
        )

    def compute(self, inputs, outputs):
        sos = inputs[Dynamic.Atmosphere.SPEED_OF_SOUND]
        velocity = inputs[Dynamic.Atmosphere.VELOCITY]

        outputs[Dynamic.Atmosphere.MACH] = velocity / sos

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(
            Dynamic.Atmosphere.MACH,
            [Dynamic.Atmosphere.SPEED_OF_SOUND, Dynamic.Atmosphere.VELOCITY],
            rows=arange,
            cols=arange,
        )

    def compute_partials(self, inputs, J):
        sos = inputs[Dynamic.Atmosphere.SPEED_OF_SOUND]
        velocity = inputs[Dynamic.Atmosphere.VELOCITY]

        J[Dynamic.Atmosphere.MACH, Dynamic.Atmosphere.VELOCITY] = 1 / sos
        J[Dynamic.Atmosphere.MACH, Dynamic.Atmosphere.SPEED_OF_SOUND] = (
            -velocity / sos**2
        )
