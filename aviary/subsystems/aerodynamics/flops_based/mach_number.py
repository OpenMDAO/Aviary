import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic, Mission


class MachNumber(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(Dynamic.Mission.VELOCITY, val=np.ones(nn),
                       desc='true airspeed', units='m/s')
        self.add_input(Dynamic.Mission.SPEED_OF_SOUND, val=np.ones(
            nn), desc='speed of sound', units='m/s')
        self.add_output(Dynamic.Mission.MACH, val=np.ones(
            nn), desc='current Mach number', units='unitless')

    def compute(self, inputs, outputs):
        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]
        velocity = inputs[Dynamic.Mission.VELOCITY]

        outputs[Dynamic.Mission.MACH] = velocity/sos

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(
            Dynamic.Mission.MACH, [Dynamic.Mission.SPEED_OF_SOUND, Dynamic.Mission.VELOCITY], rows=arange, cols=arange)

    def compute_partials(self, inputs, J):
        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]
        velocity = inputs[Dynamic.Mission.VELOCITY]

        J[Dynamic.Mission.MACH, Dynamic.Mission.VELOCITY] = 1/sos
        J[Dynamic.Mission.MACH, Dynamic.Mission.SPEED_OF_SOUND] = -velocity/sos**2
