import numpy as np
import openmdao.api as om


class ForceComponentResolver(om.ExplicitComponent):
    """
    This class will resolve forces (thrust, drag, lift, etc.) into their 
    respective x,y,z components for the 6 DOF equations of motion. 
    
    This class assumes that the total force is given and needs to be resolved 
    into the separate components.

    """

    def setup(self):

        # inputs

        self.add_input(
            'u',
            val=0.0,
            units='m/s',
            desc="axial velocity"
        )

        self.add_input(
            'v',
            val=0.0,
            units='m/s',
            desc="lateral velocity"
        )

        self.add_input(
            'w',
            val=0.0,
            units='m/s',
            desc="vertical velocity"
        )

        self.add_input(
            'drag',
            val=0.0,
            units='N',
            desc="Drag vector (unresolved)"
        )

        self.add_input(
            'thrust',
            val=0.0,
            units='N',
            desc="Thrust vector (unresolved)"
        )

        self.add_input(
            'lift',
            val=0.0,
            units='N',
            desc="Lift vector (unresolved)"
        )

        self.add_input(
            'side',
            val=0.0,
            units='N',
            desc="Side vector (unresolved)"
        )

        # outputs

        self.add_output(
            'Fx',
            val=0.0,
            units='N',
            desc="x-comp of final force"
        )

        self.add_output(
            'Fy',
            val=0.0,
            units='N',
            desc="y-comp of final force"
        )

        self.add_output(
            'Fz',
            val=0.0,
            units='N',
            desc="z-comp of final force"
        )

    def compute(self, inputs, outputs):

        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        L = inputs['lift']
        S = inputs['side'] # side force -- assume 0 for now

        # true air speed

        V = np.sqrt(u**2 + v**2 + w**2)

        # angle of attack

        alpha = np.arctan(w / u)

        # side slip angle

        # divide by zero checks
        if (V == 0) and ((u != 0 or w != 0)) :
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        elif (V == 0) and ((u == 0) and (w == 0)):
            u = 1.0e-4
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        else:
            beta = np.arcsin(v / V)
        

        # some trig needed

        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)
        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)

        outputs['Fx'] = -(cos_a * cos_b * D - cos_a * sin_b * S - sin_a * L)
        outputs['Fy'] = -(sin_b * D + cos_b * S)
        outputs['Fz'] = -(sin_a * cos_b * D + sin_a * sin_b * S + cos_a * L)







