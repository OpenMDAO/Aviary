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
        S = inputs['side'] # side force 

        # true air speed

        V = np.sqrt(u**2 + v**2 + w**2)

        # ------------------------------------------------------------------------------
        # ----------------------------- Drag calculation -------------------------------
        # ------------------------------------------------------------------------------

        alpha = np.arctan(w / u)

        beta = np.arctan(v / np.sqrt(u**2 + w**2))

        # some trig needed

        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)
        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)

        Fx_NoThrust = -(cos_a * cos_b * D - cos_a * sin_b * S - sin_a * L)
        Fy_NoThrust = -(sin_b * D + cos_b * S)
        Fz_NoThrust = -(sin_a * cos_b * D + sin_a * sin_b * S + cos_a * L)







