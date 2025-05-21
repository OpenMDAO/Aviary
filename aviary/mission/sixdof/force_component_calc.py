import numpy as np
import openmdao.api as om


class ForceComponentResolver(om.ExplicitComponent):
    """
    This class will resolve forces (thrust, drag, lift, etc.) into their 
    respective x,y,z components for the 6 DOF equations of motion. 
    
    This class assumes that the total force is given and needs to be resolved 
    into the separate components.

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs

        self.add_input(
            'u',
            val=np.zeros(nn),
            units='m/s',
            desc="axial velocity"
        )

        self.add_input(
            'v',
            val=np.zeros(nn),
            units='m/s',
            desc="lateral velocity"
        )

        self.add_input(
            'w',
            val=np.zeros(nn),
            units='m/s',
            desc="vertical velocity"
        )

        self.add_input(
            'drag',
            val=np.zeros(nn),
            units='N',
            desc="Drag vector (unresolved)"
        )

        self.add_input(
            'thrust',
            val=np.zeros(nn),
            units='N',
            desc="Thrust vector (unresolved)"
        )

        self.add_input(
            'lift',
            val=np.zeros(nn),
            units='N',
            desc="Lift vector (unresolved)"
        )

        self.add_input(
            'side',
            val=np.zeros(nn),
            units='N',
            desc="Side vector (unresolved)"
        )

        # outputs

        self.add_output(
            'Fx',
            val=np.zeros(nn),
            units='N',
            desc="x-comp of final force"
        )

        self.add_output(
            'Fy',
            val=np.zeros(nn),
            units='N',
            desc="y-comp of final force"
        )

        self.add_output(
            'Fz',
            val=np.zeros(nn),
            units='N',
            desc="z-comp of final force"
        )

        ar = np.arange(nn)

        self.declare_partials(of='Fx', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='side', rows=ar, cols=ar)

        self.declare_partials(of='Fy', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='side', rows=ar, cols=ar)

        self.declare_partials(of='Fz', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='side', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        L = inputs['lift']
        S = inputs['side'] # side force -- assume 0 for now

        nn = self.options['num_nodes']

        # true air speed

        V = np.sqrt(u**2 + v**2 + w**2)

        # angle of attack

        # divide by zero checks
        if np.any(u == 0):
            u[u == 0] = 1e-4
            alpha = np.arctan(w / u)
        else:
            alpha = np.arctan(w / u)

        # side slip angle

        # divide by zero checks
        if ((np.any(u != 0) or np.any(w != 0))) :
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        else:
            u[u == 0] = 1.0e-4
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        
        
        

        # some trig needed

        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)
        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)

        outputs['Fx'] = -(cos_a * cos_b * D - cos_a * sin_b * S - sin_a * L)
        outputs['Fy'] = -(sin_b * D + cos_b * S)
        outputs['Fz'] = -(sin_a * cos_b * D + sin_a * sin_b * S + cos_a * L)
    
    def compute_partials(self, inputs, J):

        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        L = inputs['lift']
        S = inputs['side'] # side force -- assume 0 for now

        V = np.sqrt(u**2 + v**2 + w**2)

        # divide by zero checks
        if u == 0:
            u = 1e-4
            alpha = np.arctan(w / u)
        else:
            alpha = np.arctan(w / u)

        # side slip angle

        # divide by zero checks
        if ((u != 0 or w != 0)) :
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        else:
            u = 1.0e-4
            beta = np.arctan(v / np.sqrt(u**2 + w**2))

        # note: d/dx arctan(a/x) = -a / (x^2 + a^2)
        # note: d/dx arctan(a / sqrt(b^2 + x^2)) = - ax / ((b^2 + x^2 + a^2) * sqrt(b^2 + x^2))
        # note: d/dx arctan(x / sqrt(a^2 + b^2)) = sqrt(a^2 + b^2) / (a^2 + b^2 + x^2)
        # note: d/dx arctan(x/a) = a / (a^2 + x^2)

        J['Fx', 'u'] = np.cos(alpha) * np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + \
                         np.cos(beta) * np.sin(alpha) * (-w / (w**2 + u**2)) * D + \
                         (np.cos(alpha) * np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.sin(beta) * np.sin(alpha) * (-w / (w**2 + u**2)) * S) + \
                         (np.cos(alpha) * (-w / (w**2 + u**2)) * L)
        J['Fx', 'v'] = np.cos(alpha) * np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * D + np.cos(alpha) * np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * S
        J['Fx', 'w'] = np.cos(alpha) * np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(alpha) * np.cos(beta) * (u / (w**2 + u**2)) * D + \
                       np.cos(alpha) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.sin(alpha) * np.sin(beta) * (u / (w**2 + u**2)) * S + \
                       np.cos(alpha) * (u / (w**2 + u**2)) * L
        J['Fx', 'drag'] = -np.cos(alpha) * np.cos(beta)
        J['Fx', 'lift'] = np.sin(alpha)
        J['Fx', 'side'] = np.cos(alpha) * np.sin(beta)

        J['Fy', 'u'] = -np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S
        J['Fy', 'v'] = -np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * D + np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * S
        J['Fy', 'w'] = -np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S
        J['Fy', 'drag'] = -np.sin(beta)
        J['Fy', 'side'] = -np.cos(beta)

        J['Fz', 'u'] = np.sin(alpha) * np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D - np.cos(alpha) * np.cos(beta) * (-w / (w**2 + u**2)) * D - \
                       np.sin(alpha) * np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.cos(alpha) * np.sin(beta) * (-w / (w**2 + u**2)) * S + \
                       np.sin(alpha) * (-w / (w**2 + u**2)) * L
        J['Fz', 'v'] = np.sin(alpha) * np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * D - np.sin(alpha) * np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * S 
        J['Fz', 'w'] = np.sin(alpha) * np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D - np.cos(alpha) * np.cos(beta) * (u / (w**2 + u**2)) * D - \
                       np.sin(alpha) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.cos(alpha) * np.sin(beta) * (u / (w**2 + u**2)) * S + \
                       np.sin(alpha) * (u / (w**2 + u**2)) * L
        J['Fz', 'drag'] = -np.sin(alpha) * np.cos(beta) 
        J['Fz', 'lift'] = -np.cos(alpha)
        J['Fz', 'side'] = -np.sin(alpha) * np.sin(beta)

if __name__ == "__main__":
    p = om.Problem()
    p.model = om.Group()
    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])

    des_vars.add_output('u', 0.5, units='m/s')
    des_vars.add_output('v', 0.6, units='m/s')
    des_vars.add_output('w', 0.7, units='m/s')
    des_vars.add_output('drag', 50, units='N')
    des_vars.add_output('thrust', 50, units='N')
    des_vars.add_output('lift', 60, units='N')
    des_vars.add_output('side', 70, units='N')

    p.model.add_subsystem('ForceComponentResolver', ForceComponentResolver(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.run_model()

    p.check_partials(compact_print=True, show_only_incorrect=True, method='cs')







