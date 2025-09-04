import numpy as np
import openmdao.api as om
from aviary.variable_info.variables import Dynamic
from aviary.utils.functions import add_aviary_input


class ForceComponentResolver(om.ExplicitComponent):
    """
    This class will resolve forces (thrust, drag, lift, etc.) into their 
    respective x,y,z components for the 6 DOF equations of motion. 
    
    This class assumes that the total force is given and needs to be resolved 
    into the separate components.

    Assumptions:
        - Thrust is entirely in -z direction (T = (0,0,-T_z)^T) w.r.t. body CS
        - Assuming F_i is in body CS, and D, S, and L are in wind CS. Wind -> body rotation matrix
          was applied for coordinate transformations
        - Thrust is initially in NED CS. So, two rotations (NED -> wind and wind -> body) are applied

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

        self.add_input(
            'heading_angle',
            val=np.zeros(nn),
            units='rad',
            desc='Heading angle in body'
        )

        add_aviary_input(self,
                         Dynamic.Mission.FLIGHT_PATH_ANGLE,
                         units='rad',
                         desc="Flight path angle in body")
        
        self.add_input(
            'heading_angle_NED',
            val=np.zeros(nn),
            units='rad',
            desc="Thrust heading angle in NED"
        )

        self.add_input(
            'fpa_NED',
            val=np.zeros(nn),
            units='rad',
            desc="Thrust flight path angle in NED"
        )

        # self.add_input(
        #     'true_air_speed',
        #     val=np.zeros(nn),
        #     units='m/s',
        #     desc="True air speed"
        # ) # This is an aviary variable

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
        self.declare_partials(of='Fx', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='heading_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt=Dynamic.Mission.FLIGHT_PATH_ANGLE, rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='heading_angle_NED', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='fpa_NED', rows=ar, cols=ar)

        self.declare_partials(of='Fy', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='side', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='heading_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt=Dynamic.Mission.FLIGHT_PATH_ANGLE, rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='heading_angle_NED', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='fpa_NED', rows=ar, cols=ar)

        self.declare_partials(of='Fz', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='side', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='heading_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt=Dynamic.Mission.FLIGHT_PATH_ANGLE, rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='heading_angle_NED', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='fpa_NED', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        L = inputs['lift']
        S = inputs['side'] # side force -- assume 0 for now
        chi = inputs['heading_angle']
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        chi_T = inputs['heading_angle_NED']
        gamma_T = inputs['fpa_NED']

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
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)
        cos_c = np.cos(chi)
        sin_c = np.sin(chi)

        # Thrust direction in NED -- as \hat{t}_n

        t_hat_n = [
            np.cos(gamma_T) * np.cos(chi_T),
            np.cos(gamma_T) * np.sin(chi_T),
            -np.sin(gamma_T)
        ]

        t_hat_n = np.array(t_hat_n)
        t_hat_n = t_hat_n.reshape((3, 1))

        # C_{b-<n}

        Cbn = [
            -sin_b * sin_c - cos_b * cos_c * np.cos(alpha - gamma),
             sin_b * cos_c - cos_b * sin_c * np.cos(alpha - gamma),
             cos_b * np.sin(alpha - gamma),
            cos_b * sin_c - sin_b * cos_c * np.cos(alpha - gamma),
             -cos_b * cos_c - sin_b * sin_c * np.cos(alpha - gamma),
             sin_b * np.sin(alpha - gamma),
            cos_c * np.sin(alpha - gamma),
             sin_c * np.sin(alpha - gamma),
             np.cos(alpha - gamma)
        ]

        Cbn = np.array(Cbn)
        Cbn = Cbn.reshape((3, 3))

        

        print("Cbn shape: ", np.shape(Cbn))
        print("t_hat_n shape: ", np.shape(t_hat_n))
        print("Cbn = ", Cbn)
        print("t_hat_n = ", t_hat_n)
        

        # Thrust direction in body
        t_hat_b = Cbn @ t_hat_n

        # Thrust force in body
        Fb_thrust = T * t_hat_b

        # Thrust in (x,y,z)
        F_T_x = Fb_thrust[0]
        F_T_y = Fb_thrust[1]
        F_T_z = Fb_thrust[2]
        

        outputs['Fx'] = -(cos_a * cos_b * D - cos_a * sin_b * S - sin_a * L) + F_T_x
        outputs['Fy'] = -(sin_b * D + cos_b * S) + F_T_y
        outputs['Fz'] = -(sin_a * cos_b * D + sin_a * sin_b * S + cos_a * L) + F_T_z

    def compute_partials(self, inputs, J):
        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        L = inputs['lift']
        S = inputs['side'] # side force -- assume 0 for now
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        chi = inputs['heading_angle']
        chi_T = inputs['heading_angle_NED']
        gamma_T = inputs['fpa_NED']

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
        
        t_hat_n = [
            np.cos(gamma_T) * np.cos(chi_T),
            np.cos(gamma_T) * np.sin(chi_T),
            -np.sin(gamma_T)
        ]

        t_hat_n = np.array(t_hat_n)
        t_hat_n = t_hat_n.reshape((3, 1))

        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta),  np.sin(beta)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        cos_c, sin_c = np.cos(chi),   np.sin(chi)

        # C_{b-<n}

        Cbn = [
            -sin_b * sin_c - cos_b * cos_c * np.cos(alpha - gamma),
             sin_b * cos_c - cos_b * sin_c * np.cos(alpha - gamma),
             cos_b * np.sin(alpha - gamma),
            cos_b * sin_c - sin_b * cos_c * np.cos(alpha - gamma),
             -cos_b * cos_c - sin_b * sin_c * np.cos(alpha - gamma),
             sin_b * np.sin(alpha - gamma),
            cos_c * np.sin(alpha - gamma),
             sin_c * np.sin(alpha - gamma),
             np.cos(alpha - gamma)
        ]

        Cbn = np.array(Cbn)
        Cbn = Cbn.reshape((3, 3))

        # Derivatives of t_n
        dt_n_dchi = [
            -np.cos(gamma_T) * np.sin(chi_T),
            np.cos(gamma_T) * np.cos(chi_T),
            0.0
        ]

        dt_n_dchi = np.array(dt_n_dchi, dtype="object")
        dt_n_dchi = dt_n_dchi.reshape((3, 1))

        dt_n_dgamma = [
            -np.sin(gamma_T) * np.cos(chi_T),
            -np.sin(gamma_T) * np.sin(chi_T),
            -np.cos(gamma_T)
        ]

        dt_n_dgamma = np.array(dt_n_dgamma, dtype="object")
        dt_n_dgamma = dt_n_dgamma.reshape((3, 1))

        # Rotation to body + derivatives
        t_b = Cbn @ t_hat_n
        dt_b_dchi = Cbn @ dt_n_dchi
        dt_b_dgamma = Cbn @ dt_n_dgamma

        # note: d/dx arctan(a/x) = -a / (x^2 + a^2)
        # note: d/dx arctan(a / sqrt(b^2 + x^2)) = - ax / ((b^2 + x^2 + a^2) * sqrt(b^2 + x^2))
        # note: d/dx arctan(x / sqrt(a^2 + b^2)) = sqrt(a^2 + b^2) / (a^2 + b^2 + x^2)
        # note: d/dx arctan(x/a) = a / (a^2 + x^2)

        J['Fx', 'u'] = np.cos(alpha) * np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + \
                         np.cos(beta) * np.sin(alpha) * (-w / (w**2 + u**2)) * D + \
                         (np.cos(alpha) * np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.sin(beta) * np.sin(alpha) * (-w / (w**2 + u**2)) * S) + \
                         (np.cos(alpha) * (-w / (w**2 + u**2)) * L) + T * ((-np.cos(gamma_T) * np.cos(chi_T) * cos_b * sin_c * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) + 
                                                                           np.cos(gamma_T) * np.cos(chi_T) * sin_b * cos_c * np.cos(alpha - gamma) * 
                                                                           ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) + np.cos(gamma_T) * np.cos(chi_T) * cos_b * cos_c * 
                                                                           np.sin(alpha - gamma) * (-w / (w**2 + u**2))) + (
                                                                               np.cos(gamma_T) * np.sin(chi_T) * cos_b * cos_c * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) + 
                                                                                np.cos(gamma_T) * np.sin(chi_T) * sin_b * sin_c * np.cos(alpha - gamma) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) + 
                                                                           np.cos(gamma_T) * np.sin(chi_T) * cos_b * sin_c * np.sin(alpha -gamma) * (-w / (w**2 + u**2))) + 
                                                                           (np.sin(gamma_T) * sin_b * np.sin(alpha - gamma) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) - 
                                                                            np.sin(gamma_T) * cos_b * np.cos(alpha - gamma) * (-w / (w**2 + u**2))))
        J['Fx', 'v'] = np.cos(alpha) * np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * D + np.cos(alpha) * np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * S + T * (
            (-np.cos(gamma_T) * np.cos(chi_T) * sin_b * sin_c * (np.sqrt(w**2 + u**2) / V**2) - np.cos(gamma_T) * np.cos(chi_T) * cos_b * cos_c * np.cos(alpha - gamma) * (np.sqrt(w**2 + u**2) / V**2)) + 
             (np.cos(gamma_T) * np.sin(chi_T) * sin_b * cos_c * (np.sqrt(w**2 + u**2) / V**2) - np.cos(gamma_T) * np.sin(chi_T) * cos_b * sin_c * np.cos(alpha - gamma) * (np.sqrt(w**2 + u**2) / V**2)) -
             (np.sin(gamma_T) * cos_b * np.sin(alpha - gamma) * (np.sqrt(w**2 + u**2) / V**2))
        )
        J['Fx', 'w'] = np.cos(alpha) * np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(alpha) * np.cos(beta) * (u / (w**2 + u**2)) * D + \
                       np.cos(alpha) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.sin(alpha) * np.sin(beta) * (u / (w**2 + u**2)) * S + \
                       np.cos(alpha) * (u / (w**2 + u**2)) * L + T * (
                           (np.cos(gamma_T) * np.cos(chi_T) * cos_c * np.cos(alpha - gamma) * (u / (w**2 + u**2))) + 
                           (np.cos(gamma_T) * np.sin(chi_T) * sin_c * np.cos(alpha - gamma) * (u / (w**2 + u**2))) + 
                           (np.sin(gamma_T) * np.sin(alpha - gamma) * (u / (w**2 + u**2)))
                       )
        J['Fx', 'drag'] = -np.cos(alpha) * np.cos(beta)
        J['Fx', 'lift'] = np.sin(alpha)
        J['Fx', 'side'] = np.cos(alpha) * np.sin(beta)
        J['Fx', 'thrust'] = t_b[0]
        J['Fx', Dynamic.Mission.FLIGHT_PATH_ANGLE] = T * ((-cos_b * cos_c * np.sin(alpha - gamma)) * t_hat_n[0] + (-cos_b * sin_c * np.sin(alpha - gamma)) * t_hat_n[1] + 
                                                          (-cos_b * np.cos(alpha - gamma)) * t_hat_n[2])
        J['Fx', 'heading_angle'] = T * ((-sin_b * cos_c + cos_b * sin_c * np.cos(alpha - gamma)) * t_hat_n[0] + 
                                        (-sin_b * sin_c - cos_b * cos_c * np.cos(alpha - gamma)) * t_hat_n[1])
        J['Fx', 'heading_angle_NED'] = T * dt_b_dchi[0]
        J['Fx', 'fpa_NED'] = T * dt_b_dgamma[0]

        J['Fy', 'u'] = -np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - \
                        np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * T * np.sin(alpha - gamma) - \
                        np.cos(alpha - gamma) * (-w / (w**2 + u**2)) * T * np.sin(beta)
        J['Fy', 'v'] = -np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * D + np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * S - \
                        np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * T * np.sin(alpha - gamma)
        J['Fy', 'w'] = -np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - \
                        np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * T * np.sin(alpha - gamma) - \
                        np.cos(alpha - gamma) * (u / (w**2 + u**2)) * T * np.sin(beta)
        J['Fy', 'drag'] = -np.sin(beta)
        J['Fy', 'side'] = -np.cos(beta)
        J['Fy', 'thrust'] = t_b[1]
        J['Fy', Dynamic.Mission.FLIGHT_PATH_ANGLE] = T * ((-sin_b * cos_c * np.sin(alpha - gamma)) * t_hat_n[0] + (-sin_b * sin_c * np.sin(alpha - gamma)) * t_hat_n[1] + 
                                                          (-sin_b * np.cos(alpha - gamma)) * t_hat_n[2])
        J['Fy', 'heading_angle'] = T * ((cos_b * cos_c + sin_b * sin_c * np.cos(alpha - gamma)) * t_hat_n[0] + (cos_b * sin_c - sin_b * cos_c * np.cos(alpha - gamma)) * t_hat_n[1])
        J['Fy', 'heading_angle_NED'] = T * dt_b_dchi[1]
        J['Fy', 'fpa_NED'] = T * dt_b_dgamma[1]

        J['Fz', 'u'] = np.sin(alpha) * np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D - np.cos(alpha) * np.cos(beta) * (-w / (w**2 + u**2)) * D - \
                       np.sin(alpha) * np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.cos(alpha) * np.sin(beta) * (-w / (w**2 + u**2)) * S + \
                       np.sin(alpha) * (-w / (w**2 + u**2)) * L + np.sin(alpha - gamma) * (-w / (w**2 + u**2)) * T
        J['Fz', 'v'] = np.sin(alpha) * np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * D - np.sin(alpha) * np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * S 
        J['Fz', 'w'] = np.sin(alpha) * np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D - np.cos(alpha) * np.cos(beta) * (u / (w**2 + u**2)) * D - \
                       np.sin(alpha) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.cos(alpha) * np.sin(beta) * (u / (w**2 + u**2)) * S + \
                       np.sin(alpha) * (u / (w**2 + u**2)) * L + np.sin(alpha - gamma) * (u / (w**2 + u**2)) * T
        J['Fz', 'drag'] = -np.sin(alpha) * np.cos(beta) 
        J['Fz', 'lift'] = -np.cos(alpha)
        J['Fz', 'side'] = -np.sin(alpha) * np.sin(beta)
        J['Fz', 'thrust'] = t_b[2]
        J['Fz', Dynamic.Mission.FLIGHT_PATH_ANGLE] = T * ((-cos_c * np.cos(alpha - gamma)) * t_hat_n[0] + (-sin_c * np.cos(alpha - gamma)) * t_hat_n[1] + 
                                                          np.sin(alpha - gamma) * t_hat_n[2])
        J['Fz', 'heading_angle'] = T * ((-sin_c * np.sin(alpha - gamma)) * t_hat_n[0] + (cos_c * np.sin(alpha - gamma)) * t_hat_n[1])
        J['Fz', 'heading_angle_NED'] = T * dt_b_dchi[2]
        J['Fz', 'fpa_NED'] = T * dt_b_dgamma[2]

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
    des_vars.add_output('heading_angle', units='rad')
    des_vars.add_output(Dynamic.Mission.FLIGHT_PATH_ANGLE, units='rad')
    des_vars.add_output('heading_angle_NED', units='rad')
    des_vars.add_output('fpa_NED', units='rad')

    p.model.add_subsystem('ForceComponentResolver', ForceComponentResolver(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.run_model()

    p.check_partials(compact_print=True, show_only_incorrect=True, method='cs')







