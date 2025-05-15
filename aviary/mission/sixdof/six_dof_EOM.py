import numpy as np
import openmdao.api as om

class SixDOF_EOM(om.ExplicitComponent):
    """
    Six DOF EOM component, with particular emphasis for rotorcraft. 
    ASSUMPTIONS:
        - Assume Flat Earth model (particularly for rotorcraft)
        - Earth is the internal f.o.r.
        - Gravity is constant and normal to the tangent plane (Earth's surface) -- g = (0 0 G)^T
        - (aircraft) mass is constant
        - aircraft is a rigid body
        - Symmetry in the xz plane (for moment of inertia matrix -- J_xy = J_yx = J_zy = J_yz = 0)
    
    Reference: https://youtu.be/hr_PqdkG6XY?si=_hFAZZMnk58eV9GJ

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            'mass',
            val=np.zeros(nn),
            units='kg',
            desc="mass -- assume constant"
        )

        self.add_input(
            'axial_vel',
            val=np.zeros(nn),
            units='m/s', # meters per second
            desc="axial velocity of CM wrt inertial CS resolved in aircraft body fixed CS"
        )

        self.add_input(
            'lat_vel',
            val=np.zeros(nn),
            units='m/s',
            desc="lateral velocity of CM wrt inertial CS resolved in aircraft body fixed CS"
        )

        self.add_input(
            'vert_vel',
            val=np.zeros(nn),
            units='m/s',
            desc="vertical velocity of CM wrt inertial CS resolved in aircraft body fixed CS"
        )

        self.add_input(
            'roll_ang_vel',
            val=np.zeros(nn),
            units='rad/s', # radians per second
            desc="roll angular velocity of body fixed CS wrt intertial CS"
        )

        self.add_input(
            'pitch_ang_vel',
            val=np.zeros(nn),
            units='rad/s',
            desc="pitch angular velocity of body fixed CS wrt intertial CS"
        )

        self.add_input(
            'yaw_ang_vel',
            val=np.zeros(nn),
            units='rad/s',
            desc="yaw angular velocity of body fixed CS wrt intertial CS"
        )

        self.add_input(
            'roll',
            val=np.zeros(nn),
            units='rad', # radians
            desc="roll angle"
        )

        self.add_input(
            'pitch',
            val=np.zeros(nn),
            units='rad',
            desc="pitch angle"
        )

        self.add_input(
            'yaw',
            val=np.zeros(nn),
            units='rad',
            desc="yaw angle"
        )

        # self.add_input(
        #     'x',
        #     val=0.0,
        #     units='m',
        #     desc="x-axis position of aircraft resolved in North-East-Down (NED) CS"
        # )

        # self.add_input(
        #     'y',
        #     val=0.0,
        #     units='m',
        #     desc="y-axis position of aircraft resolved in NED CS"
        # )

        # self.add_input(
        #     'z',
        #     val=0.0,
        #     units='m',
        #     desc="z-axis position of aircraft resolved in NED CS"
        # )

        # self.add_input(
        #     'time',
        #     val=0.0,
        #     desc="scalar time in seconds"
        # )

        self.add_input(
            'g',
            val=9.81,
            units='m/s**2',
            desc="acceleration due to gravity"
        )

        self.add_input(
            'Fx_ext',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the x direciton"
        )

        self.add_input(
            'Fy_ext',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the y direction"
        )

        self.add_input(
            'Fz_ext',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the z direction"
        )

        self.add_input(
            'lx_ext',
            val=np.zeros(nn),
            units='kg*m**2/s**2', # kg times m^2 / s^2
            desc="external moments in the x direction"
        )

        self.add_input(
            'ly_ext',
            val=np.zeros(nn),
            units='kg*m**2/s**2',
            desc="external moments in the y direction"
        )

        self.add_input(
            'lz_ext',
            val=np.zeros(nn),
            units='kg*m**2/s**2',
            desc="external moments in the z direction"
        )

        # Below are the necessary components for the moment of inertia matrix (J)
        # Only xx, yy, zz, and xz are needed (xy and yz are 0 with assumptions)
        # For now, these are separated.
        # TODO: Rewrite J and EOM in matrix form

        self.add_input(
            'J_xz',
            val=np.zeros(1),
            units='kg*m**2',
            desc="x-z (top right and bottom left corner of 3x3 matrix, assuming symmetry) " \
            "component"
        )

        self.add_input(
            'J_xx',
            val=np.zeros(1),
            units='kg*m**2',
            desc="first diag component"
        )

        self.add_input(
            'J_yy',
            val=np.zeros(1),
            units='kg*m**2',
            desc="second diag component"
        )

        self.add_input(
            'J_zz',
            val=np.zeros(1),
            units='kg*m**2',
            desc="third diag component"
        )

        # Outputs

        self.add_output(
            'dx_accel',
            val=np.zeros(nn),
            units='m/s**2', # meters per seconds squared
            desc="x-axis (roll-axis) velocity equation, " \
            "state: axial_vel"
        )

        self.add_output(
            'dy_accel',
            val=np.zeros(nn),
            units='m/s**2',
            desc="y-axis (pitch axis) velocity equation, " \
            "state: lat_vel"
        )

        self.add_output(
            'dz_accel',
            val=np.zeros(nn),
            units='m/s**2',
            desc="z-axis (yaw axis) velocity equation, " \
            "state: vert_vel"
        )

        self.add_output(
            'roll_accel',
            val=np.zeros(nn),
            units='rad/s**2', # radians per second squared
            desc="roll equation, " \
            "state: roll_ang_vel"
        )

        self.add_output(
            'pitch_accel',
            val=np.zeros(nn),
            units='rad/s**2',
            desc="pitch equation, " \
            "state: pitch_ang_vel"
        )

        self.add_output(
            'yaw_accel',
            val=np.zeros(nn),
            units='rad/s**2',
            desc="yaw equation, " \
            "state: yaw_ang_vel"
        )

        self.add_output(
            'roll_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular roll rate"
        )

        self.add_output(
            'pitch_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular pitch rate"
        )

        self.add_output(
            'yaw_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular yaw rate"
        )

        self.add_output(
            'dx_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="x-position derivative of aircraft COM wrt point in NED CS" 
        )

        self.add_output(
            'dy_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="y-position derivative of aircraft COM wrt point in NED CS"
        )

        self.add_output(
            'dz_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="z-position derivative of aircraft COM wrt point in NED CS"
        )
    
    def compute(self, inputs, outputs):
        """
        Compute function for EOM. 
        TODO: Same as above, potentially rewrite equations for \
              matrix form, and add potential assymetry to moment \
              of inertia matrix.

        """

        # inputs

        mass = inputs['mass']
        axial_vel = inputs['axial_vel'] # u
        lat_vel = inputs['lat_vel'] # v
        vert_vel = inputs['vert_vel'] # w
        roll_ang_vel = inputs['roll_ang_vel'] # p
        pitch_ang_vel = inputs['pitch_ang_vel'] # q
        yaw_ang_vel = inputs['yaw_ang_vel'] # r
        roll = inputs['roll'] # phi
        pitch = inputs['pitch'] # theta
        yaw = inputs['yaw'] # psi
        # x = inputs['x'] # p1
        # y = inputs['y'] # p2
        # z = inputs['z'] # p3
        # time = inputs['time']
        g = inputs['g']
        Fx_ext = inputs['Fx_ext']
        Fy_ext = inputs['Fy_ext']
        Fz_ext = inputs['Fz_ext']
        lx_ext = inputs['lx_ext'] # l
        ly_ext = inputs['ly_ext'] # m
        lz_ext = inputs['lz_ext'] # n
        J_xz = inputs['J_xz']
        J_xx = inputs['J_xx']
        J_yy = inputs['J_yy']
        J_zz = inputs['J_zz']

        # Resolve gravity in body coordinate system -- denoted with subscript 'b'
        gx_b = -np.sin(pitch) * g
        gy_b = np.sin(roll) * np.cos(pitch) * g
        gz_b = np.cos(roll) * np.cos(pitch) * g

        # TODO: could add external forces and moments here if needed

        # Denominator for roll and yaw rate equations
        Den = J_xx * J_zz - J_xz**2

        # roll-axis velocity equation

        dx_accel = 1 / mass * Fx_ext + gx_b - vert_vel * pitch_ang_vel + lat_vel * yaw_ang_vel

        # pitch-axis velocity equation

        dy_accel = 1 / mass * Fy_ext + gy_b - axial_vel * yaw_ang_vel + vert_vel * roll_ang_vel

        # yaw-axis velocity equation

        dz_accel = 1 / mass * Fz_ext + gz_b - lat_vel * roll_ang_vel + axial_vel * pitch_ang_vel

        # Roll equation

        roll_accel = (J_xz * (J_xx - J_yy + J_zz) * roll_ang_vel * pitch_ang_vel - 
                   (J_zz * (J_zz - J_yy) + J_xz**2) * pitch_ang_vel * yaw_ang_vel + 
                   J_zz * lx_ext + 
                   J_xz * lz_ext) / Den
        
        # Pitch equation

        pitch_accel = ((J_zz - J_xx) * roll_ang_vel * yaw_ang_vel - 
                    J_xz * (roll_ang_vel**2 - yaw_ang_vel**2) + ly_ext) / J_yy
        
        # Yaw equation

        yaw_accel = ((J_xx * (J_xx - J_yy) + J_xz**2) * roll_ang_vel * pitch_ang_vel + 
                  J_xz * (J_xx - J_yy + J_zz) * pitch_ang_vel * yaw_ang_vel + 
                  J_xz * lx_ext + 
                  J_xz * lz_ext) / Den
        
        # Kinematic equations
        
        roll_angle_rate_eq = roll_ang_vel + np.sin(roll) * np.tan(pitch) * pitch_ang_vel + \
                             np.cos(roll) * np.tan(pitch) * yaw_ang_vel
        
        pitch_angle_rate_eq = np.cos(roll) * pitch_ang_vel - np.sin(roll) * yaw_ang_vel

        yaw_angle_rate_eq = np.sin(roll) / np.cos(pitch) * pitch_ang_vel + \
                            np.cos(roll) / np.cos(pitch) * yaw_ang_vel

        # Position equations

        dx_dt = np.cos(pitch) * np.cos(yaw) * axial_vel + \
                (-np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw)) * lat_vel + \
                (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)) * vert_vel
        
        dy_dt = np.cos(pitch) * np.sin(yaw) * axial_vel + \
                (np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw)) * lat_vel + \
                (-np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) * vert_vel
        
        dz_dt = -np.sin(pitch) * axial_vel + \
                np.sin(roll) * np.cos(pitch) * lat_vel + \
                np.cos(roll) * np.cos(pitch) * vert_vel

        outputs['dx_accel'] = dx_accel
        outputs['dy_accel'] = dy_accel
        outputs['dz_accel'] = dz_accel
        outputs['roll_accel'] = roll_accel
        outputs['pitch_accel'] = pitch_accel
        outputs['yaw_accel'] = yaw_accel
        outputs['roll_angle_rate_eq'] = roll_angle_rate_eq
        outputs['pitch_angle_rate_eq'] = pitch_angle_rate_eq
        outputs['yaw_angle_rate_eq'] = yaw_angle_rate_eq
        outputs['dx_dt'] = dx_dt
        outputs['dy_dt'] = dy_dt
        outputs['dz_dt'] = dz_dt


if __name__ == "__main__":

    p = om.Problem()
    p.model = om.Group()
    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])

    des_vars.add_output('mass', 3.0, units='kg')
    des_vars.add_output('axial_vel', 0.1, units='m/s')
    des_vars.add_output('lat_vel', 0.7, units='m/s')
    des_vars.add_output('vert_vel', 0.12, units='m/s')
    des_vars.add_output('roll_ang_vel', 0.1, units='rad/s')
    des_vars.add_output('pitch_ang_vel', 0.9, units='rad/s')
    des_vars.add_output('yaw_ang_vel', 0.12, units='rad/s')
    des_vars.add_output('roll', 0.9, units='rad')
    des_vars.add_output('pitch', 0.19, units='rad')
    des_vars.add_output('yaw', 0.70, units='rad')
    des_vars.add_output('g', 9.81, units='m/s**2')
    des_vars.add_output('Fx_ext', 0.1, units='N')
    des_vars.add_output('Fy_ext', 0.9, units='N')
    des_vars.add_output('Fz_ext', 0.12, units='N')
    des_vars.add_output('lx_ext', 3.0, units='N*m')
    des_vars.add_output('ly_ext', 4.0, units='N*m')
    des_vars.add_output('lz_ext', 5.0, units='N*m')
    des_vars.add_output('J_xz', 9.0, units='kg*m**2')
    des_vars.add_output('J_xx', 50.0, units='kg*m**2')
    des_vars.add_output('J_yy', 51.0, units='kg*m**2')
    des_vars.add_output('J_zz', 52.0, units='kg*m**2')

    p.model.add_subsystem('SixDOF_EOM', SixDOF_EOM(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.run_model()

    dx_accel = p.get_val('dx_accel')
    dy_accel = p.get_val('dy_accel')
    dz_accel = p.get_val('dz_accel')
    roll_accel = p.get_val('roll_accel')
    pitch_accel = p.get_val('pitch_accel')
    yaw_accel = p.get_val('yaw_accel')
    roll_angle_rate_eq = p.get_val('roll_angle_rate_eq')
    pitch_angle_rate_eq = p.get_val('pitch_angle_rate_eq')
    yaw_angle_rate_eq = p.get_val('yaw_angle_rate_eq')
    dx_dt = p.get_val('dx_dt')
    dy_dt = p.get_val('dy_dt')
    dz_dt = p.get_val('dz_dt')

    print(f"Accelerations in x,y,z: {dx_accel}, {dy_accel}, {dz_accel}")
    print(f"Euler angle accels in roll, pitch, yaw: {roll_accel}, {pitch_accel}, {yaw_accel}")
    print(f"Euler angular rates: {roll_angle_rate_eq}, {pitch_angle_rate_eq}, {yaw_angle_rate_eq}")
    print(f"velocities: {dx_dt}, {dy_dt}, {dz_dt}")


