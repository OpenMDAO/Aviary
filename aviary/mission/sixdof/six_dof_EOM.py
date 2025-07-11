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
            'u',
            val=np.zeros(nn),
            units='m/s', # meters per second
            desc="axial velocity of CM wrt inertial CS resolved in aircraft body fixed CS"
        )

        self.add_input(
            'v',
            val=np.zeros(nn),
            units='m/s',
            desc="lateral velocity of CM wrt inertial CS resolved in aircraft body fixed CS"
        )

        self.add_input(
            'w',
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

        self.add_input(
            'x',
            val=np.zeros(nn),
            units='m',
            desc="x-axis position of aircraft resolved in North-East-Down (NED) CS"
        )

        self.add_input(
            'y',
            val=np.zeros(nn),
            units='m',
            desc="y-axis position of aircraft resolved in NED CS"
        )

        self.add_input(
            'z',
            val=np.zeros(nn),
            units='m',
            desc="z-axis position of aircraft resolved in NED CS"
        )

        self.add_input(
            'g',
            val=9.81,
            units='m/s**2',
            desc="acceleration due to gravity"
        )

        self.add_input(
            'Fx',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the x direciton"
        )

        self.add_input(
            'Fy',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the y direction"
        )

        self.add_input(
            'Fz',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the z direction"
        )

        self.add_input(
            'lx',
            val=np.zeros(nn),
            units='kg*m**2/s**2', # kg times m^2 / s^2
            desc="external moments in the x direction"
        )

        self.add_input(
            'ly',
            val=np.zeros(nn),
            units='kg*m**2/s**2',
            desc="external moments in the y direction"
        )

        self.add_input(
            'lz',
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
            "state: u",
            tags=['dymos.state_rate_source:u', 'dymos.state_units:m/s']
        )

        self.add_output(
            'dy_accel',
            val=np.zeros(nn),
            units='m/s**2',
            desc="y-axis (pitch axis) velocity equation, " \
            "state: v",
            tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s']
        )

        self.add_output(
            'dz_accel',
            val=np.zeros(nn),
            units='m/s**2',
            desc="z-axis (yaw axis) velocity equation, " \
            "state: w",
            tags=['dymos.state_rate_source:w', 'dymos.state_units:m/s']
        )

        self.add_output(
            'roll_accel',
            val=np.zeros(nn),
            units='rad/s**2', # radians per second squared
            desc="roll equation, " \
            "state: roll_ang_vel",
            tags=['dymos.state_rate_source:roll_ang_vel', 'dymos.state_units:rad/s']
        )

        self.add_output(
            'pitch_accel',
            val=np.zeros(nn),
            units='rad/s**2',
            desc="pitch equation, " \
            "state: pitch_ang_vel",
            tags=['dymos.state_rate_source:pitch_ang_vel', 'dymos.state_units:rad/s']
        )

        self.add_output(
            'yaw_accel',
            val=np.zeros(nn),
            units='rad/s**2',
            desc="yaw equation, " \
            "state: yaw_ang_vel",
            tags=['dymos.state_rate_source:yaw_ang_vel', 'dymos.state_units:rad/s']
        )

        self.add_output(
            'roll_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular roll rate",
            tags=['dymos.state_rate_source:roll', 'dymos.state_units:rad']
        )

        self.add_output(
            'pitch_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular pitch rate",
            tags=['dymos.state_rate_source:pitch', 'dymos.state_units:rad']
        )

        self.add_output(
            'yaw_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular yaw rate",
            tags=['dymos.state_rate_source:yaw', 'dymos.state_units:rad']
        )

        self.add_output(
            'dx_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="x-position derivative of aircraft COM wrt point in NED CS",
            tags=['dymos.state_rate_source:x', 'dymos.state_units:m'] 
        )

        self.add_output(
            'dy_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="y-position derivative of aircraft COM wrt point in NED CS",
            tags=['dymos.state_rate_source:y', 'dymos.state_units:m']
        )

        self.add_output(
            'dz_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="z-position derivative of aircraft COM wrt point in NED CS",
            tags=['dymos.state_rate_source:z', 'dymos.state_units:m']
        )

        ar = np.arange(nn)
        self.declare_partials(of='dx_accel', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='Fx', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='pitch_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='g')
        self.declare_partials(of='dx_accel', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='dy_accel', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='Fy', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='roll_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='g')
        self.declare_partials(of='dy_accel', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='dz_accel', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='Fz', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='roll_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='pitch_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='g')
        self.declare_partials(of='dz_accel', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='roll_accel', wrt='J_xz')
        self.declare_partials(of='roll_accel', wrt='J_xx')
        self.declare_partials(of='roll_accel', wrt='J_yy')
        self.declare_partials(of='roll_accel', wrt='J_zz')
        self.declare_partials(of='roll_accel', wrt='roll_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_accel', wrt='pitch_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_accel', wrt='lx', rows=ar, cols=ar)
        self.declare_partials(of='roll_accel', wrt='lz', rows=ar, cols=ar)

        self.declare_partials(of='pitch_accel', wrt='J_xz')
        self.declare_partials(of='pitch_accel', wrt='J_xx')
        self.declare_partials(of='pitch_accel', wrt='J_yy')
        self.declare_partials(of='pitch_accel', wrt='J_zz')
        self.declare_partials(of='pitch_accel', wrt='roll_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='pitch_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='pitch_accel', wrt='ly', rows=ar, cols=ar)

        self.declare_partials(of='yaw_accel', wrt='J_xz')
        self.declare_partials(of='yaw_accel', wrt='J_xx')
        self.declare_partials(of='yaw_accel', wrt='J_yy')
        self.declare_partials(of='yaw_accel', wrt='J_zz')
        self.declare_partials(of='yaw_accel', wrt='roll_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_accel', wrt='pitch_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_accel', wrt='lx', rows=ar, cols=ar)
        self.declare_partials(of='yaw_accel', wrt='lz', rows=ar, cols=ar)

        self.declare_partials(of='roll_angle_rate_eq', wrt='roll_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_angle_rate_eq', wrt='pitch_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_angle_rate_eq', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_angle_rate_eq', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='roll_angle_rate_eq', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='pitch_angle_rate_eq', wrt='pitch_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='pitch_angle_rate_eq', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='pitch_angle_rate_eq', wrt='roll', rows=ar, cols=ar)

        self.declare_partials(of='yaw_angle_rate_eq', wrt='pitch_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_angle_rate_eq', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_angle_rate_eq', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='yaw_angle_rate_eq', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='dx_dt', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='pitch', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='yaw', rows=ar, cols=ar)

        self.declare_partials(of='dy_dt', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='pitch', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='yaw', rows=ar, cols=ar)

        self.declare_partials(of='dz_dt', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dz_dt', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dz_dt', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dz_dt', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dz_dt', wrt='pitch', rows=ar, cols=ar)
        

    def compute(self, inputs, outputs):
        """
        Compute function for EOM. 
        TODO: Same as above, potentially rewrite equations for \
              matrix form, and add potential assymetry to moment \
              of inertia matrix.

        """

        # inputs

        mass = inputs['mass']
        u = inputs['u'] # u
        v = inputs['v'] # v
        w = inputs['w'] # w
        roll_ang_vel = inputs['roll_ang_vel'] # p
        pitch_ang_vel = inputs['pitch_ang_vel'] # q
        yaw_ang_vel = inputs['yaw_ang_vel'] # r
        roll = inputs['roll'] # phi
        pitch = inputs['pitch'] # theta
        yaw = inputs['yaw'] # psi
        x = inputs['x'] # p1
        y = inputs['y'] # p2
        z = inputs['z'] # p3
        # time = inputs['time']
        g = inputs['g']
        Fx = inputs['Fx']
        Fy = inputs['Fy']
        Fz = inputs['Fz']
        lx = inputs['lx'] # l
        ly = inputs['ly'] # m
        lz = inputs['lz'] # n
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

        dx_accel = 1 / mass * Fx + gx_b - w * pitch_ang_vel + v * yaw_ang_vel

        # pitch-axis velocity equation

        dy_accel = 1 / mass * Fy + gy_b - u * yaw_ang_vel + w * roll_ang_vel

        # yaw-axis velocity equation

        dz_accel = 1 / mass * Fz + gz_b - v * roll_ang_vel + u * pitch_ang_vel

        # Roll equation

        roll_accel = (J_xz * (J_xx - J_yy + J_zz) * roll_ang_vel * pitch_ang_vel - 
                   (J_zz * (J_zz - J_yy) + J_xz**2) * pitch_ang_vel * yaw_ang_vel + 
                   J_zz * lx + 
                   J_xz * lz) / Den
        
        # Pitch equation

        pitch_accel = ((J_zz - J_xx) * roll_ang_vel * yaw_ang_vel - 
                    J_xz * (roll_ang_vel**2 - yaw_ang_vel**2) + ly) / J_yy
        
        # Yaw equation

        yaw_accel = ((J_xx * (J_xx - J_yy) + J_xz**2) * roll_ang_vel * pitch_ang_vel + 
                  J_xz * (J_xx - J_yy + J_zz) * pitch_ang_vel * yaw_ang_vel + 
                  J_xz * lx + 
                  J_xz * lz) / Den
        
        # Kinematic equations
        
        roll_angle_rate_eq = roll_ang_vel + np.sin(roll) * np.tan(pitch) * pitch_ang_vel + \
                             np.cos(roll) * np.tan(pitch) * yaw_ang_vel
        
        pitch_angle_rate_eq = np.cos(roll) * pitch_ang_vel - np.sin(roll) * yaw_ang_vel

        yaw_angle_rate_eq = np.sin(roll) / np.cos(pitch) * pitch_ang_vel + \
                            np.cos(roll) / np.cos(pitch) * yaw_ang_vel

        # Position equations

        dx_dt = np.cos(pitch) * np.cos(yaw) * u + \
                (-np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw)) * v + \
                (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)) * w
        
        dy_dt = np.cos(pitch) * np.sin(yaw) * u + \
                (np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw)) * v + \
                (-np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) * w
        
        dz_dt = -np.sin(pitch) * u + \
                np.sin(roll) * np.cos(pitch) * v + \
                np.cos(roll) * np.cos(pitch) * w

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
        
    
    def compute_partials(self, inputs, J):
        mass = inputs['mass']
        u = inputs['u'] # u
        v = inputs['v'] # v
        w = inputs['w'] # w
        roll_ang_vel = inputs['roll_ang_vel'] # p
        pitch_ang_vel = inputs['pitch_ang_vel'] # q
        yaw_ang_vel = inputs['yaw_ang_vel'] # r
        roll = inputs['roll'] # phi
        pitch = inputs['pitch'] # theta
        yaw = inputs['yaw'] # psi
        x = inputs['x'] # p1
        y = inputs['y'] # p2
        z = inputs['z'] # p3
        # time = inputs['time']
        g = inputs['g']
        Fx = inputs['Fx']
        Fy = inputs['Fy']
        Fz = inputs['Fz']
        lx = inputs['lx'] # l
        ly = inputs['ly'] # m
        lz = inputs['lz'] # n
        J_xz = inputs['J_xz']
        J_xx = inputs['J_xx']
        J_yy = inputs['J_yy']
        J_zz = inputs['J_zz']

        # for roll and yaw
        Den = J_xx * J_zz - J_xz**2

        J['dx_accel', 'mass'] = -Fx / mass**2
        J['dx_accel', 'Fx'] = 1 / mass
        J['dx_accel', 'v'] = yaw_ang_vel
        J['dx_accel', 'w'] = -pitch_ang_vel
        J['dx_accel', 'yaw_ang_vel'] = v
        J['dx_accel', 'pitch_ang_vel'] = -w
        J['dx_accel', 'g'] = -np.sin(pitch)
        J['dx_accel', 'pitch'] = -np.cos(pitch) * g

        J['dy_accel', 'mass'] = -Fy / mass**2
        J['dy_accel', 'Fy'] = 1 / mass
        J['dy_accel', 'u'] = -yaw_ang_vel
        J['dy_accel', 'w'] = roll_ang_vel
        J['dy_accel', 'yaw_ang_vel'] = -u
        J['dy_accel', 'roll_ang_vel'] = w
        J['dy_accel', 'g'] = np.sin(roll) * np.cos(pitch)
        J['dy_accel', 'roll'] = np.cos(roll) * np.cos(pitch) * g
        J['dy_accel', 'pitch'] = -np.sin(roll) * np.sin(pitch) * g

        J['dz_accel', 'mass'] = -Fz / mass**2
        J['dz_accel', 'Fz'] = 1 / mass
        J['dz_accel', 'v'] = -roll_ang_vel
        J['dz_accel', 'u'] = pitch_ang_vel
        J['dz_accel', 'roll_ang_vel'] = -v
        J['dz_accel', 'pitch_ang_vel'] = u
        J['dz_accel', 'g'] = np.cos(roll) * np.cos(pitch)
        J['dz_accel', 'roll'] = -np.sin(roll) * np.cos(pitch) * g
        J['dz_accel', 'pitch'] = -np.cos(roll) * np.sin(pitch) * g

        J['roll_accel', 'J_xz'] = (Den * (
            (J_xx - J_yy + J_zz) * roll_ang_vel * pitch_ang_vel - 2 * J_xz * pitch_ang_vel * yaw_ang_vel + lz) - (
                J_xz * (J_xx - J_yy + J_zz) * roll_ang_vel * pitch_ang_vel - 
                   (J_zz * (J_zz - J_yy) + J_xz**2) * pitch_ang_vel * yaw_ang_vel + 
                   J_zz * lx + 
                   J_xz * lz) * -2 * J_xz) / Den**2
        J['roll_accel', 'J_xx'] = (Den * (
            J_xz * roll_ang_vel * pitch_ang_vel
        ) - (J_xz * (J_xx - J_yy + J_zz) * roll_ang_vel * pitch_ang_vel - 
                   (J_zz * (J_zz - J_yy) + J_xz**2) * pitch_ang_vel * yaw_ang_vel + 
                   J_zz * lx + 
                   J_xz * lz) * J_zz) / Den**2
        J['roll_accel', 'J_yy'] = (-J_xz * roll_ang_vel * pitch_ang_vel + J_zz * pitch_ang_vel * yaw_ang_vel) / Den
        J['roll_accel', 'J_zz'] = (Den * (
            J_xz * roll_ang_vel * pitch_ang_vel - 2 * J_zz * pitch_ang_vel * yaw_ang_vel + J_yy * pitch_ang_vel * yaw_ang_vel + lx
        ) - (J_xz * (J_xx - J_yy + J_zz) * roll_ang_vel * pitch_ang_vel - 
                   (J_zz * (J_zz - J_yy) + J_xz**2) * pitch_ang_vel * yaw_ang_vel + 
                   J_zz * lx + 
                   J_xz * lz) * J_xx) / Den**2
        J['roll_accel', 'roll_ang_vel'] = (J_xz * (J_xx - J_yy + J_zz) * pitch_ang_vel) / Den
        J['roll_accel', 'pitch_ang_vel'] = (J_xz * (J_xx - J_yy + J_zz) * roll_ang_vel - (J_zz * (J_zz - J_yy) + J_xz**2) * yaw_ang_vel) / Den
        J['roll_accel', 'yaw_ang_vel'] = -((J_zz * (J_zz - J_yy) + J_xz**2) * pitch_ang_vel) / Den
        J['roll_accel', 'lx'] = J_zz / Den
        J['roll_accel', 'lz'] = J_xz / Den

        J['pitch_accel', 'J_xz'] = -(roll_ang_vel**2 - yaw_ang_vel**2) / J_yy
        J['pitch_accel', 'J_xx'] = -(roll_ang_vel * yaw_ang_vel) / J_yy
        J['pitch_accel', 'J_yy'] = -((J_zz - J_xx) * roll_ang_vel * yaw_ang_vel - 
                    J_xz * (roll_ang_vel**2 - yaw_ang_vel**2) + ly) / J_yy**2
        J['pitch_accel', 'J_zz'] = roll_ang_vel * yaw_ang_vel / J_yy
        J['pitch_accel', 'roll_ang_vel'] = ((J_zz - J_xx) * yaw_ang_vel - 2 * J_xz * roll_ang_vel) / J_yy
        J['pitch_accel', 'yaw_ang_vel'] = ((J_zz - J_xx) * roll_ang_vel + 2 * J_xz * yaw_ang_vel) / J_yy
        J['pitch_accel', 'ly'] = 1 / J_yy

        J['yaw_accel', 'J_xz'] = (Den * (
            2 * J_xz * roll_ang_vel * pitch_ang_vel + (J_xx - J_yy + J_zz) * pitch_ang_vel * yaw_ang_vel + lx + lz
        ) - ((J_xx * (J_xx - J_yy) + J_xz**2) * roll_ang_vel * pitch_ang_vel + 
                  J_xz * (J_xx - J_yy + J_zz) * pitch_ang_vel * yaw_ang_vel + 
                  J_xz * lx + 
                  J_xz * lz) * -2 * J_xz) / Den**2
        J['yaw_accel', 'J_xx'] = (Den * (
            2 * J_xx * roll_ang_vel * pitch_ang_vel - J_yy * roll_ang_vel * pitch_ang_vel + J_xz * pitch_ang_vel * yaw_ang_vel
        ) - ((J_xx * (J_xx - J_yy) + J_xz**2) * roll_ang_vel * pitch_ang_vel + 
                  J_xz * (J_xx - J_yy + J_zz) * pitch_ang_vel * yaw_ang_vel + 
                  J_xz * lx + 
                  J_xz * lz) * J_zz) / Den**2
        J['yaw_accel', 'J_yy'] = (-J_xx * roll_ang_vel * pitch_ang_vel - J_xz * pitch_ang_vel * yaw_ang_vel) / Den
        J['yaw_accel', 'J_zz'] = (Den * (
            J_xz * pitch_ang_vel * yaw_ang_vel
        ) - ((J_xx * (J_xx - J_yy) + J_xz**2) * roll_ang_vel * pitch_ang_vel + 
                  J_xz * (J_xx - J_yy + J_zz) * pitch_ang_vel * yaw_ang_vel + 
                  J_xz * lx + 
                  J_xz * lz) * J_xx) / Den**2
        J['yaw_accel', 'roll_ang_vel'] = ((J_xx * (J_xx - J_yy) + J_xz**2) * pitch_ang_vel) / Den
        J['yaw_accel', 'pitch_ang_vel'] = ((J_xx * (J_xx - J_yy) + J_xz**2) * roll_ang_vel + J_xz * (J_xx - J_yy + J_zz) * yaw_ang_vel) / Den
        J['yaw_accel', 'yaw_ang_vel'] = (J_xz * (J_xx - J_yy + J_zz) * pitch_ang_vel) / Den
        J['yaw_accel', 'lx'] = J_xz / Den
        J['yaw_accel', 'lz'] = J_xz / Den

        J['roll_angle_rate_eq', 'roll_ang_vel'] = 1 
        J['roll_angle_rate_eq', 'pitch_ang_vel'] = np.sin(roll) * np.tan(pitch)
        J['roll_angle_rate_eq', 'yaw_ang_vel'] = np.cos(roll) * np.tan(pitch)
        J['roll_angle_rate_eq', 'roll'] = np.cos(roll) * np.tan(pitch) * pitch_ang_vel - np.sin(roll) * np.tan(pitch) * yaw_ang_vel
        J['roll_angle_rate_eq', 'pitch'] = np.sin(roll) * (1 / np.cos(pitch)**2) * pitch_ang_vel + np.cos(roll) * (1 / np.cos(pitch)**2) * yaw_ang_vel
        
        J['pitch_angle_rate_eq', 'pitch_ang_vel'] = np.cos(roll)
        J['pitch_angle_rate_eq', 'yaw_ang_vel'] = -np.sin(roll)
        J['pitch_angle_rate_eq', 'roll'] = -np.sin(roll) * pitch_ang_vel - np.cos(roll) * yaw_ang_vel

        J['yaw_angle_rate_eq', 'pitch_ang_vel'] = np.sin(roll) / np.cos(pitch)
        J['yaw_angle_rate_eq', 'yaw_ang_vel'] = np.cos(roll) / np.cos(pitch)
        J['yaw_angle_rate_eq', 'roll'] = np.cos(roll) / np.cos(pitch) * pitch_ang_vel - np.sin(roll) / np.cos(pitch) * yaw_ang_vel
        J['yaw_angle_rate_eq', 'pitch'] = np.sin(roll) * (np.tan(pitch) / np.cos(pitch)) * pitch_ang_vel + np.cos(roll) * (np.tan(pitch) / np.cos(pitch)) * yaw_ang_vel

        # note: d/dx tan(x) = sec^2(x) = 1 / cos^2(x)
        # note: d/dx 1 / cos(x) = d/dx sec(x) = sec(x)tan(x) = tan(x) / cos(x)

        J['dx_dt', 'u'] = np.cos(pitch) * np.cos(yaw)
        J['dx_dt', 'v'] = -np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw)
        J['dx_dt', 'w'] = np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)
        J['dx_dt', 'roll'] = (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)) * v + \
                             (np.cos(roll) * np.sin(yaw) - np.sin(roll) * np.sin(pitch) * np.cos(yaw)) * w
        J['dx_dt', 'pitch'] = -np.sin(pitch) * np.cos(yaw) * u + \
                              np.sin(roll) * np.cos(pitch) * np.cos(yaw) * v + \
                              np.cos(roll) * np.cos(pitch) * np.cos(yaw) * w
        J['dx_dt', 'yaw'] = -np.cos(pitch) * np.sin(yaw) * u + \
                            (-np.cos(roll) * np.cos(yaw) - np.sin(roll) * np.sin(pitch) * np.sin(yaw)) * v + \
                            (np.sin(roll) * np.cos(yaw) - np.cos(roll) * np.sin(pitch) * np.sin(yaw)) * w
        
        J['dy_dt', 'u'] = np.cos(pitch) * np.sin(yaw)
        J['dy_dt', 'v'] = np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw)
        J['dy_dt', 'w'] = -np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)
        J['dy_dt', 'roll'] = (-np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) * v + \
                             (-np.cos(roll) * np.cos(yaw) - np.sin(roll) * np.sin(pitch) * np.sin(yaw)) * w
        J['dy_dt', 'pitch'] = -np.sin(pitch) * np.sin(yaw) * u + \
                              np.sin(roll) * np.cos(pitch) * np.sin(yaw) * v + \
                              np.cos(roll) * np.cos(pitch) * np.sin(yaw) * w
        J['dy_dt', 'yaw'] = np.cos(pitch) * np.cos(yaw) * u + \
                            (-np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw)) * v + \
                            (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)) * w
        
        J['dz_dt', 'u'] = -np.sin(pitch)
        J['dz_dt', 'v'] = np.sin(roll) * np.cos(pitch)
        J['dz_dt', 'w'] = np.cos(roll) * np.cos(pitch)
        J['dz_dt', 'roll'] = np.cos(roll) * np.cos(pitch) * v - \
                             np.sin(roll) * np.cos(pitch) * w
        J['dz_dt', 'pitch'] = -np.cos(pitch) * u - \
                              np.sin(roll) * np.sin(pitch) * v - \
                              np.cos(roll) * np.sin(pitch) * w
                             




if __name__ == "__main__":

    p = om.Problem()
    p.model = om.Group()
    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])

    des_vars.add_output('mass', 3.0, units='kg')
    des_vars.add_output('u', 0.1, units='m/s')
    des_vars.add_output('v', 0.7, units='m/s')
    des_vars.add_output('w', 0.12, units='m/s')
    des_vars.add_output('roll_ang_vel', 0.1, units='rad/s')
    des_vars.add_output('pitch_ang_vel', 0.9, units='rad/s')
    des_vars.add_output('yaw_ang_vel', 0.12, units='rad/s')
    des_vars.add_output('roll', 0.9, units='rad')
    des_vars.add_output('pitch', 0.19, units='rad')
    des_vars.add_output('yaw', 0.70, units='rad')
    des_vars.add_output('g', 9.81, units='m/s**2')
    des_vars.add_output('Fx', 0.1, units='N')
    des_vars.add_output('Fy', 0.9, units='N')
    des_vars.add_output('Fz', 0.12, units='N')
    des_vars.add_output('lx', 3.0, units='N*m')
    des_vars.add_output('ly', 4.0, units='N*m')
    des_vars.add_output('lz', 5.0, units='N*m')
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

    p.check_partials(compact_print=True, show_only_incorrect=True, method='cs')


