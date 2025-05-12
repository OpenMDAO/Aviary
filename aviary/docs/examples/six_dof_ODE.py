import numpy as np
import openmdao.api as om

# Here will import the 6dof equations of motion

from aviary.mission.base_ode import BaseODE as _BaseODE

from aviary.variable_info.enums import AnalysisScheme, SpeedType, ThrottleAllocation
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

class SixDOF_ODE(_BaseODE):

    def initialize(self):
        super().initialize()

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        analysis_scheme = options['analysis_scheme']
        self.add_atmosphere(input_speed_type=SpeedType.MACH)

        self.add_subsystem(
            name='veclocity_rate_comp',
            subsys=om.ExecComp(
                'velocity_rate = mach_rate * sos',
                mach_rate = {'units': 'unitless/s', 'shape': (nn,)},
                sos={'units': 'm/s', 'shape': (nn,)},
                velocity_rate={'units': 'm/s**2', 'shape': (nn,)},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('mach_rate', Dynamic.Atmosphere.MACH_RATE),
                ('sos', Dynamic.Atmosphere.SPEED_OF_SOUND),
            ],
            promotes_outputs=[
                'velocity_rate', Dynamic.Mission.VELOCITY_RATE
            ],
        )

        sub1 = self.add_subsystem(
            'solver_sub',
            om.Group(),
            promotes=['*']
        )

        self.add_core_subsystems(solver_group=sub1)

        self.add_external_subsystems(solver_group=sub1)

        sub1.add_subsystem(
            name='SixDOF_EOM',
            subsys=SixDOF_EOM(num_nodes=nn),
            promotes_inputs=[
                'mass',
                'axial_vel',
                'lat_vel',
                'vert_vel',
                'roll_ang_vel',
                'pitch_ang_vel',
                'yaw_ang_vel',
                'roll',
                'pitch',
                'yaw',
                'g',
                'Fx_ext',
                'Fy_ext',
                'Fz_ext',
                'lx_ext',
                'ly_ext',
                'lz_ext',
                'J_xz',
                'J_xx',
                'J_yy',
                'J_zz',
            ],
            promotes_outputs=[
                'dx_accel',
                'dy_accel',
                'dz_accel',
                'roll_accel',
                'pitch_accel',
                'yaw_accel',
                'roll_angle_rate_eq',
                'pitch_angle_rate_eq',
                'yaw_angle_rate_eq',
                'dx_dt',
                'dy_dt',
                'dz_dt',
            ]
        )

        self.set_input_defaults(Dynamic.Atmosphere.MACH, val=np.ones(nn), units='unitless')
        self.set_input_defaults(Dynamic.Vehicle.MASS, val=np.ones(nn), units='kg')
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.ones(nn), units='m/s')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.ones(nn), units='m')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE_RATE, val=np.ones(nn), units='m/s')

        print_level = 0 if analysis_scheme is AnalysisScheme.SHOOTING else 2

        sub1.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        sub1.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        sub1.linear_solver = om.DirectSolver(assemble_jac=True)
        sub1.nonlinear_solver.options['err_on_non_converge'] = True
        sub1.nonlinear_solver.options['iprint'] = print_level

        self.options['auto_order'] = True