from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import os
import numpy as np

import unittest
import openmdao.api as om
import dymos as dm

from dymos.examples.plotting import plot_results

from openmdao.api import Group, IndepVarComp, ExecComp
from dymos.transcriptions.transcription_base import TranscriptionBase

from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.sixdof.six_dof_EOM import SixDOF_EOM
from aviary.mission.sixdof.force_component_calc import ForceComponentResolver

from openmdao.utils.assert_utils import assert_check_partials

from openmdao.utils.general_utils import set_pyoptsparse_opt
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

class vtolODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('USatm1976comp', USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['rho'])
        
        self.add_subsystem('ForceComponents', ForceComponentResolver(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('SixDOF_EOM', SixDOF_EOM(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
# class main_phase_class(dm.Phase):

#     def initialize(self):
#         super(main_phase_class, self).initialize()
#         self.options.declare('interp_to', types=TranscriptionBase)
    
#     def setup(self):
#         self.options['ode_class'] = vtolODE

#         super(main_phase_class, self).setup()

def sixdof_test():
    p = om.Problem()

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=vtolODE, 
                    transcription=dm.Radau(num_segments=10, order=3)
                    )
    
    p.model.add_subsystem('traj', traj)

    # SETUP

    traj.add_phase(name='main_phase', phase=phase)

    phase.set_time_options(fix_initial=True, fix_duration=False, units='s')
    
    phase.add_state('axial_vel', fix_initial=True, rate_source='dx_accel', targets=['axial_vel'], lower=0, upper=10, units='m/s')
    phase.add_state('lat_vel', fix_initial=True, rate_source='dy_accel', targets=['lat_vel'], lower=0, upper=10, units='m/s')
    phase.add_state('vert_vel', fix_initial=True, rate_source='dz_accel', targets=['vert_vel'], lower=0, upper=10, units='m/s')
    phase.add_state('roll_ang_vel', fix_initial=True, rate_source='roll_accel', targets=['roll_ang_vel'], lower=0, upper=10, units='rad/s')
    phase.add_state('pitch_ang_vel', fix_initial=True, rate_source='pitch_accel', targets=['pitch_ang_vel'], lower=0, upper=10, units='rad/s')
    phase.add_state('yaw_ang_vel', fix_initial=True, rate_source='yaw_accel', targets=['yaw_ang_vel'], lower=0, upper=10, units='rad/s')
    phase.add_state('roll', fix_initial=True, rate_source='roll_angle_rate_eq', targets=['roll'], lower=0, upper=np.pi, units='rad')
    phase.add_state('pitch', fix_initial=True, rate_source='pitch_angle_rate_eq', targets=['pitch'], lower=0, upper=np.pi, units='rad')
    phase.add_state('yaw', fix_initial=True, rate_source='yaw_angle_rate_eq', targets=['yaw'], lower=0, upper=np.pi, units='rad')
    phase.add_state('x', fix_initial=True, fix_final=True, rate_source='dx_dt', targets=['x'],lower=0, upper=100, units='m')
    phase.add_state('y', fix_initial=True, fix_final=True, rate_source='dy_dt', targets=['y'], lower=0, upper=100, units='m')
    phase.add_state('z', fix_initial=True, rate_source='dz_dt', targets=['z'], lower=0, upper=100, units='m')

    phase.add_control('Fx_ext', targets=['Fx_ext'], units='N')
    phase.add_control('Fy_ext', targets=['Fy_ext'], units='N')
    phase.add_control('Fz_ext', targets=['Fz_ext'], units='N')
    phase.add_control('lx_ext', targets=['lx_ext'], units='N*m')
    phase.add_control('ly_ext', targets=['ly_ext'], units='N*m')
    phase.add_control('lz_ext', targets=['lz_ext'], units='N*m')

    phase.add_parameter('mass', units='kg', targets=['mass'], opt=False)
    phase.add_parameter('J_xx', units='kg * m**2', targets=['J_xx'], opt=False)
    phase.add_parameter('J_yy', units='kg * m**2', targets=['J_yy'], opt=False)
    phase.add_parameter('J_zz', units='kg * m**2', targets=['J_zz'], opt=False)
    phase.add_parameter('J_xz', units='kg * m**2', targets=['J_xz'], opt=False)
   
    
    phase.add_objective('Fz_ext', loc='final')

    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = "IPOPT"

    p.driver.opt_settings['mu_init'] = 1e-1
    p.driver.opt_settings['max_iter'] = 600
    p.driver.opt_settings['constr_viol_tol'] = 1e-6
    p.driver.opt_settings['compl_inf_tol'] = 1e-6
    p.driver.opt_settings['tol'] = 1e-5
    p.driver.opt_settings['print_level'] = 0
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
    p.driver.opt_settings['mu_strategy'] = 'monotone'
    p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
    p.driver.options['print_results'] = True

    p.driver.declare_coloring()

    p.setup()

    phase.set_time_val(initial=0, duration=60, units='s')
    phase.set_state_val('axial_vel', vals=[0, 0], units='m/s')
    phase.set_state_val('lat_vel', vals=[0, 0], units='m/s')
    phase.set_state_val('vert_vel', vals=[0, 10], units='m/s')
    phase.set_state_val('roll_ang_vel', vals=[0, 10], units='rad/s')
    phase.set_state_val('pitch_ang_vel', vals=[0, 10], units='rad/s')
    phase.set_state_val('yaw_ang_vel', vals=[0, 10], units='rad/s')
    phase.set_state_val('roll', vals=[0, 0], units='rad')
    phase.set_state_val('pitch', vals=[0, 0], units='rad')
    phase.set_state_val('yaw', vals=[0, 0], units='rad')
    phase.set_state_val('x', vals=[0, 0], units='m')
    phase.set_state_val('y', vals=[0, 0], units='m')
    phase.set_state_val('z', vals=[0, 100], units='m')

    phase.set_control_val('Fx_ext', vals=[0, 0], units='N')
    phase.set_control_val('Fy_ext', vals=[0, 0], units='N')
    phase.set_control_val('Fz_ext', vals=[200, 200], units='N')
    phase.set_control_val('lx_ext', vals=[0, 0], units='N*m')
    phase.set_control_val('ly_ext', vals=[0, 0], units='N*m')
    phase.set_control_val('lz_ext', vals=[0, 0], units='N*m')

    phase.set_parameter_val('mass', val=10, units='kg')
    phase.set_parameter_val('J_xx', val=16, units='kg*m**2') # assume a sphere of 10 kg with radius = 2
    phase.set_parameter_val('J_yy', val=16, units='kg*m**2')
    phase.set_parameter_val('J_zz', val=16, units='kg*m**2')
    phase.set_parameter_val('J_xz', val=0, units='kg*m**2')
    
    phase.add_path_constraint('x', equals=0)
    phase.add_path_constraint('y', equals=0)
    

    p.final_setup()

    p.run_model()

    dm.run_problem(p, run_driver=True, simulate=True, make_plots=True)
    #print(p.get_reports_dir())

    exp_out = traj.simulate()

    plot_results([('traj.main_phase.timeseries.x', 'traj.main_phase.timeseries.z', 
                   'x (m)', 'z (m)'),
                   ('traj.main_phase.timeseries.y', 'traj.main_phase.timeseries.z',
                    'y (m)', 'z (m)')],
                    title='Trajectory of Vertical Take Off',
                    p_sol=p, p_sim=exp_out)

    plt.show()
    plt.savefig('./TrajPlots.pdf')


if __name__ == "__main__":
    sixdof_test() 




