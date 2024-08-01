import openmdao.api as om
import dymos as dm
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
import matplotlib.pyplot as plt
import numpy as np
import sys


class MinTimeClimbProblem(om.Problem):
    def __init__(self, target_height=20e3, initial_mass=20e3):
        super().__init__()
        self.model = om.Group()
        self.target_height = target_height
        self.initial_mass = initial_mass

    def setOptimizer(self, driver='scipy', optimizer='SLSQP'):
        self.driver = om.pyOptSparseDriver() if driver == "pyoptsparse" else om.ScipyOptimizeDriver()
        self.driver.options['optimizer'] = optimizer
        self.driver.declare_coloring()
        if optimizer == 'SNOPT':
            self.driver.opt_settings['Major iterations limit'] = 1000
            self.driver.opt_settings['iSumm'] = 6
            self.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            self.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            self.driver.opt_settings['Function precision'] = 1.0E-12
            self.driver.opt_settings['Linesearch tolerance'] = 0.1
            self.driver.opt_settings['Major step limit'] = 0.5
        elif optimizer == 'IPOPT':
            self.driver.opt_settings['tol'] = 1.0E-5
            self.driver.opt_settings['print_level'] = 0
            self.driver.opt_settings['mu_strategy'] = 'monotone'
            self.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
            self.driver.opt_settings['mu_init'] = 0.01

        self.model.linear_solver = om.DirectSolver()

    def addTrajectory(self, num_seg=9, transcription='gauss-lobatto',
                      transcription_order=3):
        t = {'gauss-lobatto': dm.GaussLobatto(
            num_segments=num_seg, order=transcription_order),
            'radau-ps': dm.Radau(num_segments=num_seg, order=transcription_order)}

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=MinTimeClimbODE, transcription=t[transcription])
        traj.add_phase('phase0', phase)

        height = self.target_height
        time_name = 'time'
        add_rate = False
        phase.set_time_options(fix_initial=True, duration_bounds=(50, 600),
                               duration_ref=100.0, name=time_name)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6,
                        ref=1.0E3, defect_ref=1.0E3, units='m',
                        rate_source='flight_dynamics.r_dot')

        phase.add_state('h', fix_initial=True, lower=0, upper=height,
                        ref=height, defect_ref=height, units='m',
                        rate_source='flight_dynamics.h_dot', targets=['h'])

        phase.add_state('v', fix_initial=True, lower=10.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m/s',
                        rate_source='flight_dynamics.v_dot', targets=['v'])

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5,
                        ref=1.0, defect_ref=1.0, units='rad',
                        rate_source='flight_dynamics.gam_dot', targets=['gam'])

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5,
                        ref=10_000, defect_ref=10_000, units='kg',
                        rate_source='prop.m_dot', targets=['m'])

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=True, rate_continuity_scaler=100.0,
                          rate2_continuity=False, targets=['alpha'])

        phase.add_parameter('S', val=49.2386, units='m**2', opt=True, targets=['S'])
        phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
        phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        phase.add_boundary_constraint(
            'h', loc='final', equals=height)  # , scaler=1.0E-3)
        phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
        phase.add_boundary_constraint('gam', loc='final', equals=0.0)

        phase.add_path_constraint(name='h', lower=100.0, upper=height, ref=height)
        phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

        # Unnecessary but included to test capability
        phase.add_path_constraint(name='alpha', lower=-8, upper=8)
        phase.add_path_constraint(name=f'{time_name}', lower=0, upper=400)
        phase.add_path_constraint(name=f'{time_name}_phase', lower=0, upper=400)

        # Minimize time at the end of the phase
        # phase.add_objective(time_name, loc='final', ref=1.0)

        # test mixing wildcard ODE variable expansion and unit overrides
        phase.add_timeseries_output(['aero.*', 'prop.thrust', 'prop.m_dot'],
                                    units={'aero.f_lift': 'lbf', 'prop.thrust': 'lbf'})

        # test adding rate as timeseries output
        if add_rate:
            phase.add_timeseries_rate_output('aero.mach')

        self.phase = phase
        self.model.add_subsystem('traj', traj, promotes=['*'])

    def setInitialConditions(self, super_prob=None, prefix=""):
        ref = self
        if super_prob is not None and prefix != "":
            ref = super_prob
        ref[prefix+'traj.phase0.t_initial'] = 0.0
        ref[prefix+'traj.phase0.t_duration'] = 350.0
        ref[prefix+'traj.phase0.states:r'] = self.phase.interp('r', [0.0, 100e3])
        ref[prefix+'traj.phase0.states:h'] = self.phase.interp(
            'h', [100.0, self.target_height])
        ref[prefix+'traj.phase0.states:v'] = self.phase.interp('v', [135.964, 283.159])
        ref[prefix+'traj.phase0.states:gam'] = self.phase.interp('gam', [0.0, 0.0])
        ref[prefix+'traj.phase0.states:m'] = self.phase.interp(
            'm', [self.initial_mass, 10e3])
        ref[prefix+'traj.phase0.controls:alpha'] = self.phase.interp('alpha', [0.0, 0.0])


if __name__ == '__main__':
    makeN2 = True if "n2" in sys.argv else False
    heights = [6e3, 18e3]
    weights = [1, 1]
    num_missions = len(heights)
    super_prob = om.Problem()
    probs = []
    for i, height in enumerate(heights):
        prob = MinTimeClimbProblem(height, 30e3)
        # prob.setOptimizer()
        prob.addTrajectory()
        super_prob.model.add_subsystem(
            f"group_{i}", prob.model, promotes=[('phase0.parameters:S', 'S')])
        probs.append(prob)

    super_prob.model.add_design_var('S', lower=1, upper=100, units='m**2')

    times = [f"time_{i}" for i in range(num_missions)]
    weighted_sum_str = "+".join([f"{time}*{weight}" for time,
                                weight in zip(times, weights)])
    super_prob.model.add_subsystem('compoundComp', om.ExecComp(
        "compound_time=" + weighted_sum_str),
        promotes=['compound_time', *times])

    for i in range(num_missions):
        super_prob.model.connect(f"group_{i}.phase0.t", times[i], src_indices=-1)
    super_prob.model.add_objective('compound_time')

    super_prob.driver = om.ScipyOptimizeDriver()
    super_prob.driver.options['optimizer'] = 'SLSQP'
    super_prob.driver.declare_coloring()
    super_prob.model.linear_solver = om.DirectSolver()

    super_prob.setup()
    if makeN2:
        sys.path.append('../')
        from createN2 import createN2
        createN2(__file__, super_prob)
    for i, prob in enumerate(probs):
        prob.setInitialConditions(super_prob, f"group_{i}.")

    dm.run_problem(super_prob)
    wing_area = super_prob.get_val('S', units='m**2')[0]
    # sol = om.CaseReader('dymos_solution.db').get_case('final')
    # sim = om.CaseReader('dymos_simulation.db').get_case('final')
    print("\n\n=====================================")
    for i in range(num_missions):
        timetoclimb = super_prob.get_val(f'group_{i}.phase0.t', units='s')[-1]
        print(f"TtoC: {timetoclimb}, S: {wing_area}")
        xsol = super_prob.get_val(f"group_{i}.phase0.timeseries.r")
        ysol = super_prob.get_val(f"group_{i}.phase0.timeseries.h")
        plt.plot(xsol, ysol)

    plt.grid()
    plt.show()
