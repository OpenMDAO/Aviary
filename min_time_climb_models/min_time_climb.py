import openmdao.api as om
import dymos as dm
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
from plot_helper import make_min_time_climb_plot
import matplotlib.pyplot as plt
import numpy as np
import sys


def min_time_climb(
        height=20e3, driver='scipy', optimizer='SLSQP', num_seg=15,
        transcription='gauss-lobatto', transcription_order=3, force_alloc_complex=False,
        add_rate=False, time_name='time', simfile='dymos_simulatin.db',
        solfile='dymos_solution.db'):

    p = om.Problem(model=om.Group())

    p.driver = om.ScipyOptimizeDriver() if driver == 'scipy' else om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()

    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['Function precision'] = 1.0E-12
        p.driver.opt_settings['Linesearch tolerance'] = 0.1
        p.driver.opt_settings['Major step limit'] = 0.5
    elif optimizer == 'IPOPT':
        p.driver.opt_settings['tol'] = 1.0E-5
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        p.driver.opt_settings['mu_init'] = 0.01

    t = {'gauss-lobatto': dm.GaussLobatto(
        num_segments=num_seg, order=transcription_order),
        'radau-ps': dm.Radau(num_segments=num_seg, order=transcription_order)}

    traj = dm.Trajectory()

    phase = dm.Phase(ode_class=MinTimeClimbODE, transcription=t[transcription])
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
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

    phase.add_parameter('S', val=42.554663555518836,
                        units='m**2', opt=True, targets=['S'])
    phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
    phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

    phase.add_boundary_constraint('h', loc='final', equals=height)  # , scaler=1.0E-3)
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0)

    phase.add_path_constraint(name='h', lower=100.0, upper=height, ref=height)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
    # phase.add_path_constraint(name='gam', lower=-23, upper=23, units='deg')

    # Unnecessary but included to test capability
    phase.add_path_constraint(name='alpha', lower=-8, upper=8)
    phase.add_path_constraint(name=f'{time_name}', lower=0, upper=400)
    phase.add_path_constraint(name=f'{time_name}_phase', lower=0, upper=400)

    # Minimize time at the end of the phase
    phase.add_objective(time_name, loc='final', ref=1.0)

    # test mixing wildcard ODE variable expansion and unit overrides
    phase.add_timeseries_output(['aero.*', 'prop.thrust', 'prop.m_dot'],
                                units={'aero.f_lift': 'lbf', 'prop.thrust': 'lbf'})

    # test adding rate as timeseries output
    if add_rate:
        phase.add_timeseries_rate_output('aero.mach')

    p.model.add_subsystem('traj', traj, promotes=['*'])

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=force_alloc_complex)

    p.set_val('phase0.t_initial', 0.0)
    p.set_val('phase0.t_duration', 350.0)

    p.set_val('phase0.states:r', phase.interp('r', [0.0, 100e3]))
    p.set_val('phase0.states:h', phase.interp('h', [100.0, height]))
    p.set_val('phase0.states:v', phase.interp('v', [135.964, 283.159]))
    p.set_val('phase0.states:gam', phase.interp('gam', [0.0, 0.0]))
    p.set_val('phase0.states:m', phase.interp('m', [30e3, 16e3]))
    p.set_val('phase0.controls:alpha', phase.interp('alpha', [0.0, 0.0]))

    dm.run_problem(p, simulate=True, simulation_record_file=simfile,
                   solution_record_file=solfile)

    return p


def checkDeviation(filenum=0):
    """Function to run min time climb problem multiple times for the same height and report any deviation in results.
       Made to check for random results in outputs, fixed 7/31, seems to be caused by transcription order being 5 not 3"""
    heights = [10e3]*2
    times_to_climb = []
    timeseries_pts = {'h': [], 'r': [], 'thrust': [], 'v': []}
    prefix = 'traj.phase0.timeseries.'

    for height in heights:
        p = min_time_climb(height=height)
        sol = om.CaseReader('dymos_solution.db').get_case('final')
        sim = om.CaseReader('dymos_simulation.db').get_case('final')
        times_to_climb.append(p.get_val('traj.phase0.timeseries.time',
                                        units='s')[-1][0])
        for var, varlst in timeseries_pts.items():
            varlst.append(sol.get_val(prefix+var))

    print("\n\n=======================================")
    with open(f'checkdiff_{filenum}.txt', 'w') as fp:
        fp.write(f'time to climb: {times_to_climb[0]: .4f}\n\n')
        fp.write(str(timeseries_pts))

    for key in timeseries_pts.keys():
        timeseries_pts[key] = np.array(
            timeseries_pts[key][1:]) - timeseries_pts[key][0]
        print(f"Max difference in {key}: {np.max(timeseries_pts[key]):.2f}")

    print(f"Time to climb: {times_to_climb[0]:.2f}")
    print(f"Standard deviation of time to climb in {len(heights)}" +
          f" cases: {np.std(times_to_climb):.2f}")
    for key in timeseries_pts.keys():
        print(f"Standard deviation of {key} in {len(heights)} cases:" +
              f" {np.std(timeseries_pts[key]):.2f}")


def multiHeightTest():
    heights = [6e3, 18e3]
    times = []
    areas = []
    outfiles = {'sol': [], 'sim': []}

    for j, height in enumerate(heights):
        solfile = f'single_solution_{j}.db'
        simfile = f'single_simulation_{j}.db'
        outfiles['sim'].append(simfile)
        outfiles['sol'].append(solfile)
        p = min_time_climb(
            height=height, simfile=simfile,
            solfile=solfile)
        areas.append(p.get_val('phase0.parameters:S', units='m**2')[0])
        times.append(p.get_val('phase0.timeseries.time', units='s')[-1][0])

    print("\n\n=======================================")
    # print(p.get_val('phase.flight_dynamics.gam_dot'))
    for time, area, height in zip(times, areas, heights):
        print(f"Time to climb {height/1e3} km: " +
              f"{time:.2f}" +
              f" s with wing area: {area} sqm")
    make_min_time_climb_plot(
        solfile=outfiles['sol'],
        simfile=outfiles['sim'],
        omitpromote='traj')


if __name__ == '__main__':
    if 'filenum' in sys.argv:
        checkDeviation(filenum=sys.argv[1].split('filenum=')[1])
    else:
        multiHeightTest()


"""
var names
       'traj.phase0.controls:alpha'
       'traj.phase0.t_initial'
       'traj.phase0.t_duration'
       'traj.phase0.parameters:S'
       'traj.phase0.parameters:Isp'
       'traj.phase0.parameters:throttle'
       'traj.phase0.collocation_constraint.defects:gam'
       'traj.phase0.control_rates:alpha_rate'
       'traj.phase0.control_rates:alpha_rate2'
       'traj.phase0.control_values:alpha'
       'traj.phase0.states:gam'
       'traj.phase0.states:h'       
       'traj.phase0.states:m'
       'traj.phase0.states:r'
       'traj.phase0.states:v'
       'traj.phase0.interleave_comp.all_values:CD'
       'traj.phase0.parameter_vals:Isp'
       'traj.phase0.parameter_vals:S'
       'traj.phase0.parameter_vals:throttle'
       'traj.phase0.t_duration_val'
       'traj.phase0.t_initial_val'
       'traj.phase0.rhs_col.aero.CD'
       'traj.phase0.rhs_disc.aero.CL'
       'traj.phase0.rhs_disc.aero.CD0'
       'traj.phase0.rhs_disc.aero.CLa'
       'traj.phase0.rhs_disc.aero.kappa'
       'traj.phase0.rhs_disc.aero.f_drag'
       'traj.phase0.rhs_disc.aero.f_lift' 
       'traj.phase0.rhs_disc.aero.mach'
       'traj.phase0.rhs_disc.aero.q'
       'traj.phase0.rhs_disc.atmos.drhos_dh'
       'traj.phase0.rhs_disc.atmos.pres'
       'traj.phase0.rhs_disc.atmos.rho' 
       'traj.phase0.rhs_disc.atmos.sos' 
       'traj.phase0.rhs_disc.atmos.temp' 
       'traj.phase0.rhs_disc.atmos.viscosity'
       'traj.phase0.rhs_disc.flight_dynamics.gam_dot'
       'traj.phase0.rhs_disc.flight_dynamics.h_dot'
       'traj.phase0.rhs_disc.flight_dynamics.r_dot'
       'traj.phase0.rhs_disc.flight_dynamics.v_dot'
       'traj.phase0.rhs_disc.prop.max_thrust'
       'traj.phase0.rhs_disc.prop.m_dot'
       'traj.phase0.rhs_disc.prop.thrust'
       'traj.phase0.state_interp.state_col:gam': array([[ 0.0702449 ],
       'traj.phase0.state_interp.state_col:h'
       'traj.phase0.state_interp.state_col:m'
       'traj.phase0.state_interp.state_col:r'
       'traj.phase0.state_interp.state_col:v'
       'traj.phase0.dt_dstau'
       'traj.phase0.t'
       'traj.phase0.t_phase'
       'traj.phase0.timeseries.CD': array([[0.0195346 ],
       'traj.phase0.timeseries.CD0'
       'traj.phase0.timeseries.CL'
       'traj.phase0.timeseries.CLa'
       'traj.phase0.timeseries.alpha'
       'traj.phase0.timeseries.f_drag'
       'traj.phase0.timeseries.f_lift'
       'traj.phase0.timeseries.gam'
       'traj.phase0.timeseries.h'
       'traj.phase0.timeseries.kappa'
       'traj.phase0.timeseries.m'
       'traj.phase0.timeseries.m_dot'
       'traj.phase0.timeseries.mach'
       'traj.phase0.timeseries.q'
       'traj.phase0.timeseries.r'
       'traj.phase0.timeseries.thrust'
       'traj.phase0.timeseries.time'
       'traj.phase0.timeseries.time_phase' 
       'traj.phase0.timeseries.v'
       
       """
